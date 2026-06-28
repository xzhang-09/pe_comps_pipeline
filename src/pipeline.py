import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import ValidationError

from src import feature_builder, fetcher, get_logger, llm_analyzer, reporter, scorer, universe_builder
from src.config_schema import PipelineConfig
from src.records import CompanyRecord

logger = get_logger(__name__)

REQUIRED_DIRECTORIES = ("data/cache", "data/checkpoints", "outputs", "logs", "eval")


def _ensure_directories() -> None:
    for directory in REQUIRED_DIRECTORIES:
        Path(directory).mkdir(parents=True, exist_ok=True)


def _load_config(config_path: str) -> PipelineConfig:
    with open(config_path, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    try:
        return PipelineConfig.model_validate(raw_config)
    except ValidationError as e:
        raise ValueError(f"Invalid {config_path}:\n{e}") from e


@dataclass(frozen=True)
class EnrichedUniverse:
    """The target-agnostic output of the enrichment layer (steps 1-4): a
    discovered, fetched, LLM-tagged comp pool plus the standardized feature
    matrix built from it. Everything here depends only on the comp universe (and
    the target's SIC codes that scoped discovery) — NOT on the target's financial
    estimates or the scorer weights — so the same EnrichedUniverse can be scored
    repeatedly against different target estimates / penalty weights without
    re-running the expensive discover→fetch→extract work. score_and_report()
    consumes it; this seam is what makes a future parameter-tuning loop (which
    re-scores many times) practical."""

    companies: list[CompanyRecord]
    llm_features: dict[str, dict]
    feature_matrix: pd.DataFrame
    ev_ebitda_raw: pd.Series
    imputation_medians: dict


UNIVERSE_SCHEMA_VERSION = 1


def save_universe(universe: EnrichedUniverse, dir_path: str | Path) -> None:
    """Persist an EnrichedUniverse to `dir_path`: parquet for the feature matrix,
    JSON for the records / LLM features / medians, plus a manifest carrying a
    schema version. Lets the expensive enrichment layer run once and be reused by
    later score-only runs (a tuning sweep, a re-score with new estimates) across
    processes. ev_ebitda_raw is not written — it is a pure transform of the
    matrix's label column that scorer recomputes anyway."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    universe.feature_matrix.to_parquet(dir_path / "feature_matrix.parquet")
    (dir_path / "companies.json").write_text(json.dumps(universe.companies, indent=2), encoding="utf-8")
    (dir_path / "llm_features.json").write_text(json.dumps(universe.llm_features, indent=2), encoding="utf-8")
    (dir_path / "imputation_medians.json").write_text(json.dumps(universe.imputation_medians, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": UNIVERSE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_companies": len(universe.companies),
        "n_feature_rows": int(universe.feature_matrix.shape[0]),
    }
    (dir_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_universe(dir_path: str | Path) -> EnrichedUniverse:
    """Load an EnrichedUniverse previously written by save_universe(). Raises if
    the artifact's schema version doesn't match this code's, rather than silently
    scoring against a stale/incompatible shape — re-enrich in that case."""
    dir_path = Path(dir_path)
    manifest = json.loads((dir_path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != UNIVERSE_SCHEMA_VERSION:
        raise ValueError(
            f"Universe artifact at {dir_path} has schema_version {manifest.get('schema_version')}, "
            f"expected {UNIVERSE_SCHEMA_VERSION}; re-enrich it."
        )
    feature_matrix = pd.read_parquet(dir_path / "feature_matrix.parquet")
    companies = json.loads((dir_path / "companies.json").read_text(encoding="utf-8"))
    llm_features = json.loads((dir_path / "llm_features.json").read_text(encoding="utf-8"))
    imputation_medians = json.loads((dir_path / "imputation_medians.json").read_text(encoding="utf-8"))
    ev_ebitda_raw = np.expm1(feature_matrix[feature_builder.LABEL_COLUMN])
    return EnrichedUniverse(
        companies=companies,
        llm_features=llm_features,
        feature_matrix=feature_matrix,
        ev_ebitda_raw=ev_ebitda_raw,
        imputation_medians=imputation_medians,
    )


def enrich_universe(config: PipelineConfig) -> EnrichedUniverse:
    """Steps 1-4: discover the comp universe from the target's SIC codes, fetch
    fundamentals + market data, extract comp business-model features, and build
    the standardized feature matrix. Target-financial-agnostic — see
    EnrichedUniverse for why that matters."""
    logger.info("STEP 1/6: Building universe")
    candidates = universe_builder.build(config)
    logger.info(f"Universe: {len(candidates)} candidates")

    logger.info("STEP 2/6: Fetching financial data")
    companies = fetcher.fetch_batch(candidates, config)
    before_market_cap_filter = len(companies)
    companies = universe_builder.filter_by_market_cap(companies)
    if len(companies) != before_market_cap_filter:
        logger.info(f"Market cap filter: kept {len(companies)}/{before_market_cap_filter} companies")
    before_domicile_filter = len(companies)
    companies = universe_builder.filter_by_domicile(companies)
    if len(companies) != before_domicile_filter:
        logger.info(f"Domicile filter: kept {len(companies)}/{before_domicile_filter} companies (US only)")
    n_valid = sum(1 for c in companies if c.get("ev_ebitda") is not None)
    logger.info(f"Fetched: {n_valid}/{len(companies)} companies with valid data")

    logger.info("STEP 3/6: Running LLM analysis")
    llm_features = llm_analyzer.analyze_batch(companies, config)
    n_success = sum(1 for v in llm_features.values() if not v.get("extraction_failed"))
    n_failed = sum(1 for v in llm_features.values() if v.get("extraction_failed"))
    n_low = sum(1 for v in llm_features.values() if v.get("low_confidence_flag"))
    logger.info(f"LLM extracted: {n_success} | failed: {n_failed} | low confidence: {n_low}")

    logger.info("STEP 4/6: Building feature matrix")
    feature_matrix, ev_ebitda_raw, imputation_medians = feature_builder.build(companies, llm_features)
    logger.info(f"Feature matrix: {feature_matrix.shape[0]} rows x {feature_matrix.shape[1]} columns")

    return EnrichedUniverse(
        companies=companies,
        llm_features=llm_features,
        feature_matrix=feature_matrix,
        ev_ebitda_raw=ev_ebitda_raw,
        imputation_medians=imputation_medians,
    )


def score_and_report(universe: EnrichedUniverse, config: PipelineConfig) -> dict:
    """Target-specific tail: extract the target's own business-model features,
    score the enriched comp pool by distance to the target (step 5), and generate
    the report (step 6). Re-runnable against the same EnrichedUniverse with
    different target estimates / scorer weights without re-enriching."""
    target_llm_features = llm_analyzer.analyze_target(config)

    logger.info("STEP 5/6: Scoring comps by distance to target")
    scorer_results = scorer.run(
        universe.feature_matrix, config.target_company, target_llm_features,
        universe.imputation_medians, config.scorer.feature_weights,
    )
    logger.info(f"Scored {len(scorer_results['company_scores'])} companies by financial-feature distance to target")

    logger.info("STEP 6/6: Generating report")
    output_paths = reporter.generate(
        scorer_results, universe.companies, universe.llm_features, target_llm_features,
        universe.imputation_medians, config,
    )
    saved_reports = [path for key, path in output_paths.items() if key != "n_comps"]
    logger.info(f"Report saved: {', '.join(saved_reports)}")
    return output_paths


def run_pipeline(
    config_path: str = "config.yaml",
    reuse_universe: str | None = None,
    save_universe_to: str | None = None,
) -> dict:
    """
    Run the full pipeline end-to-end: enrich the comp universe, then score it
    against the target and report. The two layers are separable (see
    enrich_universe / score_and_report) so the same universe can be re-scored
    without re-enriching.

    reuse_universe: load a saved EnrichedUniverse from this directory and skip
        steps 1-4 entirely (use when only the target estimates / scorer weights
        changed, not the SIC universe).
    save_universe_to: after enriching, persist the universe to this directory so
        a later run can reuse it.

    Returns the output paths dict from reporter.generate().
    """
    start_time = time.time()
    _ensure_directories()
    config = _load_config(config_path)

    if reuse_universe is not None:
        logger.info(f"Reusing enriched universe from {reuse_universe} (skipping steps 1-4)")
        universe = load_universe(reuse_universe)
    else:
        try:
            universe = enrich_universe(config)
        except Exception:
            logger.error("Pipeline failed during universe enrichment (steps 1-4)", exc_info=True)
            print("Pipeline failed during universe enrichment — see logs/pipeline.log for details.")
            raise
        if save_universe_to is not None:
            save_universe(universe, save_universe_to)
            logger.info(f"Saved enriched universe to {save_universe_to}")

    try:
        output_paths = score_and_report(universe, config)
    except Exception:
        logger.error("Pipeline failed during scoring/report (steps 5-6)", exc_info=True)
        print("Pipeline failed during scoring/report — see logs/pipeline.log for details.")
        raise

    if "n_comps" in output_paths:
        n_comps = int(output_paths["n_comps"])
    elif "csv" in output_paths:
        with open(output_paths["csv"], encoding="utf-8") as f:
            n_comps = max(sum(1 for _ in f) - 1, 0)  # minus header row
    else:
        n_comps = 0

    elapsed = time.time() - start_time
    target_name = config.target_company.name

    summary = (
        "=" * 60 + "\n"
        "PIPELINE COMPLETE\n"
        f"Target: {target_name}\n"
        f"Comparable companies found: {n_comps}\n"
        f"Report: {output_paths.get('html', output_paths.get('csv', 'not generated'))}\n"
        f"Run time: {elapsed:.0f}s\n"
        + "=" * 60
    )
    print(summary)
    logger.info(summary)

    return output_paths


def _print_sic_code_suggestions(config_path: str) -> None:
    config = _load_config(config_path)
    suggestions = llm_analyzer.suggest_sic_codes(config)

    if not suggestions:
        print("No SIC code suggestions returned (LLM call failed or returned no usable suggestions).")
        return

    print(
        "Advisory only — SIC codes are a fixed SEC list and the model may misremember\n"
        "specific 4-digit codes. Verify each one at https://www.sec.gov/info/edgar/siccodes.htm\n"
        "before adding it to config.yaml's primary_sic_codes / adjacent_sic_codes.\n"
    )
    for s in suggestions:
        bucket = s.get("bucket") or "unknown"
        print(f"[{bucket}] SIC {s.get('sic_code')} — {s.get('title')} (confidence: {s.get('confidence')})")
        print(f"    {s.get('reason')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PE Comparable Company Analysis Pipeline")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--suggest-sic-codes",
        action="store_true",
        help=(
            "Print advisory SIC code suggestions for the target's description "
            "and exit, without running the pipeline or modifying config.yaml"
        ),
    )
    parser.add_argument(
        "--save-universe",
        metavar="DIR",
        default=None,
        help="After enriching, save the comp universe to DIR so a later run can --reuse-universe it.",
    )
    parser.add_argument(
        "--reuse-universe",
        metavar="DIR",
        default=None,
        help=(
            "Load a previously saved comp universe from DIR and skip discovery/fetch/extraction "
            "(steps 1-4). Use when only the target estimates or scorer weights changed, not the SIC universe."
        ),
    )
    args = parser.parse_args()
    if args.suggest_sic_codes:
        _print_sic_code_suggestions(args.config)
        return
    run_pipeline(args.config, reuse_universe=args.reuse_universe, save_universe_to=args.save_universe)


if __name__ == "__main__":
    main()
