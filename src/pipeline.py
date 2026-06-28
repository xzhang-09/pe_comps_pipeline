import argparse
import time
from dataclasses import dataclass
from pathlib import Path

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
) -> dict:
    """
    Run the full pipeline end-to-end: enrich the comp universe, then score it
    against the target and report. The two layers are separable (see
    enrich_universe / score_and_report) so the same universe can be re-scored
    without re-enriching.

    Returns the output paths dict from reporter.generate().
    """
    start_time = time.time()
    _ensure_directories()
    config = _load_config(config_path)

    try:
        universe = enrich_universe(config)
    except Exception:
        logger.error("Pipeline failed during universe enrichment (steps 1-4)", exc_info=True)
        print("Pipeline failed during universe enrichment — see logs/pipeline.log for details.")
        raise

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
    args = parser.parse_args()
    if args.suggest_sic_codes:
        _print_sic_code_suggestions(args.config)
        return
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
