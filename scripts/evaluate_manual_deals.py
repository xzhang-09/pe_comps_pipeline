import argparse
import json
import statistics
import traceback
from copy import deepcopy
from pathlib import Path

import yaml

from eval import coverage, evaluator
from src import embedding_universe_builder, get_logger, llm_analyzer, pipeline, scorer, sic_universe_builder, universe_builder
from src.config_schema import PipelineConfig

logger = get_logger(__name__)

DEFAULT_OUTPUT_JSON = Path("outputs/eval/manual_deals/results.json")

# Discovery ladder rungs. Each rung is a separately-labeled measurement, not a
# replacement for the one below it — the deltas between rungs are the point:
# - single-sic: the target's own (corrected) SIC code only. Lower bound;
#   quantifies bare taxonomy-driven discovery with zero analyst input.
# - suggest-sic: additionally feeds llm_analyzer.suggest_sic_codes() output
#   (derived from the business description ONLY — the banker list is never an
#   input, so no answer leakage) into primary/adjacent discovery, mirroring the
#   documented real-analyst flow: ask for codes, verify, run.
# - sic+embedding: keeps the SIC setup for the rung, then enables the optional
#   semantic discovery channel in universe_builder.
DISCOVERY_MODES = ("single-sic", "suggest-sic", "sic+embedding", "suggest-sic+embedding")


def _load_config(config_path: str | Path) -> PipelineConfig:
    with open(config_path, encoding="utf-8") as f:
        return PipelineConfig.model_validate(yaml.safe_load(f))


# Revenue band relative to the deal target, replacing the base config's
# absolute band (which is tuned to the demo target's ~$150mm scale). A $2.5bn
# Masonite and a $250mm Starrett need different bands; an absolute cap of
# $800mm would silently delete most of a large target's banker comps before
# scoring — the primary cause of the first benchmark run's 0.0%.
REVENUE_BAND_MULTIPLE = 10.0


def _deal_config(
    base_config: PipelineConfig,
    deal: dict,
    extra_codes: dict[str, list[str]] | None = None,
    discovery_mode: str = "sic",
) -> PipelineConfig:
    target_financials = deal.get("target_financials") or {}
    target_revenue = target_financials.get("revenue_usd_mm")
    target_sic = str(deal["target_sic"])
    extra_primary = list((extra_codes or {}).get("primary") or [])
    adjacent = list((extra_codes or {}).get("adjacent") or [])
    primary = [target_sic] + [c for c in extra_primary if c != target_sic]

    raw = base_config.model_dump()
    raw["target_company"].update({
        "name": deal["target_name"],
        "description": deal["business_description"],
        "revenue_usd_mm": target_revenue,
        "ebitda_margin_estimate": target_financials.get("ebitda_margin_estimate"),
        "primary_sic_codes": primary,
        "adjacent_sic_codes": adjacent,
    })
    raw["universe"].update({
        "seed_tickers": [],
        "must_include_tickers": [],
        "exclude_tickers": [],
        "allow_broad_sic_codes": True,
        "discovery_mode": discovery_mode,
        # Unused bucket quota is not redistributed between buckets, so with an
        # empty adjacent bucket anything below 1.0 silently halves the pool.
        # With suggested adjacent codes present, keep most of the quota on the
        # primary bucket, matching the buckets' expected-to-surface semantics.
        "primary_allocation_pct": (
            0.7 if adjacent or discovery_mode == "suggest-sic+embedding" else 1.0
        ),
        "min_revenue_usd_mm": (
            target_revenue / REVENUE_BAND_MULTIPLE if target_revenue else None
        ),
        "max_revenue_usd_mm": (
            target_revenue * REVENUE_BAND_MULTIPLE if target_revenue else None
        ),
    })
    return PipelineConfig.model_validate(raw)


# suggest_sic_codes truncates badly at the pipeline default of 500 output
# tokens (6-12 suggestions with reasons); mirror ui._sic_suggestion_config.
SUGGESTION_MIN_MAX_TOKENS = llm_analyzer.SIC_SUGGESTION_MIN_MAX_TOKENS


def _suggested_discovery_codes(deal: dict, base_config: PipelineConfig) -> dict[str, list[str]]:
    """Rung 2 of the discovery ladder: extra SIC codes suggested by the LLM
    from the deal target's business description alone — the banker comp list
    is never an input, so discovery stays leakage-free. Suggestions are
    already validated against the official SEC SIC table inside
    suggest_sic_codes(); here they are additionally preflighted so a
    validated-but-zero-filer code can't hard-abort the deal's enrichment
    (universe_builder's preflight raises on zero-yield codes)."""
    target_sic = str(deal["target_sic"])
    suggestion_config = _deal_config(base_config, deal)
    raw = suggestion_config.model_dump()
    raw["llm"]["max_tokens"] = max(int(raw["llm"].get("max_tokens") or 0), SUGGESTION_MIN_MAX_TOKENS)
    suggestions = llm_analyzer.suggest_sic_codes(PipelineConfig.model_validate(raw))

    primary: list[str] = []
    adjacent: list[str] = []
    for suggestion in suggestions:
        code = str(suggestion.get("sic_code") or "").strip()
        if not code or code == target_sic:
            continue
        bucket = str(suggestion.get("bucket") or "").strip().lower()
        (primary if bucket == "primary" else adjacent).append(code)
    primary = list(dict.fromkeys(primary))
    adjacent = [c for c in dict.fromkeys(adjacent) if c not in primary]

    probed = primary + adjacent
    if probed:
        try:
            counts = sic_universe_builder.preflight_sic_codes(probed)
        except Exception as exc:
            logger.warning(f"{deal['deal_id']} — suggested-SIC preflight failed ({exc}); using unprobed codes")
            counts = {}
        zero_yield = {code for code, n in counts.items() if n == 0}
        if zero_yield:
            logger.info(f"{deal['deal_id']} — dropping zero-yield suggested SIC codes: {sorted(zero_yield)}")
        primary = [c for c in primary if c not in zero_yield]
        adjacent = [c for c in adjacent if c not in zero_yield]
    return {"primary": primary, "adjacent": adjacent}


def _companies_by_ticker(companies: list[dict], deal: dict) -> dict:
    by_ticker = {company["ticker"]: company for company in companies}
    target = evaluator._ticker(deal["target_ticker"])
    by_ticker[target] = {
        "ticker": target,
        "company_name": deal["target_name"],
        "revenue_ttm_usd_mm": (deal.get("target_financials") or {}).get("revenue_usd_mm"),
    }
    return by_ticker


def _json_ready(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _write_results_json(results: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(results), indent=2), encoding="utf-8")


def _write_results_md(results: dict, path: str | Path) -> str:
    path = Path(path)
    original_results_path = evaluator.RESULTS_PATH
    try:
        evaluator.RESULTS_PATH = path
        return evaluator.generate_manual_ground_truth_report(results)
    finally:
        evaluator.RESULTS_PATH = original_results_path


def _failed_deal_row(deal: dict, exc: Exception) -> dict:
    """Placeholder row for a deal whose pipeline run crashed — keeps the rest
    of the benchmark running and the failure visible in the report. precision
    None keeps it out of every aggregate."""
    return {
        "deal_id": deal["deal_id"],
        "target_ticker": evaluator._ticker(deal["target_ticker"]),
        "target_name": deal.get("target_name"),
        "filing_url": deal.get("filing_url"),
        "advisor": deal.get("advisor"),
        "filing_date": deal.get("filing_date"),
        "error": f"{type(exc).__name__}: {exc}",
        "selected_tickers": [],
        "eligible_ground_truth_tickers": [],
        "excluded_delisted_tickers": [],
        "excluded_non_us_filer_tickers": [],
        "hits": [],
        "missed_not_in_universe": [],
        "missed_not_selected": [],
        "precision": None,
    }


def _evaluate_one_deal(deal: dict, base_config: PipelineConfig, k: int, discovery: str) -> dict:
    extra_codes = None
    if discovery == "suggest-sic":
        extra_codes = _suggested_discovery_codes(deal, base_config)
    deal_config = _deal_config(
        base_config,
        deal,
        extra_codes=extra_codes,
        discovery_mode=(
            discovery if discovery in {"sic+embedding", "suggest-sic+embedding"} else "sic"
        ),
    )
    universe = pipeline.enrich_universe(deal_config)
    target_llm_features = llm_analyzer.analyze_target(deal_config)
    scorer_results = scorer.run(
        universe.feature_matrix,
        deal_config.target_company,
        target_llm_features,
        universe.imputation_medians,
        deal_config.scorer.feature_weights,
    )
    llm_features = deepcopy(universe.llm_features)
    target = evaluator._ticker(deal["target_ticker"])
    llm_features[target] = target_llm_features
    companies_by_ticker = _companies_by_ticker(universe.companies, deal)

    deal_result = evaluator.run_manual_ground_truth_evaluation(
        [deal],
        scorer_results["company_scores"],
        llm_features,
        companies_by_ticker,
        base_config.model_dump(),
        k=k,
    )
    row = deal_result["per_deal"][0]

    # Coverage waterfall: attribute every eligible comp to the stage where it
    # was lost. Probing must never take down the evaluation itself.
    try:
        sic_filers, candidate_tickers = coverage.deal_discovery_sets(deal_config)
    except Exception as exc:
        logger.warning(f"{deal['deal_id']} — discovery probe failed ({exc}); coverage stages will be partial")
        sic_filers, candidate_tickers = set(), set()
    row["coverage"] = coverage.coverage_for_deal(
        row,
        llm_features=llm_features,
        companies_by_ticker=companies_by_ticker,
        candidate_tickers=candidate_tickers,
        sic_filer_tickers=sic_filers,
        config=deal_config,
        semantic_trace=embedding_universe_builder.last_discovery_trace(),
    )
    row["reachable_precision"] = coverage.reachable_precision(row["coverage"])

    score_index = scorer_results["company_scores"].index
    n_selectable = sum(
        1 for t in score_index
        if t != target and (llm_features.get(t) or {}).get("low_confidence_flag") is False
    )
    row["n_scored"] = int(len(score_index))
    row["n_selectable"] = int(n_selectable)
    # With a selectable pool at or below K, "selection" picks everything that
    # survived coverage — precision then measures discovery, not ranking.
    row["selection_trivial"] = n_selectable <= k
    row["discovery_mode"] = discovery
    # In suggest-sic+embedding mode the suggested codes are added on a local
    # config copy inside universe_builder.build(), so deal_config still holds
    # only the pre-expansion codes — read the audited set from the snapshot.
    snapshot = universe_builder.last_discovery_snapshot()
    if (
        deal_config.universe.discovery_mode == "suggest-sic+embedding"
        and snapshot.get("discovery_mode") == "suggest-sic+embedding"
        and snapshot.get("primary_sic_codes")
    ):
        row["discovery_sic_codes"] = {
            "primary": list(snapshot["primary_sic_codes"]),
            "adjacent": list(snapshot["adjacent_sic_codes"]),
        }
    else:
        row["discovery_sic_codes"] = {
            "primary": list(deal_config.target_company.primary_sic_codes),
            "adjacent": list(deal_config.target_company.adjacent_sic_codes),
        }
    return row


def evaluate_manual_deals(
    config_path: str | Path = "config.yaml",
    manual_deals_path: str | Path = evaluator.MANUAL_DEALS_PATH,
    output_json_path: str | Path = DEFAULT_OUTPUT_JSON,
    results_md_path: str | Path = evaluator.RESULTS_PATH,
    k: int = evaluator.TOP_K,
    validate_size: bool = True,
    only_deal_ids: list[str] | None = None,
    discovery: str = "single-sic",
) -> dict:
    if discovery not in DISCOVERY_MODES:
        raise ValueError(f"Unknown discovery mode {discovery!r}; expected one of {DISCOVERY_MODES}")
    base_config = _load_config(config_path)
    deals = evaluator.load_manual_deals(manual_deals_path)
    if validate_size:
        evaluator.validate_manual_deals_benchmark(deals)
    if only_deal_ids:
        unknown = set(only_deal_ids) - {d["deal_id"] for d in deals}
        if unknown:
            raise ValueError(f"Unknown deal ids: {sorted(unknown)}")
        deals = [d for d in deals if d["deal_id"] in set(only_deal_ids)]

    per_deal = []
    for deal in deals:
        try:
            per_deal.append(_evaluate_one_deal(deal, base_config, k, discovery))
        except Exception as exc:
            logger.error(f"{deal['deal_id']} — evaluation failed", exc_info=True)
            print(f"{deal['deal_id']} — evaluation failed: {exc}\n{traceback.format_exc()}")
            per_deal.append(_failed_deal_row(deal, exc))

    precision_values = [row["precision"] for row in per_deal if row["precision"] is not None]
    reachable_values = [
        row["reachable_precision"] for row in per_deal if row.get("reachable_precision") is not None
    ]
    results = {
        "discovery_mode": discovery,
        "mean_precision": sum(precision_values) / len(precision_values) if precision_values else 0.0,
        "median_precision": statistics.median(precision_values) if precision_values else 0.0,
        "mean_reachable_precision": (
            sum(reachable_values) / len(reachable_values) if reachable_values else None
        ),
        "n_deals": len(per_deal),
        "n_failed_deals": sum(1 for row in per_deal if row.get("error")),
        "k": k,
        "coverage_waterfall": coverage.waterfall_counts(per_deal),
        "per_deal": per_deal,
    }
    _write_results_json(results, output_json_path)
    _write_results_md(results, results_md_path)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the manual fairness-opinion ground-truth evaluation.")
    parser.add_argument("--config", default="config.yaml", help="Pipeline config path.")
    parser.add_argument("--manual-deals", default=str(evaluator.MANUAL_DEALS_PATH), help="Manual deals JSON path.")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Machine-readable results JSON path.")
    parser.add_argument("--results-md", default=str(evaluator.RESULTS_PATH), help="Markdown report path.")
    parser.add_argument("--k", type=int, default=evaluator.TOP_K, help="Precision@K cutoff.")
    parser.add_argument("--skip-size-validation", action="store_true", help="Allow evaluating fewer/more than the audited 8-10 deal set.")
    parser.add_argument(
        "--deals",
        default=None,
        help=(
            "Comma-separated deal_ids to evaluate (default: all). Useful for spreading "
            "uncached runs across FMP quota days, or re-running a single failed deal."
        ),
    )
    parser.add_argument(
        "--discovery",
        choices=DISCOVERY_MODES,
        default="single-sic",
        help=(
            "Discovery ladder rung: 'single-sic' (baseline, target's own SIC only) or "
            "'suggest-sic' (adds SEC-validated LLM-suggested codes derived from the "
            "business description) or 'sic+embedding' (adds semantic discovery). "
            "Non-baseline runs write to mode-suffixed default "
            "output paths so the baseline results are not overwritten."
        ),
    )
    return parser.parse_args()


def _mode_suffixed(path_str: str, default_path: Path, discovery: str) -> str:
    """Suffix a *default* output path with the discovery mode for non-baseline
    runs (results.json -> results.suggest-sic.json), so ladder rungs land side
    by side instead of overwriting each other. Explicit user paths are kept."""
    if discovery == "single-sic" or path_str != str(default_path):
        return path_str
    suffix = "embedding" if discovery == "sic+embedding" else discovery
    return str(default_path.with_name(f"{default_path.stem}.{suffix}{default_path.suffix}"))


def main() -> None:
    args = _parse_args()
    results = evaluate_manual_deals(
        config_path=args.config,
        manual_deals_path=args.manual_deals,
        output_json_path=_mode_suffixed(args.output_json, DEFAULT_OUTPUT_JSON, args.discovery),
        results_md_path=_mode_suffixed(args.results_md, evaluator.RESULTS_PATH, args.discovery),
        k=args.k,
        validate_size=not args.skip_size_validation,
        only_deal_ids=[d.strip() for d in args.deals.split(",") if d.strip()] if args.deals else None,
        discovery=args.discovery,
    )
    print(
        f"Manual deal evaluation complete: {results['n_deals']} deals "
        f"({results['n_failed_deals']} failed), mean Precision@{results['k']} "
        f"{results['mean_precision'] * 100:.1f}%"
    )
    if results.get("mean_reachable_precision") is not None:
        print(f"Ranking-layer (reachable) precision: {results['mean_reachable_precision'] * 100:.1f}%")
    if results.get("coverage_waterfall"):
        print("Coverage waterfall:", json.dumps(results["coverage_waterfall"]))


if __name__ == "__main__":
    main()
