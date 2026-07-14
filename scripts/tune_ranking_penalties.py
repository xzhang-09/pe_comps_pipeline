"""Grid-search the ranking-layer soft-penalty magnitudes against the manual
fairness-opinion benchmark, dev/holdout-disciplined.

NOT YET USABLE FOR A REAL ANSWER, as currently pointed at single-sic
discovery. Checked directly against the committed 16-deal single-sic
results: all 10 dev deals have `selection_trivial=True` (candidate pools of
1-7 companies, all <= K=15) — `_select_ranked_tickers` returns the entire
ranked list unchanged whenever the pool is at or below K, so precision on
every dev deal is mathematically invariant to the penalties being searched.
Running this today would report a "best" configuration that is pure
tie-breaking noise. See "What the coverage waterfall taught us" #4 in
docs/known_limitations_and_roadmap.md for the full writeup. This unblocks
once a larger-pool discovery mode (suggest-sic or an embedding channel) is
re-measured on the 16-deal benchmark (see "Pending" in that same doc) —
then point `_deal_universe`'s hardcoded `discovery_mode="sic"` at that mode
instead (parametrizing it, e.g. a `--discovery` flag mirroring
scripts/evaluate_manual_deals.py's) and actually run the search.

Each dev deal's universe is enriched and scored exactly once (single-sic
discovery, matching the committed baseline) — the expensive, FMP-dependent
step — then eval.tuner sweeps only the penalty magnitudes against that fixed
data, per eval/tuner.py's design ("no re-fetching, no re-extraction"). The
best candidate found on dev is then scored once, separately, on holdout —
holdout never participates in candidate selection, so its number is an
honest read on generalization rather than a second bite at the same fit.

This only prints a report; it does not write config.yaml. Committing new
penalty values is a deliberate, reviewed act — see the report's "apply"
section for the exact values to paste in.

Usage:
    python -m scripts.tune_ranking_penalties
    python -m scripts.tune_ranking_penalties --grid small   # faster, coarser
"""
import argparse
from pathlib import Path

import yaml

from eval import evaluator, tuner
from eval.evaluator import load_manual_deals
from scripts.check_eval_regression import DEFAULT_HOLDOUT
from scripts.evaluate_manual_deals import _companies_by_ticker, _deal_config
from src import get_logger, llm_analyzer, pipeline, scorer
from src.config_schema import PipelineConfig

logger = get_logger(__name__)

DEFAULT_OUTPUT_MD = Path("eval/penalty_tuning.md")

# A coarser 2^4=16-candidate grid for a quick pass; tuner.DEFAULT_GRID
# (3^4=81) for the full search. Both hold subsector_similarity_threshold and
# size_penalty_free_log10_range at the config baseline — see
# tune_thresholds() below for that axis, tuned separately (coordinate-style)
# rather than joined into one combinatorial grid.
SMALL_GRID: dict[str, list[float]] = {
    "business_model_penalty": [0.3, 0.9],
    "customer_type_penalty": [0.25, 0.75],
    "subsector_mismatch_penalty": [0.5, 2.0],
    "size_penalty_per_extra_log10": [0.5, 1.5],
}

# Coordinate-descent follow-up sweep: subsector_similarity_threshold with the
# magnitude penalties held at whatever tune_penalties() found best. A joint
# grid over 5 dimensions would be 3x the candidates for one more axis; this
# stays cheap and still principled — see RankingPenaltiesConfig's docstring
# for why 0.48 was the original calibration.
THRESHOLD_GRID = [0.40, 0.44, 0.48, 0.52, 0.56]


def _deal_universe(deal: dict, base_config: PipelineConfig) -> tuple:
    """Enrich + score one deal's universe once (single-sic, matching the
    committed baseline) — cached after the first eval run, so this is a
    cache-hit replay, not a fresh fetch. Returns
    (company_scores, llm_features, companies_by_ticker) held fixed across
    every penalty candidate for this deal."""
    deal_config = _deal_config(base_config, deal, discovery_mode="sic")
    universe = pipeline.enrich_universe(deal_config)
    target_llm_features = llm_analyzer.analyze_target(deal_config)
    scorer_results = scorer.run(
        universe.feature_matrix, deal_config.target_company, target_llm_features,
        universe.imputation_medians, deal_config.scorer.feature_weights,
    )
    llm_features = dict(universe.llm_features)
    target = evaluator._ticker(deal["target_ticker"])
    llm_features[target] = target_llm_features
    companies_by_ticker = _companies_by_ticker(universe.companies, deal)
    return scorer_results["company_scores"], llm_features, companies_by_ticker


def _load_deal_universes(deals: list[dict], base_config: PipelineConfig) -> dict[str, tuple]:
    universes = {}
    for deal in deals:
        logger.info(f"{deal['deal_id']} — enriching + scoring (single-sic, cache-hit expected)")
        universes[deal["deal_id"]] = _deal_universe(deal, base_config)
    return universes


def make_objective(deals: list[dict], universes: dict[str, tuple], k: int, base_config_dict: dict):
    """penalties -> mean precision@k across `deals`, holding each deal's
    (company_scores, llm_features, companies_by_ticker) fixed — the only
    thing that varies per call is the penalties dict. base_config_dict
    supplies every other config field _evaluate_manual_deal reads (e.g.
    llm.embedding_model for the subsector-similarity lookup)."""
    def objective(penalties: dict) -> float:
        precisions = []
        for deal in deals:
            company_scores, llm_features, companies_by_ticker = universes[deal["deal_id"]]
            call_config = {**base_config_dict, "scorer": {**base_config_dict["scorer"], "ranking_penalties": penalties}}
            result = evaluator.run_manual_ground_truth_evaluation(
                [deal], company_scores, llm_features, companies_by_ticker,
                call_config, k=k,
            )
            precision = result["per_deal"][0]["precision"]
            if precision is not None:
                precisions.append(precision)
        return sum(precisions) / len(precisions) if precisions else 0.0
    return objective


def tune_thresholds(objective, base_penalties: dict, grid: list[float]) -> tuner.TuneResult:
    """Coordinate sweep of subsector_similarity_threshold alone, holding
    every other penalty (including whatever tune_penalties() just found) at
    base_penalties' values."""
    candidates = [{**base_penalties, "subsector_similarity_threshold": t} for t in grid]
    return tuner.tune_penalties(objective, candidates)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Pipeline config path (source of baseline penalties).")
    parser.add_argument("--manual-deals", default=str(evaluator.MANUAL_DEALS_PATH))
    parser.add_argument("--grid", choices=["small", "full"], default="full")
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--k", type=int, default=None, help="Precision@K cutoff (default: config output.top_n_comps or 15).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with open(args.config, encoding="utf-8") as f:
        base_config = PipelineConfig.model_validate(yaml.safe_load(f))
    k = args.k or base_config.output.top_n_comps or 15
    base_penalties = base_config.scorer.ranking_penalties.model_dump()
    base_config_dict = base_config.model_dump()

    deals = load_manual_deals(args.manual_deals)
    dev_deals = [d for d in deals if d["deal_id"] not in DEFAULT_HOLDOUT]
    holdout_deals = [d for d in deals if d["deal_id"] in DEFAULT_HOLDOUT]
    logger.info(f"Dev: {len(dev_deals)} deals | Holdout: {len(holdout_deals)} deals (never used for selection)")

    dev_universes = _load_deal_universes(dev_deals, base_config)
    dev_objective = make_objective(dev_deals, dev_universes, k, base_config_dict)

    baseline_score = dev_objective(base_penalties)
    logger.info(f"Baseline (current config.yaml penalties) on dev: {baseline_score * 100:.1f}%")

    grid = SMALL_GRID if args.grid == "small" else tuner.DEFAULT_GRID
    candidates = tuner.grid_candidates(base_penalties, grid)
    magnitude_result = tuner.tune_penalties(dev_objective, candidates)

    threshold_result = tune_thresholds(dev_objective, magnitude_result.best_penalties, THRESHOLD_GRID)
    final_best = threshold_result.best_penalties
    final_score = threshold_result.best_score

    holdout_universes = _load_deal_universes(holdout_deals, base_config)
    holdout_objective = make_objective(holdout_deals, holdout_universes, k, base_config_dict)
    holdout_baseline = holdout_objective(base_penalties)
    holdout_final = holdout_objective(final_best)

    report_lines = [
        "# Ranking-Layer Penalty Tuning",
        "",
        f"Grid: {args.grid} ({len(candidates)} magnitude candidates x {len(THRESHOLD_GRID)} threshold candidates)",
        f"K: {k}  |  Dev deals: {len(dev_deals)}  |  Holdout deals: {len(holdout_deals)} (not used for selection)",
        "",
        "## Dev-set search (selection)",
        f"- Baseline (current config.yaml): {baseline_score * 100:.1f}%",
        f"- Best after magnitude sweep:     {magnitude_result.best_score * 100:.1f}%",
        f"- Best after threshold sweep:     {final_score * 100:.1f}%  "
        f"({(final_score - baseline_score) * 100:+.1f}pp vs baseline)",
        "",
        "## Holdout validation (never used for selection)",
        f"- Baseline penalties on holdout:  {holdout_baseline * 100:.1f}%",
        f"- Tuned penalties on holdout:     {holdout_final * 100:.1f}%  "
        f"({(holdout_final - holdout_baseline) * 100:+.1f}pp vs baseline)",
        "",
        "## Selected penalties",
        "```yaml",
        *[f"{key}: {value}" for key, value in sorted(final_best.items())],
        "```",
        "",
        "Not applied to config.yaml — copy the values above into "
        "scorer.ranking_penalties only after confirming the holdout delta "
        "moves the same direction as dev (a dev improvement that reverses on "
        "holdout is overfitting the 10-deal dev set, not a real gain).",
    ]
    report = "\n".join(report_lines) + "\n"
    print(report)
    Path(args.output_md).write_text(report, encoding="utf-8")
    logger.info(f"Report written to {args.output_md}")


if __name__ == "__main__":
    main()
