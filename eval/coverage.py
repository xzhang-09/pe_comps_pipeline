"""Coverage waterfall for the manual ground-truth evaluation.

Turns a bare Precision@K number into a diagnosis: for every eligible banker
comp, attribute the exact pipeline stage where it was lost (or confirm it was
ranked). Discovery losses (wrong SIC, candidate-quota truncation, filters)
and ranking losses (scored but outside Top-K) have completely different
fixes, so conflating them — as a 0.0% headline number does — makes the eval
useless as a decision tool.

Stage attribution reuses the production predicates (universe_builder's
filter functions, fetcher's cache) rather than re-implemented copies, so the
attribution cannot drift from what the pipeline actually does. The probe runs
against the current fetch cache in the same process as the evaluation, so the
records it inspects are the ones the run used.
"""
from src import embedding_universe_builder, fetcher, get_logger, sic_universe_builder, universe_builder
from src.config_schema import PipelineConfig, as_config

logger = get_logger(__name__)

# Waterfall order, roughly latest-stage first. `hit` and `ranked_but_not_top_k`
# are the only two stages where the comp reached the selection ranking; every
# other stage is a coverage loss before ranking ever saw it.
STAGE_ORDER = (
    "hit",
    "ranked_but_not_top_k",
    "low_confidence_filtered",
    "missing_market_cap",
    "no_valid_ev_ebitda",
    "market_cap_filtered",
    "financial_filtered",
    "non_us_filtered",
    "fetch_failed",
    "truncated_by_max_candidates",
    "truncated_by_embedding_candidate_limit",
    "description_fetch_failed",
    "corpus_embedding_failed",
    "below_similarity_threshold",
    "outside_candidate_set_top_n",
    "outside_expanded_taxonomy",
    "not_in_sic_universe",
    "unattributed",
)

# Stages that mean "the ranking layer had a fair chance at this comp".
REACHED_RANKING_STAGES = ("hit", "ranked_but_not_top_k")


def deal_discovery_sets(config: PipelineConfig | dict) -> tuple[set[str], set[str]]:
    """(sic_filer_tickers, candidate_tickers) for a deal config, via the same
    production discovery calls the pipeline used (SIC caches make this a
    cache-hit when run right after enrichment). The gap between the two sets
    is exactly the max_candidates quota truncation."""
    cfg = as_config(config)
    if cfg.universe.discovery_mode == "suggest-sic+embedding":
        snapshot = universe_builder.last_discovery_snapshot()
        if snapshot.get("discovery_mode") == "suggest-sic+embedding":
            return set(snapshot["sic_filer_tickers"]), set(snapshot["candidate_tickers"])
    sic_codes = list(cfg.target_company.primary_sic_codes) + list(cfg.target_company.adjacent_sic_codes)
    filers = set(sic_universe_builder.discover_tickers_by_sic(sic_codes))
    candidates = universe_builder.build(cfg)
    candidate_tickers = {c["ticker"] if isinstance(c, dict) else c for c in candidates}
    return filers, candidate_tickers


def _record_is_empty(record: dict) -> bool:
    """An all-None placeholder record — the shape fetch_batch writes when every
    fetch attempt failed."""
    return (
        record.get("revenue_ttm_usd_mm") is None
        and record.get("market_cap_usd_mm") is None
        and not record.get("business_description")
    )


def attribute_missing_ticker(
    ticker: str,
    *,
    companies_by_ticker: dict,
    candidate_tickers: set[str],
    sic_filer_tickers: set[str],
    config: PipelineConfig | dict,
    semantic_trace: dict[str, str] | None = None,
) -> str:
    """Stage for an eligible ground-truth ticker that never reached
    company_scores. Checks run in reverse pipeline order: survived-the-universe
    first, then the filter chain (production predicates, one-record lists),
    then discovery membership."""
    cfg = as_config(config)

    record = companies_by_ticker.get(ticker)
    if record is not None:
        # Survived every universe filter but produced no scorable row. A
        # missing market cap (FMP quota/coverage gap) is actionable — rerun
        # after the quota resets — unlike a genuine EBITDA/XBRL gap.
        if record.get("market_cap_usd_mm") is None:
            return "missing_market_cap"
        return "no_valid_ev_ebitda"

    if ticker not in candidate_tickers:
        if cfg.universe.discovery_mode in {"sic+embedding", "suggest-sic+embedding"}:
            return (semantic_trace or {}).get(ticker) or embedding_universe_builder.unenumerated_stage(ticker)
        return "truncated_by_max_candidates" if ticker in sic_filer_tickers else "not_in_sic_universe"

    cached = fetcher._load_cache(ticker)
    if cached is None or _record_is_empty(cached):
        return "fetch_failed"
    if not universe_builder.filter_by_market_cap([cached]):
        return "market_cap_filtered"
    if not universe_builder.filter_by_financials([cached], cfg):
        return "financial_filtered"
    if not universe_builder.filter_by_domicile([cached]):
        return "non_us_filtered"

    logger.warning(f"{ticker} — was a candidate and passes every filter probe, yet absent from the universe")
    return "unattributed"


def coverage_for_deal(
    deal_row: dict,
    *,
    llm_features: dict,
    companies_by_ticker: dict,
    candidate_tickers: set[str],
    sic_filer_tickers: set[str],
    config: PipelineConfig | dict,
    semantic_trace: dict[str, str] | None = None,
) -> dict[str, str]:
    """{eligible ticker: stage} for one evaluated deal row (the dict produced
    by evaluator._evaluate_manual_deal)."""
    coverage = {}
    for ticker in deal_row["hits"]:
        coverage[ticker] = "hit"
    for ticker in deal_row["missed_not_selected"]:
        llm = llm_features.get(ticker) or {}
        coverage[ticker] = "low_confidence_filtered" if llm.get("low_confidence_flag") else "ranked_but_not_top_k"
    for ticker in deal_row["missed_not_in_universe"]:
        coverage[ticker] = attribute_missing_ticker(
            ticker,
            companies_by_ticker=companies_by_ticker,
            candidate_tickers=candidate_tickers,
            sic_filer_tickers=sic_filer_tickers,
            config=config,
            semantic_trace=semantic_trace,
        )
    return coverage


def waterfall_counts(per_deal: list[dict]) -> dict[str, int]:
    """Aggregate stage counts across deals, in STAGE_ORDER."""
    counts: dict[str, int] = {}
    for row in per_deal:
        for stage in (row.get("coverage") or {}).values():
            counts[stage] = counts.get(stage, 0) + 1
    return {stage: counts[stage] for stage in STAGE_ORDER if stage in counts}


def reachable_precision(coverage: dict[str, str]) -> float | None:
    """Hits over comps that actually reached the ranking — the ranking-layer
    quality signal, independent of discovery coverage. None when nothing
    reached ranking (a pure coverage failure; says nothing about ranking)."""
    reached = [s for s in coverage.values() if s in REACHED_RANKING_STAGES]
    if not reached:
        return None
    hits = sum(1 for s in reached if s == "hit")
    return hits / len(reached)
