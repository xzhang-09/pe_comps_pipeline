from eval import coverage


def _config():
    return {
        "target_company": {
            "name": "Target Co",
            "description": "Makes engineered industrial equipment.",
            "revenue_usd_mm": 250.0,
            "ebitda_margin_estimate": 0.18,
            "primary_sic_codes": ["3559"],
            "adjacent_sic_codes": [],
        },
        "universe": {
            "max_candidates": 10,
            "primary_allocation_pct": 1.0,
            "min_revenue_usd_mm": 30,
            "max_revenue_usd_mm": 800,
            "min_ebitda_margin": 0.05,
        },
        "llm": {
            "extraction_model": "gpt-4.1",
            "judge_model": "gpt-4.1-mini",
            "embedding_model": "text-embedding-3-small",
            "temperature": 0,
            "max_tokens": 500,
            "batch_size": 10,
            "judge_threshold": 3,
        },
        "output": {"top_n_comps": 15, "report_formats": ["csv"]},
    }


def _attribute(ticker, *, companies=None, candidates=(), filers=(), cache=None, mocker=None):
    if mocker is not None:
        mocker.patch("eval.coverage.fetcher._load_cache", return_value=cache)
    return coverage.attribute_missing_ticker(
        ticker,
        companies_by_ticker=companies or {},
        candidate_tickers=set(candidates),
        sic_filer_tickers=set(filers),
        config=_config(),
    )


def test_survived_universe_without_market_cap_is_missing_market_cap():
    stage = _attribute("AAA", companies={"AAA": {"ticker": "AAA", "market_cap_usd_mm": None}})
    assert stage == "missing_market_cap"


def test_survived_universe_with_market_cap_is_no_valid_ev_ebitda():
    stage = _attribute("AAA", companies={"AAA": {"ticker": "AAA", "market_cap_usd_mm": 500.0}})
    assert stage == "no_valid_ev_ebitda"


def test_in_filer_list_but_not_candidate_is_quota_truncation():
    assert _attribute("AAA", filers=["AAA"]) == "truncated_by_max_candidates"


def test_absent_from_filer_list_is_not_in_sic_universe():
    assert _attribute("AAA") == "not_in_sic_universe"


def test_embedding_miss_uses_semantic_stage_attribution(make_config):
    config = make_config(universe={"discovery_mode": "sic+embedding"})
    assert coverage.attribute_missing_ticker(
        "AAA",
        companies_by_ticker={},
        candidate_tickers=set(),
        sic_filer_tickers=set(),
        config=config,
        semantic_trace={"AAA": "below_similarity_threshold"},
    ) == "below_similarity_threshold"
    assert coverage.attribute_missing_ticker(
        "OUTSIDE",
        companies_by_ticker={},
        candidate_tickers=set(),
        sic_filer_tickers=set(),
        config=config,
        semantic_trace={},
    ) == "outside_expanded_taxonomy"


def test_hybrid_discovery_probe_reuses_production_snapshot(mocker, make_config):
    config = make_config(universe={"discovery_mode": "suggest-sic+embedding"})
    mocker.patch(
        "eval.coverage.universe_builder.last_discovery_snapshot",
        return_value={
            "discovery_mode": "suggest-sic+embedding",
            "sic_filer_tickers": {"SIC"},
            "candidate_tickers": {"SIC", "SEM"},
        },
    )
    rebuild = mocker.patch("eval.coverage.universe_builder.build")

    assert coverage.deal_discovery_sets(config) == ({"SIC"}, {"SIC", "SEM"})
    rebuild.assert_not_called()


def test_candidate_without_cached_record_is_fetch_failed(mocker):
    assert _attribute("AAA", candidates=["AAA"], cache=None, mocker=mocker) == "fetch_failed"


def test_candidate_with_empty_record_is_fetch_failed(mocker):
    empty = {"ticker": "AAA", "revenue_ttm_usd_mm": None, "market_cap_usd_mm": None, "business_description": None}
    assert _attribute("AAA", candidates=["AAA"], cache=empty, mocker=mocker) == "fetch_failed"


def test_candidate_below_market_cap_floor(mocker):
    rec = {"ticker": "AAA", "revenue_ttm_usd_mm": 50.0, "market_cap_usd_mm": 5.0}
    assert _attribute("AAA", candidates=["AAA"], cache=rec, mocker=mocker) == "market_cap_filtered"


def test_candidate_outside_revenue_band(mocker):
    rec = {"ticker": "AAA", "revenue_ttm_usd_mm": 5000.0, "market_cap_usd_mm": 900.0, "ebitda_margin": 0.2}
    assert _attribute("AAA", candidates=["AAA"], cache=rec, mocker=mocker) == "financial_filtered"


def test_candidate_with_foreign_domicile(mocker):
    rec = {
        "ticker": "AAA", "revenue_ttm_usd_mm": 500.0, "market_cap_usd_mm": 900.0,
        "ebitda_margin": 0.2, "raw_fmp_profile": {"country": "DE"},
    }
    assert _attribute("AAA", candidates=["AAA"], cache=rec, mocker=mocker) == "non_us_filtered"


def test_candidate_passing_all_probes_is_unattributed(mocker):
    rec = {
        "ticker": "AAA", "revenue_ttm_usd_mm": 500.0, "market_cap_usd_mm": 900.0,
        "ebitda_margin": 0.2, "raw_fmp_profile": {"country": "US"},
    }
    assert _attribute("AAA", candidates=["AAA"], cache=rec, mocker=mocker) == "unattributed"


def test_coverage_for_deal_composes_stages(mocker):
    mocker.patch("eval.coverage.fetcher._load_cache", return_value=None)
    row = {
        "hits": ["HIT"],
        "missed_not_selected": ["RANKMISS", "LOWCONF"],
        "missed_not_in_universe": ["GONE"],
    }
    llm_features = {
        "HIT": {"low_confidence_flag": False},
        "RANKMISS": {"low_confidence_flag": False},
        "LOWCONF": {"low_confidence_flag": True},
    }
    stages = coverage.coverage_for_deal(
        row,
        llm_features=llm_features,
        companies_by_ticker={},
        candidate_tickers=set(),
        sic_filer_tickers=set(),
        config=_config(),
    )
    assert stages == {
        "HIT": "hit",
        "RANKMISS": "ranked_but_not_top_k",
        "LOWCONF": "low_confidence_filtered",
        "GONE": "not_in_sic_universe",
    }


def test_waterfall_counts_aggregates_in_stage_order():
    per_deal = [
        {"coverage": {"A": "not_in_sic_universe", "B": "hit"}},
        {"coverage": {"C": "hit", "D": "financial_filtered"}},
        {"coverage": None},
    ]
    assert coverage.waterfall_counts(per_deal) == {
        "hit": 2,
        "financial_filtered": 1,
        "not_in_sic_universe": 1,
    }


def test_reachable_precision_scores_ranking_layer_only():
    assert coverage.reachable_precision({"A": "hit", "B": "ranked_but_not_top_k", "C": "not_in_sic_universe"}) == 0.5
    assert coverage.reachable_precision({"C": "not_in_sic_universe"}) is None
    assert coverage.reachable_precision({}) is None
