"""
Golden-fixture test pinning the Top-N selection ranking shared between
reporter._select_top_15 and evaluator._select_top_k. Both entry points use
src.report_selection._ranked_candidates for ranking, while retaining
intentionally different *eligibility* rules (reporter drops the "training"
bucket; the evaluator drops the target itself). This fixture neutralizes
those asymmetries and pins the shared ranking semantics — residual_abs + the
four soft penalties, all added to the continuous financial distance (never
the ordinal rank).

The GOLDEN_ORDER is hand-derived: it would catch a change to the penalty
math itself (deliberate or not), not just disagreement between the two
callers.
"""
import pandas as pd

import eval.evaluator as evaluator
import src.reporter as reporter
from src import report_selection

TARGET = "TGT"

# Penalties are the single source of truth both functions read from config;
# the magnitudes here are chosen so the soft penalties actually reorder the
# candidates (a vacuous fixture where penalties don't bite would pass even if
# the penalty logic were deleted).
PENALTIES = {
    "business_model_penalty": 0.5,
    "customer_type_penalty": 0.3,
    "size_penalty_free_log10_range": 1.0,    # within 10x of target = free
    "size_penalty_per_extra_log10": 0.5,
    "subsector_mismatch_penalty": 0.4,
    "subsector_similarity_threshold": 0.7,
}

# residual_abs chosen so financial-only order differs from post-penalty order.
_RESIDUALS = {"DDD": 0.30, "AAA": 0.40, "EEE": 0.45, "CCC": 0.50, "BBB": 0.60}

# Golden post-penalty order. Hand-derived; encodes the intended "penalties live
# in distance units" semantics, not just "the two functions happen to agree".
GOLDEN_ORDER = ["CCC", "DDD", "BBB", "AAA", "EEE"]


def _company_scores():
    # Target deliberately absent from the pool: reporter never has it there,
    # and the evaluator excludes it anyway — keeping it out makes the candidate
    # set identical for both.
    return pd.DataFrame(
        {"residual_abs": _RESIDUALS, "ev_ebitda_actual": {t: 12.0 for t in _RESIDUALS}},
    )


def _llm_features():
    feats = {
        # candidate: (business_model, customer_type)
        "AAA": ("SaaS", "B2B"),            # business-model mismatch
        "BBB": ("manufacturing", "B2B"),   # clean
        "CCC": ("manufacturing", "B2B"),   # clean
        "DDD": ("manufacturing", "B2C"),   # customer-type mismatch
        "EEE": ("manufacturing", "B2B"),   # clean attrs, size mismatch via revenue
    }
    out = {
        t: {
            "business_model": bm,
            "customer_type": ct,
            "low_confidence_flag": False,
            "sub_sector_description": None,   # -> evaluator never hits embed_texts
        }
        for t, (bm, ct) in feats.items()
    }
    out[TARGET] = {
        "business_model": "manufacturing",
        "customer_type": "B2B",
        "sub_sector_description": None,
    }
    return out


def _companies_by_ticker():
    revenue = {"AAA": 150.0, "BBB": 150.0, "CCC": 150.0, "DDD": 150.0, "EEE": 15000.0}
    out = {t: {"revenue_ttm_usd_mm": rev} for t, rev in revenue.items()}
    out[TARGET] = {"revenue_ttm_usd_mm": 150.0}
    return out


def test_selection_ranking_matches_golden_order():
    company_scores = _company_scores()
    llm_features = _llm_features()
    companies_by_ticker = _companies_by_ticker()

    reporter_order = reporter._select_top_15(
        company_scores, llm_features, companies_by_ticker,
        target_business_model="manufacturing",
        target_customer_type="B2B",
        target_revenue=150.0,
        subsector_similarities={},          # no sub-sector penalty in this fixture
        penalties=PENALTIES,
        k=len(GOLDEN_ORDER),
    )

    evaluator_order = evaluator._select_top_k(
        TARGET, company_scores, llm_features, companies_by_ticker,
        penalties=PENALTIES,
        embedding_model="text-embedding-3-small",  # never called: no sub-sector text
        k=len(GOLDEN_ORDER),
    )

    # 1. Each function matches the intended semantics (catches correlated drift).
    assert reporter_order == GOLDEN_ORDER
    assert evaluator_order == GOLDEN_ORDER
    # 2. The literal "mirror" invariant the comments promise.
    assert reporter_order == evaluator_order


def test_llm_rerank_disabled_keeps_deterministic_order(mocker):
    rerank = mocker.patch("src.report_selection.llm_reranker.rerank")
    ranked = [{"ticker": ticker} for ticker in GOLDEN_ORDER]

    order = report_selection._select_ranked_tickers(
        ranked,
        k=3,
        llm_rerank={"enabled": False, "model": "gpt-4.1", "rerank_window": 5},
    )

    assert order == GOLDEN_ORDER[:3]
    rerank.assert_not_called()


def test_llm_rerank_valid_order_applied_and_invalid_order_falls_back(mocker):
    ranked = [{"ticker": ticker} for ticker in ["AAA", "BBB", "CCC", "DDD"]]
    rerank = mocker.patch(
        "src.report_selection.llm_reranker.rerank",
        side_effect=[(["CCC", "AAA", "BBB"], []), None],
    )
    config = {"enabled": True, "model": "gpt-4.1", "rerank_window": 3}

    applied = report_selection._select_ranked_tickers(ranked, k=4, llm_rerank=config)
    fallback = report_selection._select_ranked_tickers(ranked, k=4, llm_rerank=config)

    assert applied == ["CCC", "AAA", "BBB", "DDD"]
    assert fallback == ["AAA", "BBB", "CCC", "DDD"]
    assert rerank.call_count == 2


def test_penalty_is_in_distance_units_not_ordinal_rank():
    """The specific drift the docstrings fear: a categorical flag must demote a
    comp by a comparable amount of financial distance, not leapfrog it across
    the whole list. DDD has the best raw residual (0.30) but a customer-type
    penalty (+0.3 -> 0.60); it stays 2nd, NOT dumped to the bottom as an
    ordinal +N-rank penalty would do. AAA's larger 0.5 penalty hurts it more."""
    order = reporter._select_top_15(
        _company_scores(), _llm_features(), _companies_by_ticker(),
        "manufacturing", "B2B", 150.0, {}, PENALTIES, k=5,
    )
    assert order.index("DDD") == 1          # 2nd, not last
    assert order.index("DDD") < order.index("AAA")  # 0.3 penalty hurts less than 0.5
