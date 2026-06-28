"""
Tests for the penalty tuner. The core search is unit-tested with a stub
objective (no network); the capstone test composes the REAL precision objective
over a designed, offline pool and shows tuning beats a zero-penalty baseline —
the eval loop actually closing.
"""
import pandas as pd

import eval.tuner as tuner

EMBEDDING_MODEL = "text-embedding-3-small"
TARGET = "TGT"

# All six keys the precision objective reads; the grid overrides a subset.
BASE_ZERO_PENALTIES = {
    "business_model_penalty": 0.0,
    "customer_type_penalty": 0.0,
    "subsector_similarity_threshold": 0.5,
    "subsector_mismatch_penalty": 0.0,
    "size_penalty_free_log10_range": 1.0,
    "size_penalty_per_extra_log10": 0.0,
}


def test_grid_candidates_is_cartesian_product_over_base():
    candidates = tuner.grid_candidates(
        {"a": 1.0, "b": 2.0},
        {"a": [0.1, 0.2], "b": [0.3, 0.4]},
    )
    assert len(candidates) == 4
    assert {"a": 0.1, "b": 0.3} in candidates
    assert {"a": 0.2, "b": 0.4} in candidates
    # base keys not in the grid are preserved; here both were swept.
    assert all(set(c) == {"a", "b"} for c in candidates)


def test_tune_penalties_picks_highest_scoring_candidate():
    # Stub objective: score = the candidate's "x" value. Tuner should pick max.
    candidates = [{"x": 0.1}, {"x": 0.9}, {"x": 0.5}]
    result = tuner.tune_penalties(lambda p: p["x"], candidates)
    assert result.best_penalties == {"x": 0.9}
    assert result.best_score == 0.9
    assert [score for _, score in result.ranked] == [0.9, 0.5, 0.1]  # sorted best-first


def _penalty_sensitive_pool():
    """A 20-candidate pool where only the soft penalties keep attribute-mismatched
    distractors out of the Top-15 (mirrors tests/test_eval_precision_gate). Target
    is manufacturing / B2B / $300mm; the 6 true peers match, the 14 distractors
    have tempting low residuals but a wrong model / customer / scale. No
    sub_sector_description anywhere, so the objective stays fully offline."""
    residuals = {TARGET: 0.0}
    llm_features = {TARGET: {"business_model": "manufacturing", "customer_type": "B2B", "low_confidence_flag": False, "sub_sector_description": None}}
    companies_by_ticker = {TARGET: {"revenue_ttm_usd_mm": 300.0}}

    def add(ticker, residual, model, customer, revenue):
        residuals[ticker] = residual
        llm_features[ticker] = {"business_model": model, "customer_type": customer, "low_confidence_flag": False, "sub_sector_description": None}
        companies_by_ticker[ticker] = {"revenue_ttm_usd_mm": revenue}

    peers = []
    for i, res in enumerate([0.20, 0.28, 0.36, 0.44, 0.52, 0.60]):
        add(f"PEER{i}", res, "manufacturing", "B2B", 200.0 + 40 * i)
        peers.append(f"PEER{i}")
    for i, res in enumerate([0.10, 0.14, 0.18, 0.22, 0.26]):
        add(f"SAAS{i}", res, "SaaS", "B2B", 300.0)            # wrong business_model
    for i, res in enumerate([0.12, 0.16, 0.20, 0.24, 0.30]):
        add(f"B2G{i}", res, "manufacturing", "B2G", 300.0)    # wrong customer_type
    for i, res in enumerate([0.08, 0.11, 0.15, 0.19]):
        add(f"GIANT{i}", res, "manufacturing", "B2B", 30000.0)  # ~100x size mismatch

    company_scores = pd.DataFrame({"residual_abs": residuals})
    return {TARGET: peers}, company_scores, llm_features, companies_by_ticker


def test_tuning_improves_precision_over_zero_penalty_baseline():
    ground_truth, company_scores, llm_features, companies_by_ticker = _penalty_sensitive_pool()
    objective = tuner.precision_objective(
        ground_truth, company_scores, llm_features, companies_by_ticker, EMBEDDING_MODEL,
    )

    baseline_score = objective(BASE_ZERO_PENALTIES)  # penalties off -> distractors leak in

    candidates = tuner.grid_candidates(
        BASE_ZERO_PENALTIES,
        {
            "business_model_penalty": [0.0, 0.6],
            "customer_type_penalty": [0.0, 0.5],
            "size_penalty_per_extra_log10": [0.0, 0.8],
        },
    )
    result = tuner.tune_penalties(objective, candidates)

    assert baseline_score < 0.8                       # the un-tuned starting point is poor
    assert result.best_score > baseline_score          # tuning found something strictly better
    assert result.best_score >= 0.8                    # and it's actually good
    assert any(result.best_penalties[k] > 0 for k in ("business_model_penalty", "customer_type_penalty", "size_penalty_per_extra_log10"))

    report = tuner.generate_tuning_report(result, baseline_score=baseline_score)
    assert "Improvement:" in report
