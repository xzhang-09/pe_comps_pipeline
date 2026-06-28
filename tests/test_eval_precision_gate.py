"""
Offline, deterministic precision@K *regression gate* for CI.

This is distinct from tests/test_evaluator.py, which unit-tests the precision
*mechanics* on 3-company toy inputs (e.g. precision == 2/3). This module is a
*quality* gate: a realistic-sized frozen comp pool with a known answer key,
run through the same aggregate entry point production uses
(evaluator._evaluate_precision_at_k), asserting selection quality stays above a
floor. A refactor that quietly weakens the selection logic (the kind #3/#4/#5
risk) shows up here as precision dropping through the floor.

Why only precision is gated, not the rest of run_evaluation: the LLM-consistency
evaluation deliberately measures the *non-determinism* of the extractor (same
description, two live calls, do they agree?). It needs the network and gives a
different answer every run — the opposite of what a CI gate needs. precision@K,
given frozen inputs, is fully deterministic and offline, so it is the only half
that can be a gate.

The pool is synthetic — this anchors against *regression*, not a published
benchmark. The honest real-world precision number still comes from running the
pipeline against SEC fairness-opinion ground truth (see ground_truth_builder),
which this fixture is not a substitute for.
"""
import pandas as pd

import eval.evaluator as evaluator

EMBEDDING_MODEL = "text-embedding-3-small"

# Single source of truth both reporter and evaluator read from config. Values
# here must keep the soft penalties load-bearing — the gate's whole point is
# that removing them lets attribute-mismatched distractors leak into the Top-K.
PENALTIES = {
    "business_model_penalty": 0.6,
    "customer_type_penalty": 0.5,
    "size_penalty_free_log10_range": 1.0,
    "size_penalty_per_extra_log10": 0.8,
    "subsector_similarity_threshold": 0.5,
    "subsector_mismatch_penalty": 0.4,
}

TARGET = "TGT"  # manufacturing / B2B / revenue 300

# Floor, not an exact value: exact-match gates are brittle (any benign tweak
# trips them). A floor says "don't regress below this", which is what a quality
# gate actually wants. Set below the observed clean-run precision (1.0) with
# headroom; a broken penalty path drops this to ~0.33, well under the floor.
PRECISION_FLOOR = 0.80


def _llm(business_model, customer_type):
    return {
        "business_model": business_model,
        "customer_type": customer_type,
        "low_confidence_flag": False,
        "sub_sector_description": None,   # -> _subsector_similarities short-circuits, no network
    }


def _build_pool():
    """
    A 20-candidate pool around TARGET (manufacturing / B2B / $300mm revenue),
    with 6 genuine peers and 14 distractors engineered so that *only the soft
    penalties* keep the distractors out of the Top-15:

      - 6 peers:     same model/customer, comparable size, mid residuals.
      - 5 SaaS:      tempting low residuals, wrong business_model.
      - 5 B2G:       tempting low residuals, wrong customer_type.
      - 4 giants:    tempting low residuals, ~100x revenue (size mismatch).

    With penalties on, peers occupy the top slots and precision is 1.0. With
    penalties off, the low-residual distractors outrank the peers and shove
    most of them past rank 15 — precision collapses.
    """
    residuals = {TARGET: 0.0}
    llm_features = {TARGET: _llm("manufacturing", "B2B")}
    companies_by_ticker = {TARGET: {"revenue_ttm_usd_mm": 300.0}}

    def add(ticker, residual, model, customer, revenue):
        residuals[ticker] = residual
        llm_features[ticker] = _llm(model, customer)
        companies_by_ticker[ticker] = {"revenue_ttm_usd_mm": revenue}

    peers = []
    for i, res in enumerate([0.20, 0.28, 0.36, 0.44, 0.52, 0.60]):
        t = f"PEER{i}"
        add(t, res, "manufacturing", "B2B", 200.0 + 40 * i)
        peers.append(t)

    for i, res in enumerate([0.10, 0.14, 0.18, 0.22, 0.26]):
        add(f"SAAS{i}", res, "SaaS", "B2B", 300.0)            # wrong business_model
    for i, res in enumerate([0.12, 0.16, 0.20, 0.24, 0.30]):
        add(f"B2G{i}", res, "manufacturing", "B2G", 300.0)    # wrong customer_type
    for i, res in enumerate([0.08, 0.11, 0.15, 0.19]):
        add(f"GIANT{i}", res, "manufacturing", "B2B", 30000.0)  # ~100x size mismatch

    company_scores = pd.DataFrame({"residual_abs": residuals})
    ground_truth = {TARGET: peers}
    return ground_truth, company_scores, llm_features, companies_by_ticker


def test_precision_at_k_stays_above_floor():
    ground_truth, company_scores, llm_features, companies_by_ticker = _build_pool()

    result = evaluator._evaluate_precision_at_k(
        ground_truth, company_scores, llm_features, companies_by_ticker,
        PENALTIES, EMBEDDING_MODEL,
    )

    assert result["mean"] >= PRECISION_FLOOR, (
        f"Selection precision regressed to {result['mean']:.2f}, "
        f"below the {PRECISION_FLOOR:.2f} floor — a soft-penalty path likely broke."
    )
