import pandas as pd
import pytest

import eval.evaluator as evaluator


@pytest.fixture(autouse=True)
def no_real_embedding_calls(mocker):
    """_select_top_k calls _subsector_similarities(), which hits the real
    OpenAI API via embed_texts(). Block that by default for every test in
    this module (mirrors tests/test_reporter.py's fixture of the same
    name) — tests that exercise the sub-sector penalty itself override
    this with their own mocker.patch(...) of the same target."""
    mocker.patch("src.report_selection.llm_analyzer.embed_texts", return_value=None)


def _llm(business_model="manufacturing", customer_type="B2B", low_confidence_flag=False):
    return {
        "business_model": business_model,
        "revenue_recurrence": "medium",
        "customer_type": customer_type,
        "capital_intensity": "moderate",
        "primary_value_driver": "scale",
        "sub_sector_description": "test",
        "confidence": 4,
        "judge_score": 4,
        "judge_reason": "ok",
        "low_confidence_flag": low_confidence_flag,
        "extraction_failed": False,
    }


def _company(ticker, revenue=200.0):
    return {"ticker": ticker, "revenue_ttm_usd_mm": revenue}


def _company_scores(tickers_and_residuals: dict) -> pd.DataFrame:
    return pd.DataFrame({"residual_abs": tickers_and_residuals})


# Distance-unit penalties (added to residual_abs, not an ordinal rank) — must
# match the scale used in reporter/evaluator. See _select_top_k.
DEFAULT_PENALTIES = {
    "business_model_penalty": 0.6,
    "customer_type_penalty": 0.5,
    "subsector_similarity_threshold": 0.5,
    "subsector_mismatch_penalty": 0.4,
    "size_penalty_free_log10_range": 1.0,
    "size_penalty_per_extra_log10": 1.0,
}
EMBEDDING_MODEL = "text-embedding-3-small"


def test_precision_at_k_correct_calculation():
    company_scores = _company_scores({"AAA": 0.1, "HON": 0.2, "EMR": 0.3})
    llm_features = {t: _llm() for t in ("AAA", "HON", "EMR")}
    companies_by_ticker = {t: _company(t) for t in ("AAA", "HON", "EMR")}

    precision = evaluator._precision_at_k(
        "AAA", ["HON", "EMR", "SWK"], company_scores, llm_features, companies_by_ticker,
        DEFAULT_PENALTIES, EMBEDDING_MODEL,
    )

    assert precision == 2 / 3


def test_precision_zero_when_no_overlap():
    company_scores = _company_scores({"AAA": 0.1, "MMM": 0.2, "GE": 0.3})
    llm_features = {t: _llm() for t in ("AAA", "MMM", "GE")}
    companies_by_ticker = {t: _company(t) for t in ("AAA", "MMM", "GE")}

    precision = evaluator._precision_at_k(
        "AAA", ["HON", "EMR"], company_scores, llm_features, companies_by_ticker,
        DEFAULT_PENALTIES, EMBEDDING_MODEL,
    )

    assert precision == 0.0


def test_precision_one_when_full_overlap():
    company_scores = _company_scores({"AAA": 0.1, "HON": 0.2, "EMR": 0.3})
    llm_features = {t: _llm() for t in ("AAA", "HON", "EMR")}
    companies_by_ticker = {t: _company(t) for t in ("AAA", "HON", "EMR")}

    precision = evaluator._precision_at_k(
        "AAA", ["HON", "EMR"], company_scores, llm_features, companies_by_ticker,
        DEFAULT_PENALTIES, EMBEDDING_MODEL,
    )

    assert precision == 1.0


def test_target_excluded_from_evaluation():
    company_scores = _company_scores({"MMM": 0.1, "HON": 0.2, "EMR": 0.3})
    llm_features = {t: _llm() for t in ("MMM", "HON", "EMR")}
    companies_by_ticker = {t: _company(t) for t in ("MMM", "HON", "EMR")}

    # MMM erroneously appears in its own ground truth peer list — it must
    # be excluded from both sides before computing precision.
    precision = evaluator._precision_at_k(
        "MMM", ["HON", "MMM", "EMR"], company_scores, llm_features, companies_by_ticker,
        DEFAULT_PENALTIES, EMBEDDING_MODEL,
    )

    assert precision == 1.0


def test_size_mismatch_excludes_oversized_company():
    # AAA (target, revenue 200) vs HON (revenue 200, close) and GIANT
    # (revenue 200,000 — 1000x AAA). GIANT has the best residual fit, but
    # its scale mismatch should push it below HON despite the worse fit.
    company_scores = _company_scores({"AAA": 0.5, "HON": 0.4, "GIANT": 0.1})
    llm_features = {t: _llm() for t in ("AAA", "HON", "GIANT")}
    companies_by_ticker = {
        "AAA": _company("AAA", revenue=200.0),
        "HON": _company("HON", revenue=200.0),
        "GIANT": _company("GIANT", revenue=200_000.0),
    }

    top1 = evaluator._select_top_k(
        "AAA", company_scores, llm_features, companies_by_ticker, DEFAULT_PENALTIES, EMBEDDING_MODEL, k=1,
    )

    assert top1 == ["HON"]


def test_subsector_mismatch_excludes_low_similarity_candidate(mocker):
    # FAR has the best residual fit but its sub_sector_description is
    # orthogonal (cosine similarity 0) to the target's; CLOSE is a slightly
    # worse financial fit but matches sub-sector closely. Gap (0.20 vs 0.05) is
    # within one subsector penalty (0.4) so the mismatch flips the order.
    company_scores = _company_scores({"AAA": 0.5, "CLOSE": 0.20, "FAR": 0.05})
    llm_features = {
        "AAA": _llm(),
        "CLOSE": {**_llm(), "sub_sector_description": "automotive OEM parts manufacturer"},
        "FAR": {**_llm(), "sub_sector_description": "semiconductor process equipment maker"},
    }
    llm_features["AAA"]["sub_sector_description"] = "automotive OEM parts target description"
    companies_by_ticker = {t: _company(t) for t in ("AAA", "CLOSE", "FAR")}
    mocker.patch(
        "src.report_selection.llm_analyzer.embed_texts",
        return_value=[[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]],
    )

    top1 = evaluator._select_top_k(
        "AAA", company_scores, llm_features, companies_by_ticker, DEFAULT_PENALTIES, EMBEDDING_MODEL, k=1,
    )

    assert top1 == ["CLOSE"]


def test_eval_report_written_to_file():
    eval_results = {
        "precision_at_k": {"mean": 0.5, "median": 0.5, "min": 0.2, "max": 0.8, "per_company": {"MMM": 0.5}},
        "llm_consistency": {
            "business_model_agreement": 0.9,
            "revenue_recurrence_agreement": 0.8,
            "customer_type_agreement": 0.85,
            "capital_intensity_agreement": 0.7,
            "primary_value_driver_agreement": 0.6,
        },
        "n_test_companies": 1,
        "n_consistency_samples": 30,
    }

    evaluator.generate_eval_report(eval_results)

    assert evaluator.RESULTS_PATH.exists()


def test_load_manual_deals_validates_required_shape(tmp_path):
    path = tmp_path / "manual_deals.json"
    path.write_text(
        """
        [
          {
            "deal_id": "demo-2026",
            "target_ticker": "TGT",
            "target_name": "Target Co",
            "filing_url": "https://www.sec.gov/example",
            "advisor": "Example Bank",
            "filing_date": "2026-01-15",
            "selected_companies": [
              {"ticker": "AAA", "company_name": "AAA Inc.", "still_public": true}
            ]
          }
        ]
        """,
        encoding="utf-8",
    )

    deals = evaluator.load_manual_deals(path)

    assert deals[0]["target_ticker"] == "TGT"
    assert deals[0]["selected_companies"][0]["ticker"] == "AAA"


def test_published_manual_deals_dataset_satisfies_benchmark_requirements():
    deals = evaluator.load_manual_deals()

    summary = evaluator.validate_manual_deals_benchmark(deals)

    assert summary == {
        "n_deals": 10,
        "reviewed_deals": 10,
        "target_sic_codes": ["2430", "2891", "3310", "3312", "3420", "3442", "3490", "3559", "3728"],
        "eligible_public_comps": 85,
        "excluded_delisted_comps": 11,
    }


def test_manual_deals_benchmark_validation_rejects_incomplete_deal():
    deals = [
        {
            "deal_id": "incomplete-2026",
            "target_ticker": "TGT",
            "target_name": "Target Co",
            "target_cik": "0000000000",
            "target_sic": "3559",
            "business_description": "Industrial manufacturer.",
            "target_financials": {"revenue_usd_mm": 100.0},
            "filing_url": "https://www.sec.gov/example",
            "filing_date": "2026-01-15",
            "advisor": "Example Bank",
            "selected_companies": [
                {"ticker": "AAA", "company_name": "AAA Inc."},
            ],
            "review_status": "needs_review",
        }
    ]

    with pytest.raises(ValueError, match="review_status must be reviewed"):
        evaluator.validate_manual_deals_benchmark(deals, min_deals=1, max_deals=1)


def test_manual_ground_truth_eval_filters_delisted_and_splits_misses():
    deal = {
        "deal_id": "demo-2026",
        "target_ticker": "TGT",
        "target_name": "Target Co",
        "filing_url": "https://www.sec.gov/example",
        "advisor": "Example Bank",
        "filing_date": "2026-01-15",
        "selected_companies": [
            {"ticker": "HIT", "company_name": "Hit Co", "still_public": True},
            {"ticker": "RANKMISS", "company_name": "Rank Miss Co", "still_public": True},
            {"ticker": "DISCOVERYMISS", "company_name": "Discovery Miss Co", "still_public": True},
            {"ticker": "DELISTED", "company_name": "Delisted Co", "still_public": False},
        ],
    }
    company_scores = _company_scores({"TGT": 0.0, "HIT": 0.1, "DISTRACTOR": 0.2, "RANKMISS": 0.9})
    llm_features = {t: _llm() for t in ("TGT", "HIT", "DISTRACTOR", "RANKMISS")}
    companies_by_ticker = {t: _company(t) for t in ("TGT", "HIT", "DISTRACTOR", "RANKMISS")}
    config = {"scorer": {"ranking_penalties": DEFAULT_PENALTIES}, "llm": {"embedding_model": EMBEDDING_MODEL}}

    results = evaluator.run_manual_ground_truth_evaluation(
        [deal], company_scores, llm_features, companies_by_ticker, config, k=2,
    )

    row = results["per_deal"][0]
    assert row["eligible_ground_truth_tickers"] == ["HIT", "RANKMISS", "DISCOVERYMISS"]
    assert row["selected_tickers"] == ["HIT", "DISTRACTOR"]
    assert row["precision"] == 1 / 3
    assert row["missed_not_in_universe"] == ["DISCOVERYMISS"]
    assert row["missed_not_selected"] == ["RANKMISS"]
    assert results["mean_precision"] == 1 / 3


def test_manual_ground_truth_report_written_to_file():
    results = {
        "mean_precision": 1 / 3,
        "median_precision": 1 / 3,
        "n_deals": 1,
        "k": 2,
        "per_deal": [
            {
                "deal_id": "demo-2026",
                "target_ticker": "TGT",
                "target_name": "Target Co",
                "precision": 1 / 3,
                "hits": ["HIT"],
                "selected_tickers": ["HIT", "DISTRACTOR"],
                "eligible_ground_truth_tickers": ["HIT", "RANKMISS", "DISCOVERYMISS"],
                "excluded_delisted_tickers": ["DELISTED"],
                "missed_not_in_universe": ["DISCOVERYMISS"],
                "missed_not_selected": ["RANKMISS"],
                "filing_url": "https://www.sec.gov/example",
                "advisor": "Example Bank",
                "filing_date": "2026-01-15",
            }
        ],
    }

    text = evaluator.generate_manual_ground_truth_report(results)

    assert evaluator.RESULTS_PATH.exists()
    assert "Manual Ground Truth Evaluation" in text
    assert "Precision@2" in text
    assert "DISCOVERYMISS" in text


def test_manual_ground_truth_eval_excludes_non_us_filers_from_denominator():
    deal = {
        "deal_id": "demo-2026",
        "target_ticker": "TGT",
        "target_name": "Target Co",
        "filing_url": "https://www.sec.gov/example",
        "advisor": "Example Bank",
        "filing_date": "2026-01-15",
        "selected_companies": [
            {"ticker": "HIT", "company_name": "Hit Co", "still_public": True, "us_filer": True},
            {"ticker": "SMIN.L", "company_name": "Foreign Plc", "still_public": True, "us_filer": False},
            {"ticker": "LEGACY", "company_name": "Pre-Flag Co", "still_public": True},
            {"ticker": "DELISTED", "company_name": "Delisted Co", "still_public": False, "us_filer": False},
        ],
    }
    company_scores = _company_scores({"TGT": 0.0, "HIT": 0.1, "LEGACY": 0.2})
    llm_features = {t: _llm() for t in ("TGT", "HIT", "LEGACY")}
    companies_by_ticker = {t: _company(t) for t in ("TGT", "HIT", "LEGACY")}
    config = {"scorer": {"ranking_penalties": DEFAULT_PENALTIES}, "llm": {"embedding_model": EMBEDDING_MODEL}}

    results = evaluator.run_manual_ground_truth_evaluation(
        [deal], company_scores, llm_features, companies_by_ticker, config, k=2,
    )

    row = results["per_deal"][0]
    # Foreign-listed comp is out of the denominator (data contract, not a miss);
    # a comp missing the us_filer key predates the flag and stays eligible.
    assert row["eligible_ground_truth_tickers"] == ["HIT", "LEGACY"]
    assert row["excluded_non_us_filer_tickers"] == ["SMIN.L"]
    assert row["excluded_delisted_tickers"] == ["DELISTED"]
    assert row["precision"] == 1.0
