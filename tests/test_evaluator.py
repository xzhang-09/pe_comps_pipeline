import pandas as pd

import eval.evaluator as evaluator


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


def test_precision_at_k_correct_calculation():
    company_scores = _company_scores({"AAA": 0.1, "HON": 0.2, "EMR": 0.3})
    llm_features = {t: _llm() for t in ("AAA", "HON", "EMR")}
    companies_by_ticker = {t: _company(t) for t in ("AAA", "HON", "EMR")}

    precision = evaluator._precision_at_k("AAA", ["HON", "EMR", "SWK"], company_scores, llm_features, companies_by_ticker)

    assert precision == 2 / 3


def test_precision_zero_when_no_overlap():
    company_scores = _company_scores({"AAA": 0.1, "MMM": 0.2, "GE": 0.3})
    llm_features = {t: _llm() for t in ("AAA", "MMM", "GE")}
    companies_by_ticker = {t: _company(t) for t in ("AAA", "MMM", "GE")}

    precision = evaluator._precision_at_k("AAA", ["HON", "EMR"], company_scores, llm_features, companies_by_ticker)

    assert precision == 0.0


def test_precision_one_when_full_overlap():
    company_scores = _company_scores({"AAA": 0.1, "HON": 0.2, "EMR": 0.3})
    llm_features = {t: _llm() for t in ("AAA", "HON", "EMR")}
    companies_by_ticker = {t: _company(t) for t in ("AAA", "HON", "EMR")}

    precision = evaluator._precision_at_k("AAA", ["HON", "EMR"], company_scores, llm_features, companies_by_ticker)

    assert precision == 1.0


def test_target_excluded_from_evaluation():
    company_scores = _company_scores({"MMM": 0.1, "HON": 0.2, "EMR": 0.3})
    llm_features = {t: _llm() for t in ("MMM", "HON", "EMR")}
    companies_by_ticker = {t: _company(t) for t in ("MMM", "HON", "EMR")}

    # MMM erroneously appears in its own ground truth peer list — it must
    # be excluded from both sides before computing precision.
    precision = evaluator._precision_at_k("MMM", ["HON", "MMM", "EMR"], company_scores, llm_features, companies_by_ticker)

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

    top1 = evaluator._select_top_k("AAA", company_scores, llm_features, companies_by_ticker, k=1)

    assert top1 == ["HON"]


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
