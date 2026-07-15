import csv

import pandas as pd

import src.reporter as reporter


def _company(ticker, **overrides):
    base = {
        "ticker": ticker,
        "company_name": f"{ticker} Inc.",
        "market_cap_usd_mm": 500.0,
        "revenue_ttm_usd_mm": 200.0,
        "ebitda_margin": 0.20,
        "gross_margin": 0.35,
        "revenue_cagr_3yr": 0.05,
        "net_debt_ebitda": 1.5,
        "capex_revenue": 0.04,
        "ev_ebitda": 12.0,
        "ev_revenue": 2.4,
        "gics_sector": "20",
        "business_description": "test",
        "description_source": "edgar",
        "fetch_timestamp": "2026-06-16T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def _llm(business_model="manufacturing", low_confidence_flag=False, **overrides):
    base = {
        "business_model": business_model,
        "revenue_recurrence": "medium",
        "customer_type": "B2B",
        "capital_intensity": "moderate",
        "primary_value_driver": "scale",
        "sub_sector_description": "test sub-sector",
        "confidence": 4,
        "judge_score": 4,
        "judge_reason": "ok",
        "low_confidence_flag": low_confidence_flag,
        "extraction_failed": False,
    }
    base.update(overrides)
    return base


def _build_sample(n=30, n_matching=15):
    companies = []
    llm_features = {}
    scores_rows = {}
    for i in range(n):
        ticker = f"T{i:03d}"
        matching = i < n_matching
        companies.append(_company(ticker, ev_ebitda=10.0 + i * 0.1))
        llm_features[ticker] = _llm(business_model="manufacturing" if matching else "services")
        scores_rows[ticker] = {
            "ev_ebitda_actual": 10.0 + i * 0.1,
            "ev_ebitda_predicted": 10.0 + i * 0.1 + 0.5,
            "residual_abs": 0.5 + i * 0.01,
            "residual_pct": 0.05,
        }
    company_scores = pd.DataFrame(scores_rows).T
    return companies, llm_features, company_scores


def _scorer_results(company_scores):
    feature_importance = pd.DataFrame({
        "feature": ["ebitda_margin", "revenue_cagr_3yr", "gross_margin", "capex_revenue", "net_debt_ebitda"],
        "mean_abs_shap": [0.10, 0.09, 0.08, 0.07, 0.06],
    })
    return {
        "model": None,
        "feature_columns": [],
        "cv_rmse_log": 0.4,
        "cv_rmse_multiple_space": 5.0,
        "cv_mae_median": 5.0,
        "feature_importance": feature_importance,
        "company_scores": company_scores,
        "target_prediction": {
            "predicted_ev_ebitda": 15.0,
            "range_low": 10.0,
            "range_high": 22.0,
            "cv_rmse_log": 0.4,
            "cv_mae_median": 5.0,
        },
    }


def _sample_config():
    return {
        "target_company": {
            "name": "Example Manufacturing Co.",
            "description": "A test target company.",
            "gics_sector": "20",
            "revenue_usd_mm": 150,
            "ebitda_margin_estimate": 0.18,
        },
    }


def _imputation_medians():
    return {
        "revenue_ttm_log": 5.0,
        "ebitda_margin": 0.2,
        "gross_margin": 0.35,
        "revenue_cagr_3yr": 0.05,
        "net_debt_ebitda": 1.5,
        "capex_revenue": 0.04,
    }


def test_top15_selected_correctly():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm(business_model="manufacturing")

    result = reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), _sample_config(),
    )

    with open(result["csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 15


def test_csv_file_created():
    companies, llm_features, company_scores = _build_sample()
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm()

    reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), _sample_config(),
    )

    assert reporter.CSV_PATH.exists()


def test_size_and_customer_type_mismatch_excluded():
    # Target: $150mm revenue, B2B. GIANT has the best residual fit by far
    # but is a $90B-revenue, B2G company — both a scale and customer-type
    # mismatch. CLOSE is a worse fit but matches the target's profile.
    companies = [
        _company("CLOSE", revenue_ttm_usd_mm=180.0),
        _company("GIANT", revenue_ttm_usd_mm=90_000.0),
    ]
    llm_features = {
        "CLOSE": _llm(business_model="manufacturing", customer_type="B2B"),
        "GIANT": _llm(business_model="manufacturing", customer_type="B2G"),
    }
    company_scores = pd.DataFrame({
        "residual_abs": {"CLOSE": 0.5, "GIANT": 0.05},
    })

    top1 = reporter._select_top_15(
        company_scores, llm_features, {c["ticker"]: c for c in companies},
        target_business_model="manufacturing", target_customer_type="B2B", target_revenue=150.0,
        k=1,
    )

    assert top1 == ["CLOSE"]


def test_html_file_created():
    companies, llm_features, company_scores = _build_sample()
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm()

    reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), _sample_config(),
    )

    assert reporter.HTML_PATH.exists()
    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "Example Manufacturing Co." in html_text
