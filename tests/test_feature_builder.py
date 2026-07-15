import numpy as np
import pytest

import src.feature_builder as feature_builder


def _company(ticker, ev_ebitda=12.0, revenue=100.0, ebitda_margin=0.2, gross_margin=0.4,
             revenue_cagr_3yr=0.05, net_debt_ebitda=1.5, capex_revenue=0.03):
    return {
        "ticker": ticker,
        "ev_ebitda": ev_ebitda,
        "revenue_ttm_usd_mm": revenue,
        "ebitda_margin": ebitda_margin,
        "gross_margin": gross_margin,
        "revenue_cagr_3yr": revenue_cagr_3yr,
        "net_debt_ebitda": net_debt_ebitda,
        "capex_revenue": capex_revenue,
    }


def _llm(business_model="manufacturing", **overrides):
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
        "low_confidence_flag": False,
        "extraction_failed": False,
    }
    base.update(overrides)
    return base


def test_rows_missing_label_dropped():
    companies = [
        _company("AAA", ev_ebitda=None),
        _company("BBB", ev_ebitda=None),
        _company("CCC", ev_ebitda=10.0),
    ]
    llm_features = {c["ticker"]: _llm() for c in companies}

    feature_matrix, _, _ = feature_builder.build(companies, llm_features)

    assert "AAA" not in feature_matrix.index
    assert "BBB" not in feature_matrix.index
    assert "CCC" in feature_matrix.index


def test_revenue_log_transform_applied():
    companies = [_company("AAA", revenue=100.0)]
    llm_features = {"AAA": _llm()}

    feature_matrix, _, _ = feature_builder.build(companies, llm_features)

    assert feature_matrix.loc["AAA", "revenue_ttm_log"] == pytest.approx(np.log1p(100.0))


def test_ev_ebitda_log_transform_is_label():
    companies = [_company("AAA", ev_ebitda=12.0)]
    llm_features = {"AAA": _llm()}

    feature_matrix, _, _ = feature_builder.build(companies, llm_features)

    assert feature_matrix.loc["AAA", "ev_ebitda_log"] == pytest.approx(np.log1p(12.0))


def test_nan_imputed_with_median():
    companies = [
        _company("AAA", ebitda_margin=0.20),
        _company("BBB", ebitda_margin=0.30),
        _company("CCC", ebitda_margin=None),
    ]
    llm_features = {c["ticker"]: _llm() for c in companies}

    feature_matrix, _, medians = feature_builder.build(companies, llm_features)

    assert feature_matrix.loc["CCC", "ebitda_margin"] == pytest.approx(0.25)
    assert medians["ebitda_margin"] == pytest.approx(0.25)


def test_llm_fields_one_hot_encoded():
    companies = [_company("AAA"), _company("BBB")]
    llm_features = {
        "AAA": _llm(business_model="manufacturing"),
        "BBB": _llm(business_model="services"),
    }

    feature_matrix, _, _ = feature_builder.build(companies, llm_features)

    assert any(col.startswith("business_model_") for col in feature_matrix.columns)
