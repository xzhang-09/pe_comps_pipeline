import numpy as np
import pytest

import src.feature_builder as feature_builder


def _company(ticker, ev_ebitda=12.0, revenue=100.0, ebitda_margin=0.2, gross_margin=0.4,
             revenue_cagr_3yr=0.05, net_debt_ebitda=1.5, capex_revenue=0.03, **overrides):
    base = {
        "ticker": ticker,
        "ev_ebitda": ev_ebitda,
        "revenue_ttm_usd_mm": revenue,
        "ebitda_margin": ebitda_margin,
        "gross_margin": gross_margin,
        "revenue_cagr_3yr": revenue_cagr_3yr,
        "net_debt_ebitda": net_debt_ebitda,
        "capex_revenue": capex_revenue,
    }
    base.update(overrides)
    return base


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
    # All 3 companies share business_model="manufacturing" but the group is
    # below MIN_GROUP_SIZE_FOR_IMPUTATION, so this exercises the fallback to
    # the global median rather than a real per-group median.
    companies = [
        _company("AAA", ebitda_margin=0.20),
        _company("BBB", ebitda_margin=0.30),
        _company("CCC", ebitda_margin=None),
    ]
    llm_features = {c["ticker"]: _llm() for c in companies}

    feature_matrix, _, medians = feature_builder.build(companies, llm_features)

    assert feature_matrix.loc["CCC", "ebitda_margin"] == pytest.approx(0.25)
    assert medians["global"]["ebitda_margin"] == pytest.approx(0.25)
    assert feature_builder.median_for(medians, "ebitda_margin", "manufacturing") == pytest.approx(0.25)


def test_group_median_used_when_group_large_enough():
    # 5 manufacturing companies (meets MIN_GROUP_SIZE_FOR_IMPUTATION) with a
    # distinctly different ebitda_margin than 5 services companies — the
    # missing value in a manufacturing row should be filled with the
    # manufacturing group's own median, not the global median across both.
    manufacturing = [
        _company(f"MFG{i}", ebitda_margin=0.10 + i * 0.01) for i in range(5)
    ] + [_company("MFG_MISSING", ebitda_margin=None)]
    services = [_company(f"SVC{i}", ebitda_margin=0.50 + i * 0.01) for i in range(5)]
    companies = manufacturing + services
    llm_features = {
        **{c["ticker"]: _llm(business_model="manufacturing") for c in manufacturing},
        **{c["ticker"]: _llm(business_model="services") for c in services},
    }

    feature_matrix, _, medians = feature_builder.build(companies, llm_features)

    manufacturing_group_median = feature_builder.median_for(medians, "ebitda_margin", "manufacturing")
    global_median = medians["global"]["ebitda_margin"]
    assert manufacturing_group_median != pytest.approx(global_median)
    assert feature_matrix.loc["MFG_MISSING", "ebitda_margin"] == pytest.approx(manufacturing_group_median)


def test_group_median_falls_back_to_global_when_group_too_small():
    # Only 2 manufacturing companies — below MIN_GROUP_SIZE_FOR_IMPUTATION —
    # so the missing value should fall back to the global median across
    # every company, not a noisy 2-company "group median".
    companies = [
        _company("MFG1", ebitda_margin=0.10),
        _company("MFG_MISSING", ebitda_margin=None),
        _company("SVC1", ebitda_margin=0.50),
        _company("SVC2", ebitda_margin=0.60),
    ]
    llm_features = {
        "MFG1": _llm(business_model="manufacturing"),
        "MFG_MISSING": _llm(business_model="manufacturing"),
        "SVC1": _llm(business_model="services"),
        "SVC2": _llm(business_model="services"),
    }

    feature_matrix, _, medians = feature_builder.build(companies, llm_features)

    global_median = medians["global"]["ebitda_margin"]
    assert feature_matrix.loc["MFG_MISSING", "ebitda_margin"] == pytest.approx(global_median)


def test_feature_matrix_has_only_financial_columns_and_label():
    companies = [_company("AAA"), _company("BBB")]
    llm_features = {c["ticker"]: _llm() for c in companies}

    feature_matrix, _, _ = feature_builder.build(companies, llm_features)

    expected = set(feature_builder.FINANCIAL_FEATURE_COLUMNS) | {feature_builder.LABEL_COLUMN}
    assert set(feature_matrix.columns) == expected
