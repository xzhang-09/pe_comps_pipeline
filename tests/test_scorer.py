import numpy as np
import pandas as pd
import xgboost as xgb

import src.scorer as scorer

N_ROWS = 60
FINANCIAL_COLUMNS = (
    "revenue_ttm_log", "ebitda_margin", "gross_margin",
    "revenue_cagr_3yr", "net_debt_ebitda", "capex_revenue",
)


def _synthetic_feature_matrix(n=N_ROWS, seed=42):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n)]

    ebitda_margin = rng.uniform(0.05, 0.35, n)
    revenue_cagr_3yr = rng.normal(0.05, 0.05, n)
    df = pd.DataFrame({
        "revenue_ttm_log": rng.normal(5, 1, n),
        "ebitda_margin": ebitda_margin,
        "gross_margin": rng.uniform(0.2, 0.6, n),
        "revenue_cagr_3yr": revenue_cagr_3yr,
        "net_debt_ebitda": rng.normal(2.0, 1.0, n),
        "capex_revenue": rng.uniform(0.01, 0.08, n),
        "business_model_services": rng.integers(0, 2, n),
    }, index=tickers)

    noise = rng.normal(0, 0.3, n)
    df["ev_ebitda_log"] = 2.3 + 0.5 * ebitda_margin + 0.3 * revenue_cagr_3yr + noise

    return df


def _medians(feature_matrix):
    return {col: float(feature_matrix[col].median()) for col in FINANCIAL_COLUMNS}


def _target_config():
    return {"revenue_usd_mm": 150, "ebitda_margin_estimate": 0.18}


def _target_llm_features():
    return {
        "business_model": "manufacturing",
        "revenue_recurrence": "high",
        "customer_type": "B2B",
        "capital_intensity": "asset_heavy",
        "primary_value_driver": "scale",
    }


def _run(fm=None):
    fm = fm if fm is not None else _synthetic_feature_matrix()
    return scorer.run(fm, _target_config(), _target_llm_features(), _medians(fm))


def test_company_scores_has_all_tickers():
    fm = _synthetic_feature_matrix()
    result = _run(fm)

    assert len(result["company_scores"]) == len(fm)


def test_cv_rmse_is_positive():
    result = _run()

    assert isinstance(result["cv_rmse_multiple_space"], float)
    assert result["cv_rmse_multiple_space"] > 0


def test_feature_importance_has_correct_shape():
    fm = _synthetic_feature_matrix()
    result = _run(fm)

    expected_feature_count = len(fm.columns) - 1  # exclude ev_ebitda_log
    assert len(result["feature_importance"]) == expected_feature_count
    assert "ev_ebitda_log" not in set(result["feature_importance"]["feature"])


def test_model_saved_to_disk():
    _run()

    assert scorer.MODEL_PATH.exists()


def test_model_loaded_on_second_run(mocker):
    mock_fit = mocker.spy(xgb.XGBRegressor, "fit")

    fm = _synthetic_feature_matrix()
    _run(fm)
    calls_after_first_run = mock_fit.call_count

    _run(fm)

    assert mock_fit.call_count == calls_after_first_run
