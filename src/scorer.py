import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import KFold

from src import get_logger
from src.feature_builder import LABEL_COLUMN, LLM_CATEGORICAL_FIELDS

logger = get_logger(__name__)

MODEL_DIR = Path("data/models")
MODEL_PATH = MODEL_DIR / "xgb_model.json"
MEDIANS_PATH = MODEL_DIR / "imputation_medians.json"
# CV-derived outputs (feature importance, company scores, CV RMSE) are tied
# to a specific training run, so they're cached alongside the model itself —
# loading the model without these would mean recomputing them via more
# fit() calls, defeating the point of skipping retraining.
SCORER_CACHE_PATH = MODEL_DIR / "scorer_cache.json"

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    early_stopping_rounds=30,
    random_state=42,
)

N_CV_FOLDS = 5
TOP_N_FEATURE_IMPORTANCE_CHECK = 5
SANITY_CHECK_FEATURES = ("ebitda_margin", "revenue_cagr_3yr")
TARGET_RANGE_MULTIPLIER = 1.5


def _make_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(**XGB_PARAMS)


def _cross_validate(X: pd.DataFrame, y: pd.Series) -> tuple[float, float, np.ndarray]:
    """
    Returns (cv_rmse_log, cv_mae_median, oof_predictions).

    cv_rmse_log is RMSE in log1p space — the actual model-quality metric,
    and the right space to build a symmetric +/- confidence band in (then
    exponentiate the *bounds*, not the RMSE itself — see predict_target).

    cv_mae_median is the median absolute error in multiple space, for
    human-readable reporting. It is NOT numpy.expm1(cv_rmse_log) — that
    transform doesn't carry over between an aggregate statistic and a
    single value, and empirically understated the real error by ~24x here
    (0.57x vs. a true direct RMSE of ~14x, which itself is skewed by a
    handful of companies with genuinely extreme multiples). The median is
    robust to those outliers and matches the typical per-company residual.
    """
    kfold = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
    oof_predictions = np.full(len(y), np.nan)

    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = _make_model()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_predictions[val_idx] = model.predict(X_val)

    actual_log = y.to_numpy()
    cv_rmse_log = float(np.sqrt(np.mean((oof_predictions - actual_log) ** 2)))

    actual_multiple = np.expm1(actual_log)
    predicted_multiple = np.expm1(oof_predictions)
    cv_mae_median = float(np.median(np.abs(actual_multiple - predicted_multiple)))

    return cv_rmse_log, cv_mae_median, oof_predictions


def _train_final_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    model = _make_model()
    # Spec calls for training on ALL data with no held-out set. early_stopping_rounds
    # requires an eval_set to function at all, so we pass the training data back to
    # itself purely to satisfy that API requirement (not as a real validation signal).
    model.fit(X, y, eval_set=[(X, y)], verbose=False)
    return model


def _feature_importance(model: xgb.XGBRegressor, X: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    top_features = set(importance["feature"].head(TOP_N_FEATURE_IMPORTANCE_CHECK))
    if not (set(SANITY_CHECK_FEATURES) & top_features):
        logger.warning("Expected financial features not in top 5 — check data quality")

    return importance


def _company_scores(ev_ebitda_raw: pd.Series, oof_log_predictions: np.ndarray) -> pd.DataFrame:
    actual = ev_ebitda_raw.to_numpy()
    predicted = np.expm1(oof_log_predictions)
    residual_abs = np.abs(actual - predicted)
    residual_pct = residual_abs / actual

    return pd.DataFrame({
        "ev_ebitda_actual": actual,
        "ev_ebitda_predicted": predicted,
        "residual_abs": residual_abs,
        "residual_pct": residual_pct,
    }, index=ev_ebitda_raw.index)


def _save_artifacts(
    model: xgb.XGBRegressor, imputation_medians: dict, cv_rmse_log: float,
    cv_mae_median: float, feature_importance: pd.DataFrame, company_scores: pd.DataFrame,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))

    with open(MEDIANS_PATH, "w", encoding="utf-8") as f:
        json.dump(imputation_medians, f, indent=2)

    cache = {
        "cv_rmse_log": cv_rmse_log,
        "cv_mae_median": cv_mae_median,
        "feature_importance": feature_importance.to_dict(orient="records"),
        "company_scores": company_scores.reset_index(names="ticker").to_dict(orient="records"),
    }
    with open(SCORER_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _load_artifacts() -> tuple[xgb.XGBRegressor, float, float, pd.DataFrame, pd.DataFrame]:
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))

    with open(SCORER_CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    feature_importance = pd.DataFrame(cache["feature_importance"])
    company_scores = pd.DataFrame(cache["company_scores"]).set_index("ticker")

    return model, cache["cv_rmse_log"], cache["cv_mae_median"], feature_importance, company_scores


def _existing_artifacts_present() -> bool:
    return MODEL_PATH.exists() and SCORER_CACHE_PATH.exists()


def _one_hot_target_row(target_llm_features: dict, feature_columns: list) -> dict:
    """Map the target's LLM categorical fields onto the exact one-hot column
    structure used during training. Columns not matched stay 0 — this
    correctly represents both "unknown" values and the drop_first reference
    category."""
    llm_columns = [
        col for col in feature_columns
        if any(col.startswith(f"{field}_") for field in LLM_CATEGORICAL_FIELDS)
    ]
    encoded = {col: 0 for col in llm_columns}

    for field in LLM_CATEGORICAL_FIELDS:
        value = target_llm_features.get(field) or "unknown"
        col_name = f"{field}_{value}"
        if col_name in encoded:
            encoded[col_name] = 1

    return encoded


def predict_target(
    model: xgb.XGBRegressor,
    target_config: dict,
    target_llm_features: dict,
    feature_columns: list,
    imputation_medians: dict,
    cv_rmse_log: float,
    cv_mae_median: float,
) -> dict:
    """
    Predict EV/EBITDA for the private target company.

    The +/- band is applied in log space (where cv_rmse_log was actually
    computed) and only then exponentiated into each bound separately. This
    is what makes the resulting range correctly asymmetric — expm1 is
    convex, so the same log-space delta maps to a bigger move on the high
    side than the low side, matching the real right-skew of EV/EBITDA
    multiples. Applying a flat +/- in multiple space (the original approach)
    loses that and used a badly miscalibrated delta besides.
    """
    revenue = target_config.get("revenue_usd_mm")
    revenue_log = np.log1p(revenue) if revenue is not None else imputation_medians.get("revenue_ttm_log")

    ebitda_margin = target_config.get("ebitda_margin_estimate")
    if ebitda_margin is None:
        ebitda_margin = imputation_medians.get("ebitda_margin")

    row = {
        "revenue_ttm_log": revenue_log,
        "ebitda_margin": ebitda_margin,
        "gross_margin": imputation_medians.get("gross_margin"),
        "revenue_cagr_3yr": imputation_medians.get("revenue_cagr_3yr"),
        "net_debt_ebitda": imputation_medians.get("net_debt_ebitda"),
        "capex_revenue": imputation_medians.get("capex_revenue"),
    }
    row.update(_one_hot_target_row(target_llm_features, feature_columns))

    X_target = pd.DataFrame([row], columns=feature_columns)
    predicted_log = float(model.predict(X_target)[0])
    predicted = float(np.expm1(predicted_log))

    range_low = float(np.expm1(predicted_log - TARGET_RANGE_MULTIPLIER * cv_rmse_log))
    range_high = float(np.expm1(predicted_log + TARGET_RANGE_MULTIPLIER * cv_rmse_log))

    return {
        "predicted_ev_ebitda": predicted,
        "range_low": range_low,
        "range_high": range_high,
        "cv_rmse_log": cv_rmse_log,
        "cv_mae_median": cv_mae_median,
    }


def run(
    feature_matrix: pd.DataFrame,
    target_config: dict,
    target_llm_features: dict,
    imputation_medians: dict,
    force_retrain: bool = False,
) -> dict:
    """
    Full scoring run.

    Note: imputation_medians is required here (from feature_builder.build())
    even though it isn't part of the financial feature matrix itself — it's
    needed to fill in missing fields for the target company in predict_target().
    """
    feature_columns = [c for c in feature_matrix.columns if c != LABEL_COLUMN]
    X = feature_matrix[feature_columns]
    y = feature_matrix[LABEL_COLUMN]

    if not force_retrain and _existing_artifacts_present():
        logger.info(f"Loaded existing model from {MODEL_PATH}")
        model, cv_rmse_log, cv_mae_median, feature_importance, company_scores = _load_artifacts()
    else:
        cv_rmse_log, cv_mae_median, oof_predictions = _cross_validate(X, y)
        model = _train_final_model(X, y)
        feature_importance = _feature_importance(model, X)

        ev_ebitda_raw = np.expm1(y)
        ev_ebitda_raw.index = feature_matrix.index
        company_scores = _company_scores(ev_ebitda_raw, oof_predictions)

        _save_artifacts(model, imputation_medians, cv_rmse_log, cv_mae_median, feature_importance, company_scores)

    target_prediction = predict_target(
        model, target_config, target_llm_features, feature_columns, imputation_medians, cv_rmse_log, cv_mae_median,
    )

    return {
        "model": model,
        "feature_columns": feature_columns,
        "cv_rmse_log": cv_rmse_log,
        # Field name kept for backward compatibility; the value is now the
        # median absolute error in multiple space, not expm1(cv_rmse_log).
        "cv_rmse_multiple_space": cv_mae_median,
        "cv_mae_median": cv_mae_median,
        "feature_importance": feature_importance,
        "company_scores": company_scores,
        "target_prediction": target_prediction,
    }
