import numpy as np
import pandas as pd

from src import get_logger
from src.feature_builder import FINANCIAL_FEATURE_COLUMNS, LABEL_COLUMN, median_for

logger = get_logger(__name__)

TOP_N_FEATURE_BREAKDOWN = 5
DEFAULT_WEIGHT = 1.0
DEFAULT_WEIGHT_TEMPLATE = "default"

# config.yaml target_company keys that directly supply a financial feature's
# value for the target. revenue is handled separately below (it comes in as
# revenue_usd_mm and is log-transformed); every other feature an analyst
# leaves unset falls back to a peer-group median AND drops out of the distance
# (see _distance_to_target).
TARGET_FEATURE_CONFIG_KEYS = {
    "ebitda_margin": "ebitda_margin_estimate",
    "gross_margin": "gross_margin_estimate",
    "revenue_cagr_3yr": "revenue_cagr_3yr_estimate",
    "net_debt_ebitda": "net_debt_ebitda_estimate",
    "capex_revenue": "capex_revenue_estimate",
}


def _standardize(financial_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    means = financial_df.mean()
    stds = financial_df.std().replace(0, 1.0)
    return (financial_df - means) / stds, means, stds


def _feature_weights(feature_weights_config: dict, business_model: str | None) -> pd.Series:
    """
    Resolves the per-feature weight vector for distance scoring, looked up
    by the *target's* business_model so every candidate is measured against
    the same ruler (using each candidate's own business_model instead would
    make distances incomparable across candidates).

    Lookup order: the business_model's own template -> the "default"
    template -> 1.0 for any feature missing from whichever template was
    found. An empty feature_weights_config (the default) uses unweighted
    Euclidean distance.
    """
    group = business_model or DEFAULT_WEIGHT_TEMPLATE
    template = feature_weights_config.get(group) or feature_weights_config.get(DEFAULT_WEIGHT_TEMPLATE) or {}
    return pd.Series({col: template.get(col, DEFAULT_WEIGHT) for col in FINANCIAL_FEATURE_COLUMNS})


def _target_financial_row(
    target_config: dict, imputation_medians: dict, target_business_model: str | None,
) -> tuple[dict, list[str]]:
    """
    Builds the target's financial feature vector and the list of features the
    analyst actually provided ("observed"). Provided features (revenue plus any
    of the optional *_estimate fields in config.yaml's target_company) are used
    directly; features the private target cannot source fall back to
    business-model medians, then global medians — but those imputed features are
    reported as *not* observed so _distance_to_target can exclude them. The
    imputed value is still returned (it's a harmless placeholder once excluded)
    to keep the row a complete 6-feature vector for any caller that needs one.
    """
    def median(field: str) -> float | None:
        return median_for(imputation_medians, field, target_business_model)

    observed: list[str] = []

    revenue = target_config.get("revenue_usd_mm")
    if revenue is not None:
        revenue_log = np.log1p(revenue)
        observed.append("revenue_ttm_log")
    else:
        revenue_log = median("revenue_ttm_log")

    target_row = {"revenue_ttm_log": revenue_log}
    for feature, config_key in TARGET_FEATURE_CONFIG_KEYS.items():
        provided = target_config.get(config_key)
        if provided is not None:
            target_row[feature] = provided
            observed.append(feature)
        else:
            target_row[feature] = median(feature)

    return target_row, observed


def _distance_to_target(
    financial_df: pd.DataFrame, target_row: dict, weights: pd.Series, observed_features: list[str],
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Standardizes each financial feature (z-score across the comp universe),
    weights each feature's squared deviation per `weights` (1.0 everywhere
    reproduces plain Euclidean distance), and computes every company's
    distance to the target's standardized financial profile. Smaller distance
    = more financially similar to the target.

    Only features in `observed_features` contribute to the distance. Features
    the private target couldn't source (imputed to a peer median) carry no real
    information about the target — including them would just pull the ranking
    toward whichever comps happen to sit near the pool median on that axis. We
    zero their contribution rather than drop the columns so the returned
    per-feature breakdown keeps a stable 6-feature shape (an imputed feature
    simply shows 0 contribution — it drove nothing).

    Returns (distance, per_feature_weighted_sq_diff); the latter lets
    callers attribute a company's distance to individual features (already
    weighted, so it reflects what actually drove the ranking) for the
    post-hoc diagnostic breakdown of the selected Top-N.
    """
    standardized, means, stds = _standardize(financial_df)
    target_standardized = pd.Series(
        {col: (target_row.get(col, means[col]) - means[col]) / stds[col] for col in financial_df.columns},
    )
    sq_diff = standardized.subtract(target_standardized, axis=1) ** 2
    weighted_sq_diff = sq_diff.multiply(weights, axis=1)
    unobserved = [col for col in financial_df.columns if col not in observed_features]
    if unobserved:
        weighted_sq_diff[unobserved] = 0.0
    distance = np.sqrt(weighted_sq_diff.sum(axis=1))
    return distance, weighted_sq_diff


def _company_scores(ev_ebitda_raw: pd.Series, distance: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "ev_ebitda_actual": ev_ebitda_raw,
        "residual_abs": distance,
    }, index=ev_ebitda_raw.index)


def feature_distance_breakdown(sq_diff: pd.DataFrame, tickers: list[str], top_n: int = TOP_N_FEATURE_BREAKDOWN) -> pd.DataFrame:
    """
    Post-hoc diagnostic: averages each financial feature's (weighted)
    squared distance to the target across a set of companies (typically the
    selected Top-N), showing which features drive how close/far those comps
    are from the target's financial profile — reflects scorer.feature_weights
    if configured, so this matches what actually drove the ranking rather
    than an unweighted view of it.
    """
    mean_sq = sq_diff.loc[tickers].mean(axis=0)
    return pd.DataFrame({
        "feature": mean_sq.index,
        "mean_sq_distance": mean_sq.to_numpy(),
    }).sort_values("mean_sq_distance", ascending=False).reset_index(drop=True).head(top_n)


def run(
    feature_matrix: pd.DataFrame,
    target_config: dict,
    target_llm_features: dict,
    imputation_medians: dict,
    feature_weights_config: dict | None = None,
) -> dict:
    """
    Ranks comps by financial-feature distance to the target. This is directly
    interpretable per company and pairs with the business-attribute hard/soft
    filters in reporter._select_top_15.

    feature_weights_config (config.yaml's scorer.feature_weights) lets
    different industries weight the 6 financial features differently — e.g.
    growth matters more for SaaS, EBITDA margin matters more for asset-heavy
    manufacturing — instead of treating them as equally informative for
    every target. Omit it (or leave it empty) to use unweighted Euclidean
    distance.
    """
    target_business_model = target_llm_features.get("business_model")
    financial_df = feature_matrix[list(FINANCIAL_FEATURE_COLUMNS)]
    ev_ebitda_raw = np.expm1(feature_matrix[LABEL_COLUMN])

    target_row, observed_features = _target_financial_row(target_config, imputation_medians, target_business_model)
    if not observed_features:
        # No analyst-provided features at all (target gives neither revenue nor
        # any margin/growth/leverage estimate). Excluding everything would make
        # every distance zero, so fall back to the old behaviour — all six
        # features, imputed — and flag it, rather than emit a degenerate ranking.
        logger.warning(
            "Target has no analyst-provided financial features; falling back to "
            "all six (fully imputed) features for distance scoring."
        )
        observed_features = list(FINANCIAL_FEATURE_COLUMNS)

    weights = _feature_weights(feature_weights_config or {}, target_business_model)
    distance, sq_diff = _distance_to_target(financial_df, target_row, weights, observed_features)

    company_scores = _company_scores(ev_ebitda_raw, distance)

    return {
        "company_scores": company_scores,
        "feature_distance_sq_diff": sq_diff,
        "observed_target_features": observed_features,
    }
