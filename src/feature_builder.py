import numpy as np
import pandas as pd

from src import get_logger

logger = get_logger(__name__)

# Pre-transform source fields for the 6 financial features (revenue is
# log-transformed into revenue_ttm_log; the rest are used as-is).
FINANCIAL_SOURCE_FIELDS = (
    "revenue_ttm_usd_mm", "ebitda_margin", "gross_margin",
    "revenue_cagr_3yr", "net_debt_ebitda", "capex_revenue",
)

FINANCIAL_FEATURE_COLUMNS = (
    "revenue_ttm_log", "ebitda_margin", "gross_margin",
    "revenue_cagr_3yr", "net_debt_ebitda", "capex_revenue",
)

LLM_CATEGORICAL_FIELDS = (
    "business_model", "revenue_recurrence", "customer_type",
    "capital_intensity", "primary_value_driver",
)

MAX_MISSING_FINANCIAL_FEATURES = 3
LABEL_COLUMN = "ev_ebitda_log"


def _financial_raw_values(company: dict) -> dict:
    return {field: company.get(field) for field in FINANCIAL_SOURCE_FIELDS}


def _drop_rows(companies: list[dict], llm_features: dict[str, dict]) -> list[dict]:
    total = len(companies)

    with_label = [c for c in companies if c.get("ev_ebitda") is not None]
    logger.info(f"Dropped {total - len(with_label)} companies with no ev_ebitda label")

    after_financials = []
    for c in with_label:
        missing = sum(1 for v in _financial_raw_values(c).values() if v is None)
        if missing <= MAX_MISSING_FINANCIAL_FEATURES:
            after_financials.append(c)
    logger.info(
        f"Dropped {len(with_label) - len(after_financials)} companies with more than "
        f"{MAX_MISSING_FINANCIAL_FEATURES} missing financial features"
    )

    after_llm = []
    for c in after_financials:
        llm = llm_features.get(c["ticker"])
        if llm is not None and not llm.get("extraction_failed", True):
            after_llm.append(c)
    logger.info(f"Dropped {len(after_financials) - len(after_llm)} companies with failed/missing LLM extraction")

    logger.info(f"{len(after_llm)} companies remain out of {total} after all drop steps")
    return after_llm


def _build_financial_dataframe(companies: list[dict], tickers: list[str]) -> pd.DataFrame:
    rows = []
    for c in companies:
        raw = _financial_raw_values(c)
        revenue = raw["revenue_ttm_usd_mm"]
        rows.append({
            "revenue_ttm_log": np.log1p(revenue) if revenue is not None else None,
            "ebitda_margin": raw["ebitda_margin"],
            "gross_margin": raw["gross_margin"],
            "revenue_cagr_3yr": raw["revenue_cagr_3yr"],
            "net_debt_ebitda": raw["net_debt_ebitda"],
            "capex_revenue": raw["capex_revenue"],
        })
    df = pd.DataFrame(rows, index=tickers)
    for col in FINANCIAL_FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _impute_medians(financial_df: pd.DataFrame) -> dict:
    imputation_medians = {}
    for col in FINANCIAL_FEATURE_COLUMNS:
        median = financial_df[col].median()
        imputation_medians[col] = median
        n_missing = int(financial_df[col].isna().sum())
        if n_missing > 0:
            logger.info(f"Imputed {n_missing} missing values in {col} with median {median}")
        financial_df[col] = financial_df[col].fillna(median)
    return imputation_medians


def _build_llm_dataframe(companies: list[dict], llm_features: dict[str, dict], tickers: list[str]) -> pd.DataFrame:
    rows = []
    for c in companies:
        llm = llm_features[c["ticker"]]
        row = {}
        for field in LLM_CATEGORICAL_FIELDS:
            value = llm.get(field)
            row[field] = value if value is not None else "unknown"
        rows.append(row)
    df = pd.DataFrame(rows, index=tickers)
    return pd.get_dummies(df, columns=list(LLM_CATEGORICAL_FIELDS), drop_first=True)


def build(
    companies: list[dict],
    llm_features: dict[str, dict],
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Returns:
        feature_matrix: DataFrame with features + ev_ebitda_log, indexed by ticker
        ev_ebitda_raw: Series of untransformed ev_ebitda values, indexed by ticker
        imputation_medians: dict mapping column_name -> median_value
    """
    kept = _drop_rows(companies, llm_features)
    tickers = [c["ticker"] for c in kept]

    financial_df = _build_financial_dataframe(kept, tickers)
    imputation_medians = _impute_medians(financial_df)

    llm_encoded = _build_llm_dataframe(kept, llm_features, tickers)

    ev_ebitda_raw = pd.Series([c["ev_ebitda"] for c in kept], index=tickers, name="ev_ebitda")

    feature_matrix = pd.concat([financial_df, llm_encoded], axis=1)
    feature_matrix[LABEL_COLUMN] = np.log1p(ev_ebitda_raw)

    return feature_matrix, ev_ebitda_raw, imputation_medians
