import numpy as np
import pandas as pd

from src import get_logger
from src.records import CompanyRecord

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

MAX_MISSING_FINANCIAL_FEATURES = 3
LABEL_COLUMN = "ev_ebitda_log"

# Imputing a missing financial value with the *global* median pulls in
# whatever's in the (possibly heterogeneous) adjacent bucket — e.g. a
# light-asset SaaS company's missing capex_revenue getting filled with a
# traditional manufacturer's median. Grouping by business_model keeps the
# fallback within companies that actually look like the one being imputed.
# Below this group size, the group's own median is too noisy to trust, so
# it falls back to the global median instead.
MIN_GROUP_SIZE_FOR_IMPUTATION = 5
UNKNOWN_GROUP = "unknown"
GLOBAL_GROUP_KEY = "global"


def _financial_raw_values(company: CompanyRecord) -> dict:
    return {field: company.get(field) for field in FINANCIAL_SOURCE_FIELDS}


def _drop_rows(companies: list[CompanyRecord], llm_features: dict[str, dict]) -> list[CompanyRecord]:
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


def _build_financial_dataframe(companies: list[CompanyRecord], tickers: list[str]) -> pd.DataFrame:
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


def _group_key(company: CompanyRecord, llm_features: dict[str, dict]) -> str:
    llm = llm_features.get(company["ticker"]) or {}
    return llm.get("business_model") or UNKNOWN_GROUP


def median_for(imputation_medians: dict, field: str, business_model: str | None) -> float | None:
    """Look up the imputation fallback for `field`, preferring the median
    within `business_model`'s group and falling back to the global median
    if that group wasn't large enough to get its own (see
    MIN_GROUP_SIZE_FOR_IMPUTATION)."""
    group = business_model or UNKNOWN_GROUP
    by_group = imputation_medians.get("by_group", {})
    if group in by_group and field in by_group[group]:
        return by_group[group][field]
    return imputation_medians.get(GLOBAL_GROUP_KEY, {}).get(field)


def _impute_medians(financial_df: pd.DataFrame, group_keys: list[str]) -> dict:
    global_medians = {col: financial_df[col].median() for col in FINANCIAL_FEATURE_COLUMNS}

    group_medians: dict[str, dict] = {}
    for group in sorted(set(group_keys)):
        in_group = financial_df.loc[[g == group for g in group_keys]]
        group_medians[group] = {}
        for col in FINANCIAL_FEATURE_COLUMNS:
            non_null = in_group[col].dropna()
            if len(non_null) >= MIN_GROUP_SIZE_FOR_IMPUTATION:
                group_medians[group][col] = float(non_null.median())
            else:
                group_medians[group][col] = global_medians[col]

    fill_values_by_row = pd.Series(group_keys, index=financial_df.index)
    for col in FINANCIAL_FEATURE_COLUMNS:
        n_missing = int(financial_df[col].isna().sum())
        if n_missing > 0:
            fill_values = fill_values_by_row.map(lambda g: group_medians[g][col])
            logger.info(f"Imputed {n_missing} missing values in {col} with group/global medians")
            financial_df[col] = financial_df[col].fillna(fill_values)

    return {"by_group": group_medians, GLOBAL_GROUP_KEY: global_medians}


def build(
    companies: list[CompanyRecord],
    llm_features: dict[str, dict],
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Returns:
        feature_matrix: DataFrame with the 6 financial features + ev_ebitda_log,
            indexed by ticker
        ev_ebitda_raw: Series of untransformed ev_ebitda values, indexed by ticker
        imputation_medians: {"by_group": {business_model: {col: median}},
            "global": {col: median}} — use median_for() to look up a value
            rather than indexing this directly.
    """
    kept = _drop_rows(companies, llm_features)
    tickers = [c["ticker"] for c in kept]
    group_keys = [_group_key(c, llm_features) for c in kept]

    financial_df = _build_financial_dataframe(kept, tickers)
    imputation_medians = _impute_medians(financial_df, group_keys)

    ev_ebitda_raw = pd.Series([c["ev_ebitda"] for c in kept], index=tickers, name="ev_ebitda")

    feature_matrix = financial_df.copy()
    feature_matrix[LABEL_COLUMN] = np.log1p(ev_ebitda_raw)

    return feature_matrix, ev_ebitda_raw, imputation_medians
