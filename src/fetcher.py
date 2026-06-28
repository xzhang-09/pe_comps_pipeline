import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import edgar
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src import fmp_client, get_logger
from src.config_schema import PipelineConfig, as_config

logger = get_logger(__name__)

CACHE_DIR = Path("data/cache")
OUTPUTS_DIR = Path("outputs")
FAILED_TICKERS_CSV = OUTPUTS_DIR / "failed_tickers.csv"

DEFAULT_SEC_IDENTITY = "PE-Comps-Pipeline research@example.com"
EDGAR_IDENTITY = os.environ.get("SEC_IDENTITY", DEFAULT_SEC_IDENTITY)
edgar.set_identity(EDGAR_IDENTITY)

BUSINESS_DESCRIPTION_MAX_CHARS = 1500

# Revenue, EBITDA, gross profit, capex, and net_debt_ebitda all come from
# SEC XBRL via edgartools (see _fetch_single). FMP is only ever called for
# get_profile() (market cap, sector, description fallback) — by design, see
# README's "Using a paid FMP plan" note. EV/EBITDA/EV-Revenue are then
# derived from that market cap plus the EDGAR-sourced figures above (see
# _enrich_with_fmp_data) rather than pulled directly from FMP's key-metrics/
# enterprise-values endpoints, which return 402 on FMP's free tier for
# anything beyond a handful of demo mega-caps.
FMP_REQUEST_DELAY_SECONDS = 1

# Columns in edgartools' Statement.to_dataframe() that aren't fiscal-period values.
STATEMENT_METADATA_COLUMNS = {
    "concept", "label", "standard_concept", "level", "abstract", "dimension",
    "is_breakdown", "dimension_axis", "dimension_member", "dimension_member_label",
    "dimension_label", "balance", "weight", "preferred_sign", "parent_concept",
    "parent_abstract_concept",
}

EMPTY_FIELDS = (
    "company_name", "market_cap_usd_mm", "revenue_ttm_usd_mm", "ebitda_margin",
    "gross_margin", "revenue_cagr_3yr", "net_debt_ebitda", "capex_revenue",
    "ev_ebitda", "ev_revenue", "gics_sector", "business_description",
    "description_source", "enterprise_value_usd_mm", "ebitda_usd_mm",
    "valuation_source", "ev_ebitda_source_value", "ev_revenue_source_value",
    "net_debt_usd_mm", "minority_interest_usd_mm", "preferred_equity_usd_mm",
    "operating_lease_liability_usd_mm", "ebit_usd_mm", "net_income_usd_mm",
    "gross_profit_usd_mm", "free_cash_flow_usd_mm", "fcf_conversion",
    "interest_coverage", "debt_to_equity", "ev_ebit", "ev_gross_profit",
    "pe_ratio", "fcf_yield",
)

# Enterprise-value bridge items beyond debt/cash. Each is looked up by trying
# a list of candidate XBRL standard_concept names (edgartools' normalization
# isn't perfectly stable across filers, and noncontrolling interest /
# preferred / operating-lease lines are tagged a few different ways), taking
# the first that matches. Absent entirely -> treated as 0 in the bridge, which
# is correct for the many companies that simply have no minority interest or
# preferred stock.
MINORITY_INTEREST_CONCEPTS = (
    "MinorityInterest",             # edgartools balance-sheet standard_concept (varies by company)
    "NoncontrollingInterest",
)
PREFERRED_EQUITY_CONCEPTS = (
    "PreferredStock",               # edgartools standard_concept (confirmed from HLLY)
    "TemporaryAndMezzanineFinancing",
)
OPERATING_LEASE_LIABILITY_CONCEPTS = ("OperatingLeaseDebtEquivalent",)
OPERATING_LEASE_CURRENT_CONCEPTS = ("OperatingLeaseCurrentDebtEquivalent",)    # confirmed
OPERATING_LEASE_NONCURRENT_CONCEPTS = ("OperatingLeaseNonCurrentDebtEquivalent",)  # confirmed

# Income-statement / cash-flow / balance-sheet concepts for the additional
# PE metrics (EV/EBIT, EV/Gross Profit, P/E, FCF conversion, FCF yield,
# interest coverage, debt/equity). Tried as candidate lists for the same
# reason as the EV-bridge concepts above. EBIT reuses OperatingIncomeLoss
# (the same line _ebitda already reads), so it has no separate list.
NET_INCOME_CONCEPTS = (
    "NetIncome",        # edgartools standard_concept (confirmed from PPIH)
    "ProfitLoss",       # fallback: consolidated net income incl. minorities
)
INTEREST_EXPENSE_CONCEPTS = (
    "InterestExpense",          # edgartools standard_concept (confirmed)
)
OPERATING_CASH_FLOW_CONCEPTS = (
    "NetCashFromOperatingActivities",   # edgartools standard_concept (confirmed)
)
TOTAL_EQUITY_CONCEPTS = (
    "AllEquityBalance",         # edgartools standard_concept for StockholdersEquity (confirmed)
)


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.json"


def _load_cache(ticker: str) -> dict | None:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_cache(ticker: str, record: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(ticker), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def _candidate_ticker(candidate: str | dict) -> str:
    return candidate["ticker"] if isinstance(candidate, dict) else candidate


def _candidate_metadata(candidate: str | dict) -> dict:
    if not isinstance(candidate, dict):
        return {}
    return {
        key: candidate.get(key)
        for key in (
            "matched_sic_codes", "primary_matched_sic_codes", "adjacent_matched_sic_codes",
            "source_bucket", "sic_2_digit", "sic_3_digit", "industry_cluster", "candidate_source",
        )
        if candidate.get(key) is not None
    }


def _attach_universe_metadata(record: dict, candidate: str | dict) -> dict:
    metadata = _candidate_metadata(candidate)
    if not metadata:
        record.setdefault("universe_metadata", {})
        return record

    existing = record.get("universe_metadata") or {}
    merged = {**existing, **metadata}
    record["universe_metadata"] = merged
    for key in ("source_bucket", "sic_2_digit", "sic_3_digit", "industry_cluster"):
        if merged.get(key) is not None:
            record[key] = merged[key]
    return record


def _reset_failed_tickers_csv() -> None:
    """
    Start each fetch_batch() run with a clean file so reporter.py's
    failed_fetch_count reflects the current run.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FAILED_TICKERS_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["ticker", "error_type", "error_message", "timestamp"])


def _record_failed_ticker(ticker: str, error_type: str, error_message: str) -> None:
    with open(FAILED_TICKERS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            ticker, error_type, error_message,
            datetime.now(timezone.utc).isoformat(),
        ])


def _period_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in STATEMENT_METADATA_COLUMNS]


def _concept_series(df: pd.DataFrame, standard_concept: str) -> list[float]:
    """Values for a standardized XBRL concept across fiscal periods, most
    recent first. Filtering by standard_concept (vs. label text) is what
    lets this work the same way across companies that word their statements
    differently — e.g. Apple's "Gross margin" line and another filer's
    "Gross profit" line both normalize to the GrossProfit concept."""
    if "standard_concept" not in df.columns:
        return []

    matches = df[df["standard_concept"] == standard_concept]
    if matches.empty:
        return []

    # When multiple rows share the same standard_concept (edgartools labels
    # sub-items with their parent's standard_concept), prefer the aggregate
    # total row: non-breakdown, non-abstract, at the highest level in the
    # statement hierarchy (lowest level number = outermost indentation).
    candidates = matches
    for col, false_val in (("is_breakdown", True), ("abstract", True), ("dimension", True)):
        if col in candidates.columns:
            filtered = candidates[candidates[col] != false_val]
            if not filtered.empty:
                candidates = filtered
    if "level" in candidates.columns:
        min_level = candidates["level"].min()
        candidates = candidates[candidates["level"] == min_level]
    row = candidates.iloc[0]
    values = []
    for col in _period_columns(df):
        val = row.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        try:
            values.append(float(val))
        except (TypeError, ValueError):
            continue
    return values


def _concept_value(df: pd.DataFrame, standard_concept: str) -> float | None:
    values = _concept_series(df, standard_concept)
    return values[0] if values else None


def _concept_value_any(df: pd.DataFrame, concepts: tuple[str, ...]) -> float | None:
    """First non-null value among a list of candidate standard_concept names —
    used where edgartools may tag the same economic line under one of several
    concept names across filers (see the EV-bridge concept lists above)."""
    for concept in concepts:
        value = _concept_value(df, concept)
        if value is not None:
            return value
    return None


def _operating_lease_liability(balance_sheet_df: pd.DataFrame) -> float | None:
    """Total operating-lease liability (ASC 842). Prefers a single reported
    total; otherwise sums the current and noncurrent lines. Finance leases are
    deliberately excluded — those are typically already inside the debt lines
    picked up by _net_debt. Returns None only when no operating-lease line is
    tagged at all (vs. 0, which would wrongly assert the company has none)."""
    total = _concept_value_any(balance_sheet_df, OPERATING_LEASE_LIABILITY_CONCEPTS)
    if total is not None:
        return total

    current = _concept_value_any(balance_sheet_df, OPERATING_LEASE_CURRENT_CONCEPTS)
    noncurrent = _concept_value_any(balance_sheet_df, OPERATING_LEASE_NONCURRENT_CONCEPTS)
    if current is None and noncurrent is None:
        return None
    return (current or 0.0) + (noncurrent or 0.0)


def _revenue_and_cagr(ticker: str, income_df: pd.DataFrame) -> tuple[float | None, float | None]:
    revenue_values = _concept_series(income_df, "Revenue")
    if not revenue_values:
        logger.warning(f"{ticker} — Revenue not available from EDGAR XBRL, setting None")
        return None, None

    revenue_ttm = revenue_values[0]

    # A single 10-K's XBRL income statement typically only carries 3 fiscal
    # years of comparatives, so the longest CAGR span available here is
    # 2 years (not a true 3-year-ago figure, which would require a second,
    # older filing). We use whatever span the filing actually gives us.
    if len(revenue_values) < 2:
        return revenue_ttm, None

    n_years = len(revenue_values) - 1
    oldest = revenue_values[-1]
    if not oldest or oldest <= 0:
        return revenue_ttm, None

    try:
        cagr = (revenue_ttm / oldest) ** (1 / n_years) - 1
    except Exception:
        cagr = None

    return revenue_ttm, cagr


def _gross_profit(ticker: str, income_df: pd.DataFrame, revenue: float | None) -> float | None:
    gross_profit = _concept_value(income_df, "GrossProfit")
    if gross_profit is not None:
        return gross_profit

    # Many industrials don't tag a GrossProfit subtotal at all; derive it
    # from Revenue - Cost of Goods/Services when that's the case.
    cogs = _concept_value(income_df, "CostOfGoodsAndServicesSold")
    if cogs is None:
        cogs = _concept_value(income_df, "CostOfRevenue")

    if revenue is not None and cogs is not None:
        return revenue - cogs

    logger.warning(f"{ticker} — gross profit not available from EDGAR XBRL, setting None")
    return None


def _ebitda(ticker: str, income_df: pd.DataFrame, cashflow_df: pd.DataFrame) -> float | None:
    operating_income = _concept_value(income_df, "OperatingIncomeLoss")
    depreciation_amortization = _concept_value(cashflow_df, "DepreciationExpense")

    if operating_income is None or depreciation_amortization is None:
        logger.warning(f"{ticker} — EBITDA not available (operating income or D&A missing), setting None")
        return None

    return operating_income + depreciation_amortization


def _total_debt(balance_sheet_df: pd.DataFrame) -> float | None:
    """Interest-bearing debt = short-term + current-portion-LTD + long-term.
    Returns None only when no debt line is tagged at all (vs. 0)."""
    short_term_debt = _concept_value(balance_sheet_df, "ShortTermDebt")
    current_portion_ltd = _concept_value(balance_sheet_df, "CurrentPortionOfLongTermDebt")
    long_term_debt = _concept_value(balance_sheet_df, "LongTermDebt")

    if short_term_debt is None and current_portion_ltd is None and long_term_debt is None:
        return None
    return (short_term_debt or 0.0) + (current_portion_ltd or 0.0) + (long_term_debt or 0.0)


def _net_debt(ticker: str, balance_sheet_df: pd.DataFrame) -> float | None:
    """Net debt = interest-bearing debt - cash & marketable securities, all
    from the same EDGAR XBRL balance sheet used for everything else. Cross-
    checked against FMP's own totalDebt/cashAndCashEquivalents for AAPL."""
    cash = _concept_value(balance_sheet_df, "CashAndMarketableSecurities")
    total_debt = _total_debt(balance_sheet_df)

    if cash is None and total_debt is None:
        logger.warning(f"{ticker} — debt/cash not available from EDGAR XBRL, setting net_debt None")
        return None

    return (total_debt or 0.0) - (cash or 0.0)


def _ebit(income_df: pd.DataFrame) -> float | None:
    """EBIT = operating income — the same line _ebitda adds D&A back to."""
    return _concept_value(income_df, "OperatingIncomeLoss")


def _free_cash_flow(cashflow_df: pd.DataFrame, capex: float | None) -> float | None:
    """Levered free cash flow = cash from operations - capex, both straight
    from the cash flow statement. Deliberately the simple, robust definition
    (it avoids reconstructing FCF from EBITDA via fragile working-capital and
    cash-tax estimates) and is after interest and tax, which is the cash a PE
    buyer actually has to service debt — the angle EBITDA alone misses."""
    operating_cash_flow = _concept_value_any(cashflow_df, OPERATING_CASH_FLOW_CONCEPTS)
    if operating_cash_flow is None or capex is None:
        return None
    return operating_cash_flow - capex


def _capital_expenditures(ticker: str, financials) -> float | None:
    try:
        capex = financials.get_capital_expenditures()
        return abs(capex) if capex is not None else None
    except Exception as e:
        logger.warning(f"{ticker} — capex not available: {e}")
        return None


def _fetch_edgar_description(ticker: str) -> str | None:
    """Item 1 Business text from the company's latest 10-K, via edgartools'
    structured filing parser (TenK.business) rather than raw text scraping."""
    company = edgar.Company(ticker)
    filing = company.get_filings(form="10-K").latest()
    if filing is None:
        return None

    doc = filing.obj()
    text = getattr(doc, "business", None)
    if not text:
        return None

    return text[:BUSINESS_DESCRIPTION_MAX_CHARS]


def _fetch_business_description(ticker: str) -> tuple[str | None, str | None]:
    try:
        description = _fetch_edgar_description(ticker)
        if description:
            return description, "edgar"
    except Exception as e:
        logger.warning(f"{ticker} — EDGAR description failed: {e}")

    return None, None


def _log_retry(retry_state):
    ticker = retry_state.args[0] if retry_state.args else "?"
    error = retry_state.outcome.exception()
    logger.warning(f"{ticker} — attempt {retry_state.attempt_number} failed: {error}. Retrying...")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(Exception),
    before_sleep=_log_retry,
    reraise=True,
)
def _fetch_single(ticker: str) -> dict:
    """
    Fetch fundamentals from SEC EDGAR XBRL (10-K) — revenue, EBITDA, gross
    profit, capex, and now net_debt_ebitda too, since debt/cash are balance
    sheet items available the same way. Market cap, sector, and the
    description fallback are deliberately NOT fetched here — they come from
    FMP separately in _enrich_with_fmp_data, which never raises, so an FMP
    outage can't wipe out good EDGAR-sourced data via this function's
    retry/failure path.
    """
    company = edgar.Company(ticker)
    financials = company.get_financials()
    if financials is None:
        raise ValueError(f"No annual financials (10-K/20-F/40-F) available for {ticker}")

    income_df = financials.income_statement().to_dataframe()
    cashflow_df = financials.cashflow_statement().to_dataframe()
    balance_sheet_df = financials.balance_sheet().to_dataframe()

    revenue, revenue_cagr_3yr = _revenue_and_cagr(ticker, income_df)
    revenue_ttm_usd_mm = revenue / 1e6 if revenue is not None else None

    ebitda = _ebitda(ticker, income_df, cashflow_df)
    ebitda_margin = ebitda / revenue if (ebitda is not None and revenue) else None

    gross_profit = _gross_profit(ticker, income_df, revenue)
    gross_margin = gross_profit / revenue if (gross_profit is not None and revenue) else None

    capex = _capital_expenditures(ticker, financials)
    capex_revenue = capex / revenue if (capex is not None and revenue) else None

    net_debt = _net_debt(ticker, balance_sheet_df)
    net_debt_ebitda = net_debt / ebitda if (net_debt is not None and ebitda) else None
    ebitda_usd_mm = ebitda / 1e6 if ebitda is not None else None

    # EV-bridge items beyond net debt. Minority interest and preferred equity
    # are standard enterprise-value additions (they're claims on the
    # consolidated business that sit above common equity) and are NOT part of
    # net_debt / the leverage feature — they only enter the EV built in
    # _enrich_with_fmp_data. Operating-lease liability is captured here too but
    # only added to EV when config opts in (see _enrich_with_fmp_data).
    minority_interest = _concept_value_any(balance_sheet_df, MINORITY_INTEREST_CONCEPTS)
    preferred_equity = _concept_value_any(balance_sheet_df, PREFERRED_EQUITY_CONCEPTS)
    operating_lease_liability = _operating_lease_liability(balance_sheet_df)

    # Additional PE metrics. EV-denominated multiples (EV/EBIT, EV/Gross Profit,
    # P/E, FCF yield) need market cap, so they're finished in
    # _enrich_with_fmp_data; the operating/leverage ratios below are computed
    # here since they don't depend on FMP at all (same as net_debt_ebitda).
    ebit = _ebit(income_df)
    net_income = _concept_value_any(income_df, NET_INCOME_CONCEPTS)
    interest_expense = _concept_value_any(income_df, INTEREST_EXPENSE_CONCEPTS)
    total_equity = _concept_value_any(balance_sheet_df, TOTAL_EQUITY_CONCEPTS)
    total_debt = _total_debt(balance_sheet_df)
    free_cash_flow = _free_cash_flow(cashflow_df, capex)

    fcf_conversion = free_cash_flow / ebitda if (free_cash_flow is not None and ebitda) else None
    # Interest coverage uses the *magnitude* of interest expense: edgartools'
    # InterestExpense concept comes through with an inconsistent sign across
    # filers (some tag it as a positive cost, others as a negative/net figure),
    # so dividing by the raw value produced negative "coverage" for healthy,
    # profitable comps — nonsensical as a leverage metric. EBIT keeps its sign:
    # a genuinely loss-making company (negative EBIT) still yields negative
    # coverage, which correctly flags that it can't cover interest.
    interest_coverage = ebit / abs(interest_expense) if (ebit is not None and interest_expense) else None
    debt_to_equity = total_debt / total_equity if (total_debt is not None and total_equity) else None

    business_description, description_source = _fetch_business_description(ticker)

    return {
        "ticker": ticker,
        "company_name": company.name,
        "market_cap_usd_mm": None,
        "revenue_ttm_usd_mm": revenue_ttm_usd_mm,
        "ebitda_margin": ebitda_margin,
        "gross_margin": gross_margin,
        "revenue_cagr_3yr": revenue_cagr_3yr,
        "net_debt_ebitda": net_debt_ebitda,
        "capex_revenue": capex_revenue,
        "net_debt_usd_mm": net_debt / 1e6 if net_debt is not None else None,
        "minority_interest_usd_mm": minority_interest / 1e6 if minority_interest is not None else None,
        "preferred_equity_usd_mm": preferred_equity / 1e6 if preferred_equity is not None else None,
        "operating_lease_liability_usd_mm": (
            operating_lease_liability / 1e6 if operating_lease_liability is not None else None
        ),
        "enterprise_value_usd_mm": None,
        "ebitda_usd_mm": ebitda_usd_mm,
        "ebit_usd_mm": ebit / 1e6 if ebit is not None else None,
        "net_income_usd_mm": net_income / 1e6 if net_income is not None else None,
        "gross_profit_usd_mm": gross_profit / 1e6 if gross_profit is not None else None,
        "free_cash_flow_usd_mm": free_cash_flow / 1e6 if free_cash_flow is not None else None,
        "fcf_conversion": fcf_conversion,
        "interest_coverage": interest_coverage,
        "debt_to_equity": debt_to_equity,
        "ev_ebitda": None,
        "ev_revenue": None,
        "ev_ebit": None,
        "ev_gross_profit": None,
        "pe_ratio": None,
        "fcf_yield": None,
        "valuation_source": None,
        "ev_ebitda_source_value": None,
        "ev_revenue_source_value": None,
        "gics_sector": None,
        "business_description": business_description,
        "description_source": description_source,
        "universe_metadata": {},
        "raw_fmp_profile": None,
        "missing_flags": {},
        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _include_operating_leases(config: PipelineConfig) -> bool:
    """Whether to capitalize operating leases into EV. Defaults False: under
    ASC 842 operating-lease cost still runs through operating income, so EBITDA
    is net of rent — adding the lease liability to EV without also adding rent
    back (i.e. moving to EV/EBITDAR) would make the multiple internally
    inconsistent. The toggle exists for lease-heavy targets where the analyst
    wants the lease-capitalized view and accepts that caveat."""
    return config.valuation.include_operating_leases_in_ev


def _ev_from_bridge(record: dict, market_cap_usd_mm: float, include_leases: bool) -> float:
    """Enterprise value = equity value (market cap) + net debt + minority
    interest + preferred equity (+ operating leases if opted in). Minority
    interest and preferred are senior claims on the consolidated business, so
    omitting them — as the earlier market_cap + net_debt bridge did —
    understates EV (and thus EV/EBITDA) for any company that has them, which
    biases the comp set against, and the implied valuation away from, those
    capital structures. Absent components are treated as 0."""
    net_debt_usd_mm = record.get("net_debt_usd_mm") or 0.0
    minority = record.get("minority_interest_usd_mm") or 0.0
    preferred = record.get("preferred_equity_usd_mm") or 0.0
    leases = (record.get("operating_lease_liability_usd_mm") or 0.0) if include_leases else 0.0
    return market_cap_usd_mm + net_debt_usd_mm + minority + preferred + leases


def _enrich_with_fmp_data(record: dict, ticker: str, config: PipelineConfig, sleep_before: bool) -> None:
    """
    Best-effort FMP lookup via get_profile() only — market cap, sector, and
    a fallback for business_description when EDGAR's didn't come through.
    Never raises and never causes the company to be dropped — on any
    failure the record just keeps its EDGAR-sourced fields with
    market_cap/EV left as None. ev_ebitda/ev_revenue are then derived below
    from this market cap plus the EDGAR-sourced EV bridge already on the
    record (net debt, minority interest, preferred equity, and optionally
    operating leases — see _ev_from_bridge), rather than pulled directly from
    FMP's key-metrics/enterprise-values endpoints (paid-tier only on FMP —
    see README's "Using a paid FMP plan" note if you have access to them).
    """
    if sleep_before:
        time.sleep(FMP_REQUEST_DELAY_SECONDS)

    try:
        profile = fmp_client.get_profile(ticker)
    except Exception as e:
        logger.warning(f"{ticker} — FMP market cap/sector lookup failed: {e}. Leaving market data as None.")
        return

    if not profile:
        logger.warning(f"{ticker} — FMP returned no profile data. Leaving market data as None.")
        return
    record["raw_fmp_profile"] = profile

    if not record.get("business_description"):
        description = profile.get("description")
        if description:
            record["business_description"] = description[:BUSINESS_DESCRIPTION_MAX_CHARS]
            record["description_source"] = "fmp"

    market_cap = profile.get("marketCap")
    if market_cap is not None and record.get("market_cap_usd_mm") is None:
        record["market_cap_usd_mm"] = market_cap / 1e6
    record["gics_sector"] = profile.get("sector")

    if market_cap is None:
        return

    revenue = record.get("revenue_ttm_usd_mm")
    ebitda_usd_mm = record.get("ebitda_usd_mm")

    # Net debt is required for the bridge; without it we can't build a
    # defensible EV, so leave market_cap set but EV/multiples None.
    if record.get("net_debt_usd_mm") is None:
        return

    market_cap_usd_mm = market_cap / 1e6
    include_leases = _include_operating_leases(config)
    ev_usd_mm = _ev_from_bridge(record, market_cap_usd_mm, include_leases)

    if ebitda_usd_mm:
        record["ev_ebitda"] = ev_usd_mm / ebitda_usd_mm
        record["ev_ebitda_source_value"] = record["ev_ebitda"]
        record["enterprise_value_usd_mm"] = ev_usd_mm
        record["valuation_source"] = (
            "sec_xbrl_derived_lease_adj" if include_leases else "sec_xbrl_derived"
        )
    if revenue:
        record["ev_revenue"] = ev_usd_mm / revenue
        record["ev_revenue_source_value"] = record["ev_revenue"]

    # EV-denominated and equity multiples that need market cap / EV. Each is
    # left None unless its denominator is present and positive (a negative EBIT
    # or net income makes the multiple meaningless, not informative — outlier
    # filtering in _validate also drops absurd positives).
    ebit_usd_mm = record.get("ebit_usd_mm")
    if ebit_usd_mm and ebit_usd_mm > 0:
        record["ev_ebit"] = ev_usd_mm / ebit_usd_mm
    gross_profit_usd_mm = record.get("gross_profit_usd_mm")
    if gross_profit_usd_mm and gross_profit_usd_mm > 0:
        record["ev_gross_profit"] = ev_usd_mm / gross_profit_usd_mm
    net_income_usd_mm = record.get("net_income_usd_mm")
    if net_income_usd_mm and net_income_usd_mm > 0:
        record["pe_ratio"] = market_cap_usd_mm / net_income_usd_mm
    free_cash_flow_usd_mm = record.get("free_cash_flow_usd_mm")
    if free_cash_flow_usd_mm is not None and ev_usd_mm:
        # FCF yield can legitimately be negative (cash-burning comp); keep it.
        record["fcf_yield"] = free_cash_flow_usd_mm / ev_usd_mm


def _validate(record: dict, ticker: str) -> dict:
    ev_ebitda = record.get("ev_ebitda")
    if ev_ebitda is not None and (ev_ebitda > 100 or ev_ebitda < 0):
        logger.warning(f"{ticker} — ev_ebitda outlier ({ev_ebitda}), setting None")
        record["ev_ebitda"] = None

    ebitda_margin = record.get("ebitda_margin")
    if ebitda_margin is not None and (ebitda_margin > 0.80 or ebitda_margin < -0.50):
        logger.warning(f"{ticker} — ebitda_margin outlier ({ebitda_margin}), setting None")
        record["ebitda_margin"] = None

    revenue_ttm_usd_mm = record.get("revenue_ttm_usd_mm")
    if revenue_ttm_usd_mm is not None and revenue_ttm_usd_mm <= 0:
        logger.warning(f"{ticker} — revenue_ttm_usd_mm non-positive ({revenue_ttm_usd_mm}), setting None")
        record["revenue_ttm_usd_mm"] = None

    gross_margin = record.get("gross_margin")
    if gross_margin is not None and (gross_margin > 1.0 or gross_margin < 0):
        logger.warning(f"{ticker} — gross_margin outlier ({gross_margin}), setting None")
        record["gross_margin"] = None

    # Sanity bounds on the added multiples — broad bands to catch a
    # unit/field-mapping bug, not to enforce an industry-normal range. EBIT and
    # net income are smaller denominators than EBITDA, so their multiples run
    # higher before they're implausible.
    for field, hi in (("ev_ebit", 150.0), ("ev_gross_profit", 100.0), ("pe_ratio", 200.0)):
        value = record.get(field)
        if value is not None and (value > hi or value < 0):
            logger.warning(f"{ticker} — {field} outlier ({value}), setting None")
            record[field] = None

    # Interest coverage above ~100x almost always means the interest-expense
    # line was tagged with only a partial/immaterial figure (e.g. a company
    # carrying real debt showing $0.1mm interest), not genuinely negligible
    # leverage — drop it rather than let a tiny denominator distort the
    # leverage benchmark. Negative values (loss-making, EBIT < 0) are kept:
    # they're a real signal, not a data artifact.
    interest_coverage = record.get("interest_coverage")
    if interest_coverage is not None and interest_coverage > 100.0:
        logger.warning(f"{ticker} — interest_coverage outlier ({interest_coverage}), setting None")
        record["interest_coverage"] = None

    record["missing_flags"] = {
        field: record.get(field) is None
        for field in (
            "market_cap_usd_mm", "revenue_ttm_usd_mm", "ebitda_margin", "gross_margin",
            "revenue_cagr_3yr", "net_debt_ebitda", "capex_revenue", "ev_ebitda", "ev_revenue",
            "business_description",
        )
    }
    return record


def _empty_record(ticker: str) -> dict:
    record = {field: None for field in EMPTY_FIELDS}
    record["ticker"] = ticker
    record["universe_metadata"] = {}
    record["raw_fmp_profile"] = None
    record["missing_flags"] = {field: True for field in EMPTY_FIELDS}
    record["fetch_timestamp"] = datetime.now(timezone.utc).isoformat()
    return record


def fetch_batch(tickers: list[str | dict], config: PipelineConfig | dict) -> list[dict]:
    """
    Fetch financial data for all tickers.
    Returns list of dicts (one per ticker, including failed ones with None values).
    Uses cache — skips API call if cache/{ticker}.json exists.
    Writes failures to outputs/failed_tickers.csv.
    """
    cfg = as_config(config)
    results = []
    total = len(tickers)
    fmp_calls_made = 0

    _reset_failed_tickers_csv()

    for i, candidate in enumerate(tickers, start=1):
        ticker = _candidate_ticker(candidate)
        logger.info(f"Processing {i}/{total}: {ticker}")

        cached = _load_cache(ticker)
        if cached is not None:
            logger.info(f"{ticker} — loaded from cache")
            results.append(_attach_universe_metadata(cached, candidate))
            continue

        try:
            record = _fetch_single(ticker)
        except Exception as e:
            logger.error(f"{ticker} — failed after 3 retries: {e}")
            _record_failed_ticker(ticker, type(e).__name__, str(e))
            results.append(_attach_universe_metadata(_empty_record(ticker), candidate))
            continue

        record = _attach_universe_metadata(record, candidate)
        _enrich_with_fmp_data(record, ticker, cfg, sleep_before=fmp_calls_made > 0)
        fmp_calls_made += 1

        record = _validate(record, ticker)
        _save_cache(ticker, record)
        results.append(record)

    return results
