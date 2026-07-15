import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import edgar
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src import fmp_client, get_logger

logger = get_logger(__name__)

CACHE_DIR = Path("data/cache")
OUTPUTS_DIR = Path("outputs")
FAILED_TICKERS_CSV = OUTPUTS_DIR / "failed_tickers.csv"

EDGAR_IDENTITY = "PE-Comps-Pipeline research@example.com"
edgar.set_identity(EDGAR_IDENTITY)

BUSINESS_DESCRIPTION_MAX_CHARS = 1500

# FMP is only used for market cap / sector / description-fallback now — its
# free tier doesn't expose balance-sheet/key-metrics for non-demo tickers,
# so debt, cash, revenue, EBITDA, gross profit, and capex all come from
# SEC XBRL via edgartools instead.
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
    "description_source",
)


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.json"


def _load_cache(ticker: str) -> dict | None:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache(ticker: str, record: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(ticker), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def _reset_failed_tickers_csv() -> None:
    """
    Start each fetch_batch() run with a clean file. Without this, failures
    from past runs (including ones from since-removed code paths, e.g. the
    old Yahoo Finance fetcher) accumulate indefinitely and get double-counted
    by reporter.py's failed_fetch_count, which just counts rows in this file.
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

    row = matches.iloc[0]
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


def _net_debt(ticker: str, balance_sheet_df: pd.DataFrame) -> float | None:
    """Net debt = interest-bearing debt - cash & marketable securities, all
    from the same EDGAR XBRL balance sheet used for everything else. Cross-
    checked against FMP's own totalDebt/cashAndCashEquivalents for AAPL."""
    cash = _concept_value(balance_sheet_df, "CashAndMarketableSecurities")
    short_term_debt = _concept_value(balance_sheet_df, "ShortTermDebt")
    current_portion_ltd = _concept_value(balance_sheet_df, "CurrentPortionOfLongTermDebt")
    long_term_debt = _concept_value(balance_sheet_df, "LongTermDebt")

    if cash is None and short_term_debt is None and current_portion_ltd is None and long_term_debt is None:
        logger.warning(f"{ticker} — debt/cash not available from EDGAR XBRL, setting net_debt None")
        return None

    total_debt = (short_term_debt or 0.0) + (current_portion_ltd or 0.0) + (long_term_debt or 0.0)
    return total_debt - (cash or 0.0)


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
        "ev_ebitda": None,
        "ev_revenue": None,
        "gics_sector": None,
        "business_description": business_description,
        "description_source": description_source,
        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _enrich_with_fmp_data(record: dict, ticker: str, sleep_before: bool) -> None:
    """
    Best-effort FMP lookup for market cap / sector, and a fallback for
    business_description when EDGAR's didn't come through. Never raises and
    never causes the company to be dropped — on any failure the record just
    keeps its EDGAR-sourced fields with market_cap/EV left as None.
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

    if not record.get("business_description"):
        description = profile.get("description")
        if description:
            record["business_description"] = description[:BUSINESS_DESCRIPTION_MAX_CHARS]
            record["description_source"] = "fmp"

    market_cap = profile.get("marketCap")
    record["market_cap_usd_mm"] = market_cap / 1e6 if market_cap is not None else None
    record["gics_sector"] = profile.get("sector")

    if market_cap is None:
        return

    revenue = record.get("revenue_ttm_usd_mm")
    ebitda_margin = record.get("ebitda_margin")
    ebitda_usd_mm = ebitda_margin * revenue if (ebitda_margin is not None and revenue is not None) else None

    # net_debt_ebitda was already computed from EDGAR alone in _fetch_single;
    # recover net_debt_usd_mm from it here to build EV without re-fetching.
    net_debt_ebitda = record.get("net_debt_ebitda")
    net_debt_usd_mm = net_debt_ebitda * ebitda_usd_mm if (net_debt_ebitda is not None and ebitda_usd_mm) else None

    if net_debt_usd_mm is None:
        return

    market_cap_usd_mm = market_cap / 1e6
    ev_usd_mm = market_cap_usd_mm + net_debt_usd_mm

    if ebitda_usd_mm:
        record["ev_ebitda"] = ev_usd_mm / ebitda_usd_mm
    if revenue:
        record["ev_revenue"] = ev_usd_mm / revenue


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

    return record


def _empty_record(ticker: str) -> dict:
    record = {field: None for field in EMPTY_FIELDS}
    record["ticker"] = ticker
    record["fetch_timestamp"] = datetime.now(timezone.utc).isoformat()
    return record


def fetch_batch(tickers: list[str], config: dict) -> list[dict]:
    """
    Fetch financial data for all tickers.
    Returns list of dicts (one per ticker, including failed ones with None values).
    Uses cache — skips API call if cache/{ticker}.json exists.
    Writes failures to outputs/failed_tickers.csv.
    """
    results = []
    total = len(tickers)
    fmp_calls_made = 0

    _reset_failed_tickers_csv()

    for i, ticker in enumerate(tickers, start=1):
        logger.info(f"Processing {i}/{total}: {ticker}")

        cached = _load_cache(ticker)
        if cached is not None:
            logger.info(f"{ticker} — loaded from cache")
            results.append(cached)
            continue

        try:
            record = _fetch_single(ticker)
        except Exception as e:
            logger.error(f"{ticker} — failed after 3 retries: {e}")
            _record_failed_ticker(ticker, type(e).__name__, str(e))
            results.append(_empty_record(ticker))
            continue

        _enrich_with_fmp_data(record, ticker, sleep_before=fmp_calls_made > 0)
        fmp_calls_made += 1

        record = _validate(record, ticker)
        _save_cache(ticker, record)
        results.append(record)

    return results
