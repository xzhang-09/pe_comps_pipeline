import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
import yfinance as yf
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path("data/cache")
OUTPUTS_DIR = Path("outputs")
FAILED_TICKERS_CSV = OUTPUTS_DIR / "failed_tickers.csv"

EDGAR_HEADERS = {"User-Agent": "PE-Comps-Pipeline research@example.com"}
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

EBITDA_ROW_LABELS = ("EBITDA", "Normalized EBITDA")
REVENUE_ROW_LABELS = ("Total Revenue", "TotalRevenue")
CAPEX_ROW_LABELS = ("Capital Expenditure", "CapitalExpenditure")

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


def _record_failed_ticker(ticker: str, error_type: str, error_message: str) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = FAILED_TICKERS_CSV.exists()
    with open(FAILED_TICKERS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["ticker", "error_type", "error_message", "timestamp"])
        writer.writerow([
            ticker, error_type, error_message,
            datetime.now(timezone.utc).isoformat(),
        ])


def _row_value(financials, labels: tuple[str, ...]):
    if financials is None or financials.empty:
        return None
    for label in labels:
        if label in financials.index:
            try:
                return float(financials.loc[label].iloc[0])
            except Exception:
                return None
    return None


def _ebitda_and_margin(ticker: str, financials):
    ebitda = _row_value(financials, EBITDA_ROW_LABELS)
    revenue = _row_value(financials, REVENUE_ROW_LABELS)
    if ebitda is None:
        logger.warning(f"{ticker} — EBITDA not available, setting None")
    margin = ebitda / revenue if (ebitda is not None and revenue) else None
    return ebitda, revenue, margin


def _revenue_cagr(financials):
    if financials is None or financials.empty:
        return None
    for label in REVENUE_ROW_LABELS:
        if label not in financials.index:
            continue
        row = financials.loc[label]
        if len(row) < 4:
            return None
        recent, past = row.iloc[0], row.iloc[3]
        if not past or past <= 0:
            return None
        try:
            return (recent / past) ** (1 / 3) - 1
        except Exception:
            return None
    return None


def _capex(financials_cashflow):
    if financials_cashflow is None or financials_cashflow.empty:
        return None
    for label in CAPEX_ROW_LABELS:
        if label in financials_cashflow.index:
            try:
                return float(financials_cashflow.loc[label].iloc[0])
            except Exception:
                return None
    return None


def _filing_url_from_hit(hit: dict) -> str | None:
    source = hit.get("_source", {})
    cik = source.get("cik")
    accession = source.get("adsh")
    file_name = source.get("file_name")
    if not (cik and accession and file_name):
        return None
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{file_name}"


def _fetch_edgar_description(ticker: str) -> str | None:
    params = {
        "q": f'"{ticker}"',
        "dateRange": "custom",
        "startdt": "2022-01-01",
        "forms": "10-K",
    }
    resp = requests.get(EDGAR_SEARCH_URL, params=params, headers=EDGAR_HEADERS, timeout=10)
    resp.raise_for_status()
    hits = resp.json().get("hits", {}).get("hits", [])
    if not hits:
        return None

    filing_url = _filing_url_from_hit(hits[0])
    if not filing_url:
        return None

    filing_resp = requests.get(filing_url, headers=EDGAR_HEADERS, timeout=10)
    filing_resp.raise_for_status()
    text = filing_resp.text

    item1_idx = text.find("Item 1")
    start = item1_idx if item1_idx != -1 else 0
    return text[start:start + 1500]


def _fetch_business_description(ticker: str, yf_ticker) -> tuple[str | None, str | None]:
    try:
        description = _fetch_edgar_description(ticker)
        if description:
            return description, "edgar"
    except Exception as e:
        logger.warning(f"{ticker} — EDGAR description failed: {e}")

    try:
        summary = yf_ticker.info.get("longBusinessSummary")
        if summary:
            return summary, "yfinance"
    except Exception as e:
        logger.warning(f"{ticker} — yfinance description fallback failed: {e}")

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
    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.info
    financials = yf_ticker.financials

    company_name = info.get("longName") or info.get("shortName") or ticker

    market_cap = info.get("marketCap")
    market_cap_usd_mm = market_cap / 1e6 if market_cap is not None else None

    ebitda, revenue, ebitda_margin = _ebitda_and_margin(ticker, financials)
    revenue_ttm_usd_mm = revenue / 1e6 if revenue is not None else None

    gross_margin = info.get("grossMargins")

    revenue_cagr_3yr = _revenue_cagr(financials)

    total_debt = info.get("totalDebt")
    total_cash = info.get("totalCash")
    net_debt_ebitda = None
    if total_debt is not None and total_cash is not None and ebitda:
        net_debt_ebitda = (total_debt - total_cash) / ebitda

    capex = _capex(getattr(yf_ticker, "cashflow", None))
    capex_revenue = abs(capex) / revenue if (capex is not None and revenue) else None

    enterprise_value = info.get("enterpriseValue")
    ev_ebitda = enterprise_value / ebitda if (enterprise_value is not None and ebitda) else None
    ev_revenue = enterprise_value / revenue if (enterprise_value is not None and revenue) else None

    gics_sector = info.get("sector")

    business_description, description_source = _fetch_business_description(ticker, yf_ticker)

    return {
        "ticker": ticker,
        "company_name": company_name,
        "market_cap_usd_mm": market_cap_usd_mm,
        "revenue_ttm_usd_mm": revenue_ttm_usd_mm,
        "ebitda_margin": ebitda_margin,
        "gross_margin": gross_margin,
        "revenue_cagr_3yr": revenue_cagr_3yr,
        "net_debt_ebitda": net_debt_ebitda,
        "capex_revenue": capex_revenue,
        "ev_ebitda": ev_ebitda,
        "ev_revenue": ev_revenue,
        "gics_sector": gics_sector,
        "business_description": business_description,
        "description_source": description_source,
        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
    }


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

    for i, ticker in enumerate(tickers, start=1):
        logger.info(f"Processing {i}/{total}: {ticker}")

        cached = _load_cache(ticker)
        if cached is not None:
            logger.info(f"{ticker} — loaded from cache")
            results.append(cached)
            continue

        try:
            record = _fetch_single(ticker)
            record = _validate(record, ticker)
            _save_cache(ticker, record)
            results.append(record)
        except Exception as e:
            logger.error(f"{ticker} — failed after 3 retries: {e}")
            _record_failed_ticker(ticker, type(e).__name__, str(e))
            results.append(_empty_record(ticker))

    return results
