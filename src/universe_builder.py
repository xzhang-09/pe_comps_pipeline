import time

import yfinance as yf

from src import get_logger

logger = get_logger(__name__)

# Yahoo Finance rate-limits rapid sequential info lookups; this keeps the
# market-cap filtering loop under that threshold.
MARKET_CAP_REQUEST_DELAY_SECONDS = 0.5

# Maps GICS sector codes to yfinance's sector domain keys, used for the
# (unstable) primary lookup path before falling back to the hardcoded list.
GICS_SECTOR_TO_YFINANCE_KEY = {
    "20": "industrials",
}

MIN_MARKET_CAP_USD = 200_000_000

# Real US-listed industrial / manufacturing tickers used when the yfinance
# sector screener is unavailable or returns nothing usable.
FALLBACK_INDUSTRIAL_TICKERS = [
    "MMM", "HON", "GE", "EMR", "ITW", "PH", "ROK", "AME", "IEX", "GNRC",
    "SWK", "PNR", "RRX", "XYL", "FELE", "TT", "IR", "CARR", "OTIS", "LMT",
    "RTX", "GD", "NOC", "LHX", "TDG", "HII", "MOG.A", "HEI", "ESAB", "ITT",
    "WTS", "RXN", "ACCO", "LYTS", "KBAL", "GFF", "ASTE", "HCSG", "HI", "CSWI",
    "NVT", "TRMK", "DXPE", "CAT", "DE", "PCAR", "CMI", "DOV", "AOS", "ALLE",
    "MAS", "JCI", "LII", "PWR", "FAST", "GWW", "WSO", "AIT", "MSM", "SITE",
    "BLDR", "EXP", "VMC", "MLM", "NUE", "STLD", "X", "CLF", "RS", "ATI",
    "CRS", "KMT", "FLS", "CR", "DCI", "GTLS", "CFX", "HEES", "TKR", "ROLL",
    "B", "WCC", "AYI", "HUBB", "POWL", "THR", "MWA", "AWI", "JBT", "GGG",
    "SPXC", "CIR", "CSL", "EME", "MTZ", "BMI", "NDSN", "AAON", "LECO",
    "KAMN", "CW", "HXL", "TXT", "SPR",
]


def _sector_tickers_from_yfinance(gics_sector: str) -> list[str]:
    sector_key = GICS_SECTOR_TO_YFINANCE_KEY.get(gics_sector)
    if not sector_key:
        raise ValueError(f"No yfinance sector mapping for GICS sector {gics_sector!r}")

    sector = yf.Sector(sector_key)
    top_companies = sector.top_companies
    if top_companies is None or top_companies.empty:
        raise ValueError(f"yfinance sector {sector_key!r} returned no companies")

    return list(top_companies.index)


def _filter_by_market_cap(tickers: list[str]) -> list[str]:
    filtered = []
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(MARKET_CAP_REQUEST_DELAY_SECONDS)

        try:
            market_cap = yf.Ticker(ticker).info.get("marketCap")
        except Exception as e:
            logger.warning(f"{ticker} — failed to fetch market cap: {e}")
            continue

        if market_cap is not None and market_cap >= MIN_MARKET_CAP_USD:
            filtered.append(ticker)

    return filtered


def build(config: dict) -> list[str]:
    """
    Return list of candidate ticker symbols.
    Uses yfinance screener with hardcoded fallback.
    Filters out micro-caps below $200M market cap.
    Respects config['universe']['max_candidates'] limit.
    """
    gics_sector = config["target_company"]["gics_sector"]

    try:
        candidates = _sector_tickers_from_yfinance(gics_sector)
        logger.info(f"yfinance sector screener returned {len(candidates)} candidates")
    except Exception as e:
        logger.warning(f"yfinance sector screener failed: {e}. Falling back to hardcoded list.")
        candidates = list(FALLBACK_INDUSTRIAL_TICKERS)

    logger.info(f"Found {len(candidates)} candidate tickers before filtering")

    filtered = _filter_by_market_cap(candidates)
    logger.info(f"{len(filtered)} candidates remain after market cap filter")

    max_candidates = config["universe"]["max_candidates"]
    return filtered[:max_candidates]
