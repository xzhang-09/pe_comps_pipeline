import time

from src import fmp_client, get_logger

logger = get_logger(__name__)

# FMP's free tier doesn't allow per-second hammering; this paces the
# market-cap filtering loop.
MARKET_CAP_REQUEST_DELAY_SECONDS = 1

MIN_MARKET_CAP_USD = 200_000_000

# Real US-listed tickers per GICS sector. FMP's screener/stock-list endpoints
# are premium-only on the free tier (402), so candidate discovery is purely
# this hardcoded list rather than a dynamic sector lookup.
FALLBACK_TICKERS_BY_SECTOR = {
    # Industrials
    "20": [
        "MMM", "HON", "GE", "EMR", "ITW", "PH", "ROK", "AME", "IEX", "GNRC",
        "SWK", "PNR", "RRX", "XYL", "FELE", "TT", "IR", "CARR", "OTIS", "LMT",
        "RTX", "GD", "NOC", "LHX", "TDG", "HII", "MOG.A", "HEI", "ESAB", "ITT",
        "WTS", "RXN", "ACCO", "KBAL", "GFF", "ASTE", "HI", "CSWI",
        "NVT", "DXPE", "CAT", "DE", "PCAR", "CMI", "DOV", "AOS", "ALLE",
        "JCI", "LII", "PWR", "FAST", "GWW", "WSO", "AIT", "MSM", "SITE",
        "BLDR", "EXP", "VMC", "MLM", "NUE", "STLD", "X", "CLF", "RS", "ATI",
        "CRS", "KMT", "FLS", "CR", "DCI", "GTLS", "CFX", "HEES", "TKR", "ROLL",
        "B", "WCC", "AYI", "HUBB", "POWL", "THR", "MWA", "AWI", "JBT", "GGG",
        "SPXC", "CIR", "CSL", "EME", "MTZ", "BMI", "NDSN", "AAON", "LECO",
        "KAMN", "CW", "HXL", "TXT", "SPR", "WAB", "J", "URI", "FTV",
    ],
    # Healthcare Equipment
    "35": [
        "ABT", "MDT", "SYK", "BSX", "ZBH", "EW", "HOLX", "DXCM", "ISRG", "RMD",
        "BAX", "BDX", "HAE", "ICUI", "AMED", "NVCR", "RGEN", "IART", "MMSI", "PEN",
        "OSIS", "ATRC", "NTRA", "INSP", "SWAV", "AXNX", "TNDM", "NVST", "ALGN", "VRTX",
        "GMED", "CNMD", "STE", "TFX", "XRAY",
    ],
    # Technology Hardware
    "45": [
        "AAPL", "DELL", "HPQ", "HPE", "NTAP", "STX", "WDC", "PSTG", "SMCI",
        "CSCO", "ANET", "CIEN", "VIAV", "CLFD", "ATEN", "ARLO", "CALX", "DIGI", "LIQT",
        "PCTI", "SIFY", "SMSI", "SPOK", "SYNA", "TTEC", "UTSI", "VNET",
        "GLW", "TEL", "APH", "KEYS", "FFIV", "MSI",
    ],
}


def _filter_by_market_cap(tickers: list[str]) -> list[str]:
    """
    Keep a ticker unless we successfully fetch its market cap and confirm
    it's below MIN_MARKET_CAP_USD. Lookup failures (rate limiting, missing
    data, etc.) are not treated as disqualifying — we'd rather keep a few
    small caps than lose large swaths of the universe to a flaky API.
    """
    filtered = []
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(MARKET_CAP_REQUEST_DELAY_SECONDS)

        try:
            profile = fmp_client.get_profile(ticker)
            market_cap = profile.get("marketCap") if profile else None
        except Exception as e:
            logger.warning(f"{ticker} — failed to fetch market cap: {e}. Keeping ticker.")
            filtered.append(ticker)
            continue

        if market_cap is None:
            logger.warning(f"{ticker} — market cap not available. Keeping ticker.")
            filtered.append(ticker)
        elif market_cap < MIN_MARKET_CAP_USD:
            logger.info(f"{ticker} — market cap {market_cap} below ${MIN_MARKET_CAP_USD}, filtering out")
        else:
            filtered.append(ticker)

    return filtered


def build(config: dict) -> list[str]:
    """
    Return list of candidate ticker symbols across all configured industries.
    Uses the hardcoded fallback list per industry (FMP has no free dynamic
    sector-screener endpoint). Merges and deduplicates across industries,
    filters out micro-caps below $200M market cap via FMP, and respects
    config['universe']['max_candidates'].
    """
    industries = config["universe"]["industries"]

    merged = []
    seen = set()

    for industry in industries:
        gics_sector = industry["gics_sector"]
        label = industry.get("label", gics_sector)

        candidates = list(FALLBACK_TICKERS_BY_SECTOR.get(gics_sector, []))

        added = 0
        for ticker in candidates:
            if ticker not in seen:
                seen.add(ticker)
                merged.append(ticker)
                added += 1

        logger.info(
            f"{label} ({gics_sector}) — {len(candidates)} candidates found, "
            f"{added} new tickers added after de-dup"
        )

    logger.info(f"Merged universe before market cap filter: {len(merged)} tickers across {len(industries)} industries")

    filtered = _filter_by_market_cap(merged)
    logger.info(f"{len(filtered)} candidates remain after market cap filter")

    max_candidates = config["universe"]["max_candidates"]
    return filtered[:max_candidates]
