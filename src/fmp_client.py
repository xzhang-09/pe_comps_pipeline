import os

import requests

FMP_BASE_URL = "https://financialmodelingprep.com/stable"
REQUEST_TIMEOUT_SECONDS = 10
FMP_FALLBACK_STATUS_CODES = {402, 403, 429}


def _api_key() -> str:
    key = os.environ.get("FMP_API_KEY")
    if not key:
        raise RuntimeError(
            "FMP_API_KEY environment variable is not set. "
            "Export FMP_API_KEY=<your key> before running."
        )
    return key


def _alternate_api_key() -> str | None:
    return os.environ.get("FMP_API_KEY_ALTERNATE")


def _profile_request(ticker: str, api_key: str) -> requests.Response:
    return requests.get(
        f"{FMP_BASE_URL}/profile",
        params={"symbol": ticker, "apikey": api_key},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def get_profile(ticker: str) -> dict | None:
    """
    Fetch FMP's /profile data for a ticker: market cap, sector, description.
    The only FMP endpoint this pipeline calls, by design — it's available
    on FMP's free tier for every symbol, unlike balance-sheet-statement,
    enterprise-values, key-metrics, and the screener, which return 402 on
    the free tier for anything beyond a handful of demo mega-caps. EV/EBITDA
    is instead derived from this profile's market cap plus SEC EDGAR XBRL
    fundamentals (see fetcher._enrich_with_fmp_data). If you have a paid FMP
    plan, see README's "Using a paid FMP plan" note for how to get direct
    multiples from key-metrics/enterprise-values instead.
    """
    resp = _profile_request(ticker, _api_key())
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        alternate_key = _alternate_api_key()
        if resp.status_code not in FMP_FALLBACK_STATUS_CODES or not alternate_key:
            raise
        resp = _profile_request(ticker, alternate_key)
        resp.raise_for_status()
    data = resp.json()
    return data[0] if data else None
