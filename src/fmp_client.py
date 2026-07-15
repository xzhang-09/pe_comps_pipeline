import os

import requests

FMP_BASE_URL = "https://financialmodelingprep.com/stable"
REQUEST_TIMEOUT_SECONDS = 10


def _api_key() -> str:
    key = os.environ.get("FMP_API_KEY")
    if not key:
        raise RuntimeError(
            "FMP_API_KEY environment variable is not set. "
            "Export FMP_API_KEY=<your key> before running."
        )
    return key


def get_profile(ticker: str) -> dict | None:
    """
    Fetch FMP's /profile data for a ticker: market cap, sector, description.
    Available on FMP's free tier for every symbol — unlike balance-sheet-statement,
    enterprise-values, key-metrics, and the screener, which return 402 on the
    free tier for anything beyond a handful of demo mega-caps.
    """
    resp = requests.get(
        f"{FMP_BASE_URL}/profile",
        params={"symbol": ticker, "apikey": _api_key()},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else None
