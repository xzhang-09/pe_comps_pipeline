import json
import os
import re
import time
from pathlib import Path

import requests

from src import get_logger

logger = get_logger(__name__)

DEFAULT_SEC_IDENTITY = "PE-Comps-Pipeline research@example.com"
SEC_IDENTITY = os.environ.get("SEC_IDENTITY", DEFAULT_SEC_IDENTITY)
HEADERS = {"User-Agent": SEC_IDENTITY}
REQUEST_TIMEOUT_SECONDS = 10

# SEC's stated limit is 10 requests/second; this stays well under it.
SEC_REQUEST_DELAY_SECONDS = 0.2

BROWSE_EDGAR_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# Each browse-edgar page returns at most 100 entries; an empty/partial page
# means we've reached the end.
PAGE_SIZE = 100
MAX_PAGES_PER_SIC = 20

CACHE_DIR = Path("data/cache")


def _cache_path(sic: str) -> Path:
    return CACHE_DIR / f"sic_universe_{sic}.json"


def _load_cache(sic: str) -> list[str] | None:
    path = _cache_path(sic)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_cache(sic: str, tickers: list[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(sic), "w", encoding="utf-8") as f:
        json.dump(tickers, f, indent=2)


def _fetch_ciks_for_sic(sic: str, max_pages: int = MAX_PAGES_PER_SIC) -> list[int]:
    """
    Enumerate every CIK that has filed a 10-K under this SIC code, via
    SEC's browse-edgar atom feed. The feed's <title> field is unreliable
    (a known SEC-side bug renders it as "ARRAY(0x...)" for many SIC
    queries), so only the <cik> tag is parsed.
    """
    ciks = []
    for page in range(max_pages):
        start = page * PAGE_SIZE
        resp = requests.get(
            BROWSE_EDGAR_URL,
            params={
                "action": "getcompany",
                "SIC": sic,
                "type": "10-K",
                "dateb": "",
                "owner": "include",
                "count": PAGE_SIZE,
                "start": start,
                "output": "atom",
            },
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        page_ciks = [int(c) for c in re.findall(r"<cik>(\d+)</cik>", resp.text)]
        ciks.extend(page_ciks)

        if len(page_ciks) < PAGE_SIZE:
            break
        time.sleep(SEC_REQUEST_DELAY_SECONDS)

    return ciks


def fetch_company_profile(cik: int) -> dict | None:
    """
    Fetch a CIK's full submissions profile (tickers, SIC code, name, etc.)
    via SEC's submissions API. Returns None for foreign private issuers,
    shell companies, or any CIK with no submissions on file (404) — this
    also covers delisted/acquired companies, since EDGAR keeps their
    historical submissions data indefinitely even after delisting.
    """
    resp = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _fetch_company_tickers(cik: int) -> list[str]:
    """Look up a CIK's current US ticker(s). Returns an empty list for
    foreign private issuers, shell companies, or anything else without a
    standard US ticker."""
    profile = fetch_company_profile(cik)
    if profile is None:
        return []
    return [t for t in profile.get("tickers", []) if t]


# Above this many SEC filers, a SIC code is almost certainly broader than
# any single comp universe (validation hit this with 7373, "computer
# integrated systems design") and the per-CIK ticker lookups it triggers
# dominate the run's SEC request budget. universe_builder.build treats
# crossing it as an error unless universe.allow_broad_sic_codes is set.
BROAD_SIC_CIK_THRESHOLD = 500


def preflight_sic_codes(sic_codes: list[str]) -> dict[str, int]:
    """
    Cheap SIC -> filer-count check run before ticker discovery, so a
    zero-result or too-broad code is caught before the expensive per-CIK
    ticker lookups (one SEC request each) rather than discovered mid-run.
    Uses the ticker cache where one exists (that count is exact); otherwise
    one browse-edgar enumeration per code (~1 request per 100 CIKs), which
    discover_tickers_by_sic would have issued anyway.
    """
    counts = {}
    for sic in dict.fromkeys(sic_codes):
        cached = _load_cache(sic)
        counts[sic] = len(cached) if cached is not None else len(_fetch_ciks_for_sic(sic))
    return counts


def discover_tickers_by_sic(sic_codes: list[str]) -> list[str]:
    """
    Return a deduplicated list of tickers for every company that has filed
    a 10-K under any of the given SIC codes. Results per SIC code are
    cached indefinitely (data/cache/sic_universe_{sic}.json) since this
    mapping changes rarely and re-enumerating costs one request per CIK.
    """
    seen = set()
    tickers: list[str] = []

    for sic in sic_codes:
        cached = _load_cache(sic)
        if cached is not None:
            logger.info(f"SIC {sic} — {len(cached)} tickers loaded from cache")
            sic_tickers = cached
        else:
            ciks = _fetch_ciks_for_sic(sic)
            logger.info(f"SIC {sic} — {len(ciks)} CIKs found, looking up tickers")

            sic_tickers = []
            for cik in ciks:
                time.sleep(SEC_REQUEST_DELAY_SECONDS)
                try:
                    cik_tickers = _fetch_company_tickers(cik)
                except Exception as e:
                    logger.warning(f"CIK {cik} — failed to fetch tickers: {e}. Skipping.")
                    continue
                if cik_tickers:
                    sic_tickers.append(cik_tickers[0])

            logger.info(f"SIC {sic} — {len(sic_tickers)} of {len(ciks)} CIKs have a usable ticker")
            _save_cache(sic, sic_tickers)

        for ticker in sic_tickers:
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)

    return tickers
