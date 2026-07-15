import json
import time
from pathlib import Path

from src import fmp_client, get_logger, sic_universe_builder

logger = get_logger(__name__)

# FMP's free tier doesn't allow per-second hammering; this paces the
# market-cap filtering loop.
MARKET_CAP_REQUEST_DELAY_SECONDS = 1

# Lowered from $200M so SIC-discovered small caps near the target's own
# scale aren't filtered out before they ever reach comp selection — a $200M
# floor defeats the purpose of widening the universe toward smaller peers.
MIN_MARKET_CAP_USD = 30_000_000

# Successful market-cap lookups are cached indefinitely (same philosophy as
# fetcher.py's per-ticker cache) so a repeat run of build() doesn't re-spend
# FMP quota re-checking ~170 tickers every single time.
MARKET_CAP_CACHE_PATH = Path("data/cache/universe_market_cap.json")


def _load_market_cap_cache() -> dict:
    if not MARKET_CAP_CACHE_PATH.exists():
        return {}
    with open(MARKET_CAP_CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_market_cap_cache(cache: dict) -> None:
    MARKET_CAP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MARKET_CAP_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _filter_by_market_cap(tickers: list[str]) -> list[str]:
    """
    Keep a ticker unless we successfully fetch its market cap and confirm
    it's below MIN_MARKET_CAP_USD. Lookup failures (rate limiting, missing
    data, etc.) are not treated as disqualifying — we'd rather keep a few
    small caps than lose large swaths of the universe to a flaky API.
    Successful lookups (including a legitimately missing marketCap field)
    are cached indefinitely; failures are not, so they get retried next run.
    """
    cache = _load_market_cap_cache()
    cache_dirty = False
    made_a_call = False

    filtered = []
    for ticker in tickers:
        if ticker in cache:
            market_cap = cache[ticker]
            logger.info(f"{ticker} — market cap loaded from cache")
        else:
            if made_a_call:
                time.sleep(MARKET_CAP_REQUEST_DELAY_SECONDS)
            made_a_call = True

            try:
                profile = fmp_client.get_profile(ticker)
            except Exception as e:
                logger.warning(f"{ticker} — failed to fetch market cap: {e}. Keeping ticker.")
                filtered.append(ticker)
                continue

            market_cap = profile.get("marketCap") if profile else None
            cache[ticker] = market_cap
            cache_dirty = True

        if market_cap is None:
            logger.warning(f"{ticker} — market cap not available. Keeping ticker.")
            filtered.append(ticker)
        elif market_cap < MIN_MARKET_CAP_USD:
            logger.info(f"{ticker} — market cap {market_cap} below ${MIN_MARKET_CAP_USD}, filtering out")
        else:
            filtered.append(ticker)

    if cache_dirty:
        _save_market_cap_cache(cache)

    return filtered


def _dedup(tickers: list[str]) -> list[str]:
    seen = set()
    out = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def build(config: dict) -> list[str]:
    """
    Return a list of candidate ticker symbols sourced entirely from the
    target company's own SIC codes — no hardcoded ticker list. The universe
    should reflect what this specific target actually competes with, not a
    fixed set of GICS sectors or a memory-curated company list (which has
    no way to stay accurate as tickers change and can't cover small caps
    the way an exhaustive SEC registry query can).

    Candidates come from two buckets:
    - primary: companies discovered via target_company['primary_sic_codes']
      — expected to actually surface as comps
    - adjacent: companies discovered via target_company['adjacent_sic_codes']
      only — included purely to add training-data volume/diversity, not
      because they're expected to surface as comps

    Each bucket is market-cap filtered and capped according to
    universe['primary_allocation_pct'] of universe['max_candidates'], so a
    SIC code with a much larger company count can't crowd out the target's
    actual industry.
    """
    target = config["target_company"]
    primary_sics = target["primary_sic_codes"]
    adjacent_sics = target.get("adjacent_sic_codes", [])

    primary_candidates = _dedup(sic_universe_builder.discover_tickers_by_sic(primary_sics))
    logger.info(f"Primary bucket — {len(primary_candidates)} candidates (SIC {primary_sics})")

    primary_set = set(primary_candidates)
    adjacent_candidates = [
        t for t in _dedup(sic_universe_builder.discover_tickers_by_sic(adjacent_sics))
        if t not in primary_set
    ]
    logger.info(f"Adjacent bucket — {len(adjacent_candidates)} candidates (SIC {adjacent_sics})")

    max_candidates = config["universe"]["max_candidates"]
    primary_allocation_pct = config["universe"]["primary_allocation_pct"]
    primary_quota = round(max_candidates * primary_allocation_pct)
    adjacent_quota = max_candidates - primary_quota

    filtered_primary = _filter_by_market_cap(primary_candidates)[:primary_quota]
    filtered_adjacent = _filter_by_market_cap(adjacent_candidates)[:adjacent_quota]
    logger.info(
        f"After market cap filter — primary: {len(filtered_primary)}/{primary_quota}, "
        f"adjacent: {len(filtered_adjacent)}/{adjacent_quota}"
    )

    return filtered_primary + filtered_adjacent
