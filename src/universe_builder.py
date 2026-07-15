from src import get_logger, sic_universe_builder

logger = get_logger(__name__)

# In millions, matching the market_cap_usd_mm scale fetcher.fetch_batch
# produces (see filter_by_market_cap below). Set low enough to keep smaller
# public peers in the candidate pool while filtering obvious micro-caps.
MIN_MARKET_CAP_USD_MM = 30.0


def filter_by_market_cap(companies: list[dict]) -> list[dict]:
    """
    Drop companies below MIN_MARKET_CAP_USD_MM, using the market_cap_usd_mm
    field fetcher.fetch_batch already populated via FMP during valuation
    enrichment — this makes zero additional FMP calls (a separate pre-fetch
    market-cap lookup would compete with fetcher's own FMP calls for the
    same daily quota). A missing market cap (lookup failure, free-tier gap,
    etc.) is not treated as disqualifying — we'd rather keep a few
    small/unknown caps than lose large swaths of the universe to incomplete
    data.
    """
    filtered = []
    for company in companies:
        market_cap = company.get("market_cap_usd_mm")
        if market_cap is None:
            filtered.append(company)
        elif market_cap < MIN_MARKET_CAP_USD_MM:
            logger.info(
                f"{company.get('ticker')} — market cap ${market_cap:.1f}mm below "
                f"${MIN_MARKET_CAP_USD_MM:.0f}mm, filtering out"
            )
        else:
            filtered.append(company)

    return filtered


def _dedup(tickers: list[str]) -> list[str]:
    seen = set()
    out = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def _sic_parts(sic: str | None) -> tuple[str | None, str | None]:
    if not sic:
        return None, None
    return sic[:2], sic[:3]


def _cluster_for_sics(sic_codes: list[str], sic_clusters: dict[str, str]) -> str | None:
    for sic in sic_codes:
        if sic in sic_clusters:
            return sic_clusters[sic]
    return None


def _records_for_bucket(sic_codes: list[str], bucket: str, sic_clusters: dict[str, str]) -> list[dict]:
    bucket_key = f"{bucket}_matched_sic_codes"
    records_by_ticker: dict[str, dict] = {}
    for sic in sic_codes:
        for ticker in sic_universe_builder.discover_tickers_by_sic([sic]):
            record = records_by_ticker.setdefault(
                ticker,
                {
                    "ticker": ticker,
                    "matched_sic_codes": [],
                    "primary_matched_sic_codes": [],
                    "adjacent_matched_sic_codes": [],
                    "source_bucket": bucket,
                    "sic_2_digit": None,
                    "sic_3_digit": None,
                    "industry_cluster": None,
                    "candidate_source": "sec_sic",
                },
            )
            record["matched_sic_codes"].append(sic)
            record[bucket_key].append(sic)
            if record["sic_2_digit"] is None:
                record["sic_2_digit"], record["sic_3_digit"] = _sic_parts(sic)
            if record["industry_cluster"] is None:
                record["industry_cluster"] = _cluster_for_sics([sic], sic_clusters)

    return list(records_by_ticker.values())


def _merge_records(primary_records: list[dict], adjacent_records: list[dict], sic_clusters: dict[str, str]) -> list[dict]:
    merged: dict[str, dict] = {}
    for record in primary_records + adjacent_records:
        ticker = record["ticker"]
        if ticker not in merged:
            merged[ticker] = {**record}
            continue

        existing = merged[ticker]
        existing["matched_sic_codes"] = _dedup(existing["matched_sic_codes"] + record["matched_sic_codes"])
        existing["primary_matched_sic_codes"] = _dedup(
            existing["primary_matched_sic_codes"] + record["primary_matched_sic_codes"]
        )
        existing["adjacent_matched_sic_codes"] = _dedup(
            existing["adjacent_matched_sic_codes"] + record["adjacent_matched_sic_codes"]
        )
        if existing["primary_matched_sic_codes"]:
            existing["source_bucket"] = "primary"
        elif existing["adjacent_matched_sic_codes"]:
            existing["source_bucket"] = "adjacent"
        if existing["industry_cluster"] is None:
            existing["industry_cluster"] = _cluster_for_sics(existing["matched_sic_codes"], sic_clusters)

    return list(merged.values())


def build(config: dict) -> list[dict]:
    """
    Return candidate records sourced entirely from the target company's own
    SIC codes — no hardcoded ticker list. Each record carries the ticker plus
    SIC/bucket metadata that downstream fetch and feature stages need.

    Candidates come from two buckets:
    - primary: companies discovered via target_company['primary_sic_codes']
      — expected to actually surface as comps
    - adjacent: companies discovered via target_company['adjacent_sic_codes']
      only — included to widen recall/diversity in the comp pool, not
      because they're expected to surface as comps themselves

    Both buckets are capped according to universe['primary_allocation_pct']
    of universe['max_candidates']. FMP market-cap filtering is deliberately
    not performed here, so the daily call budget is reserved for valuation
    enrichment.
    """
    target = config["target_company"]
    primary_sics = target["primary_sic_codes"]
    adjacent_sics = target.get("adjacent_sic_codes", [])
    universe_config = config.get("universe", {})

    sic_clusters = universe_config.get("sic_clusters", {})
    primary_records = _records_for_bucket(primary_sics, "primary", sic_clusters)
    adjacent_records = _records_for_bucket(adjacent_sics, "adjacent", sic_clusters)
    merged_records = _merge_records(primary_records, adjacent_records, sic_clusters)

    primary_candidates = [r for r in merged_records if r["source_bucket"] == "primary"]
    adjacent_candidates = [r for r in merged_records if r["source_bucket"] == "adjacent"]
    logger.info(f"Primary bucket — {len(primary_candidates)} candidates (SIC {primary_sics})")
    logger.info(f"Adjacent bucket — {len(adjacent_candidates)} candidates (SIC {adjacent_sics})")

    max_candidates = config["universe"]["max_candidates"]
    primary_allocation_pct = config["universe"]["primary_allocation_pct"]
    primary_quota = round(max_candidates * primary_allocation_pct)
    adjacent_quota = max_candidates - primary_quota

    filtered_primary = primary_candidates[:primary_quota]
    filtered_adjacent = adjacent_candidates[:adjacent_quota]
    logger.info(
        f"After quota allocation — primary: {len(filtered_primary)}/{primary_quota}, "
        f"adjacent: {len(filtered_adjacent)}/{adjacent_quota}"
    )

    return filtered_primary + filtered_adjacent
