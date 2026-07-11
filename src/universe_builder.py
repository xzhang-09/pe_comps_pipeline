from src import embedding_universe_builder, get_logger, llm_analyzer, sic_universe_builder
from src.config_schema import PipelineConfig, as_config
from src.records import CompanyRecord

logger = get_logger(__name__)
_last_discovery_snapshot: dict = {}

# In millions, matching the market_cap_usd_mm scale fetcher.fetch_batch
# produces (see filter_by_market_cap below). Set low enough to keep smaller
# public peers in the candidate pool while filtering obvious micro-caps.
MIN_MARKET_CAP_USD_MM = 30.0


def filter_by_market_cap(companies: list[CompanyRecord]) -> list[CompanyRecord]:
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


def filter_by_financials(companies: list[CompanyRecord], config: PipelineConfig | dict) -> list[CompanyRecord]:
    """Apply optional revenue and EBITDA-margin hard filters from config.
    Missing values are kept, matching filter_by_market_cap's data-gap behavior."""
    cfg = as_config(config)
    min_revenue = cfg.universe.min_revenue_usd_mm
    max_revenue = cfg.universe.max_revenue_usd_mm
    min_margin = cfg.universe.min_ebitda_margin
    if min_revenue is None and max_revenue is None and min_margin is None:
        return companies

    filtered = []
    for company in companies:
        ticker = company.get("ticker")
        revenue = company.get("revenue_ttm_usd_mm")
        margin = company.get("ebitda_margin")
        if revenue is not None and min_revenue is not None and revenue < min_revenue:
            logger.info(f"{ticker} — revenue ${revenue:.1f}mm below ${min_revenue:.0f}mm, filtering out")
        elif revenue is not None and max_revenue is not None and revenue > max_revenue:
            logger.info(f"{ticker} — revenue ${revenue:.1f}mm above ${max_revenue:.0f}mm, filtering out")
        elif margin is not None and min_margin is not None and margin < min_margin:
            logger.info(f"{ticker} — EBITDA margin {margin:.1%} below {min_margin:.1%}, filtering out")
        else:
            filtered.append(company)
    return filtered


def filter_by_domicile(companies: list[CompanyRecord], allowed_countries: set[str] | None = None) -> list[CompanyRecord]:
    """Drop companies domiciled outside the allowed set (default: US only).
    Country comes from FMP's raw_fmp_profile.country field populated during
    enrichment. Companies with no FMP profile (country unknown) are kept so a
    data gap doesn't silently discard legitimate candidates — the analyst can
    spot them in the failed-tickers report."""
    if allowed_countries is None:
        allowed_countries = {"US"}
    filtered = []
    for company in companies:
        profile = company.get("raw_fmp_profile") or {}
        country = profile.get("country")
        if country is None:
            filtered.append(company)
        elif country in allowed_countries:
            filtered.append(company)
        else:
            logger.info(
                f"{company.get('ticker')} — domicile '{country}' not in "
                f"{allowed_countries}, filtering out"
            )
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


def _analyst_candidate(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "matched_sic_codes": [],
        "primary_matched_sic_codes": [],
        "adjacent_matched_sic_codes": [],
        "source_bucket": "analyst_specified",
        "sic_2_digit": None,
        "sic_3_digit": None,
        "industry_cluster": None,
        "candidate_source": "analyst_specified",
    }


def _apply_analyst_ticker_overrides(records: list[dict], must_include: list[str], exclude: list[str]) -> list[dict]:
    exclude_set = set(exclude)
    records_by_ticker = {record["ticker"].upper(): {**record, "ticker": record["ticker"].upper()} for record in records}
    for ticker in must_include:
        if ticker in exclude_set:
            continue
        record = records_by_ticker.get(ticker)
        if record is None:
            records_by_ticker[ticker] = _analyst_candidate(ticker)
        else:
            record["candidate_source"] = "analyst_specified"
            record["universe_metadata"] = {**record.get("universe_metadata", {}), "candidate_source": "analyst_specified"}
    return [record for ticker, record in records_by_ticker.items() if ticker not in exclude_set]


def _apply_bucket_quotas(records: list[dict], max_candidates: int, primary_allocation_pct: float) -> list[dict]:
    primary_candidates = [r for r in records if r["source_bucket"] == "primary"]
    adjacent_candidates = [r for r in records if r["source_bucket"] == "adjacent"]
    primary_quota = round(max_candidates * primary_allocation_pct)
    adjacent_quota = max_candidates - primary_quota
    logger.info(f"Primary bucket — {len(primary_candidates)} candidates")
    logger.info(f"Adjacent bucket — {len(adjacent_candidates)} candidates")
    logger.info(
        f"After quota allocation — primary: {min(len(primary_candidates), primary_quota)}/{primary_quota}, "
        f"adjacent: {min(len(adjacent_candidates), adjacent_quota)}/{adjacent_quota}"
    )
    return primary_candidates[:primary_quota] + adjacent_candidates[:adjacent_quota]


def _sics_for_seed_tickers(seed_tickers: list[str]) -> list[str]:
    sics = []
    for ticker in seed_tickers:
        try:
            sic = sic_universe_builder.fetch_sic_for_ticker(ticker)
        except Exception as e:
            logger.warning(f"{ticker} — failed to resolve seed ticker SIC: {e}")
            continue
        if sic:
            sics.append(sic)
        else:
            logger.warning(f"{ticker} — seed ticker has no SEC SIC code; skipping SIC expansion")
    return _dedup(sics)


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


def _merge_records(
    primary_records: list[dict],
    adjacent_records: list[dict],
    sic_clusters: dict[str, str],
    embedding_records: list[dict] | None = None,
) -> list[dict]:
    merged: dict[str, dict] = {}
    for record in primary_records + adjacent_records + (embedding_records or []):
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
            existing["candidate_source"] = "sec_sic"
        elif existing["adjacent_matched_sic_codes"]:
            existing["source_bucket"] = "adjacent"
            existing["candidate_source"] = "sec_sic"
        elif record.get("source_bucket") == "embedding":
            existing["source_bucket"] = "embedding"
            existing["candidate_source"] = "embedding"
        if record.get("embedding_similarity") is not None:
            existing["embedding_similarity"] = record["embedding_similarity"]
        if existing["industry_cluster"] is None:
            existing["industry_cluster"] = _cluster_for_sics(existing["matched_sic_codes"], sic_clusters)

    return list(merged.values())


def _preflight_sic_codes(sic_codes: list[str], allow_broad: bool) -> None:
    """
    Fail fast on SIC codes that can't produce a sane universe, before the
    per-CIK ticker lookups start (see sic_universe_builder.preflight_sic_codes
    for cost). Both failure modes were hit in validation and previously
    surfaced only mid-run: a code with zero SEC filers silently contributed
    nothing, and an over-broad code triggered hundreds of lookups for
    companies unrelated to the target.
    """
    counts = sic_universe_builder.preflight_sic_codes(sic_codes)
    zero = sorted(sic for sic, n in counts.items() if n == 0)
    broad = {sic: n for sic, n in counts.items() if n > sic_universe_builder.BROAD_SIC_CIK_THRESHOLD}

    if zero:
        raise ValueError(
            f"SIC preflight: no SEC filers with a usable US ticker for SIC code(s) {', '.join(zero)}. "
            "Remove them from primary/adjacent_sic_codes (verify the code against the SEC SIC list, "
            "or pick a nearby code that public companies actually file under)."
        )
    if broad and not allow_broad:
        detail = ", ".join(f"{sic} ({n} filers)" for sic, n in sorted(broad.items()))
        raise ValueError(
            f"SIC preflight: {detail} exceed(s) the {sic_universe_builder.BROAD_SIC_CIK_THRESHOLD}-filer "
            "threshold — such broad codes mostly fetch companies unrelated to the target and burn the SEC "
            "request budget. Replace with a narrower code, or set universe.allow_broad_sic_codes: true to proceed."
        )


def _config_with_suggested_sics(config: PipelineConfig) -> PipelineConfig:
    suggestion_raw = config.model_dump()
    suggestion_raw["llm"]["max_tokens"] = max(
        suggestion_raw["llm"]["max_tokens"], llm_analyzer.SIC_SUGGESTION_MIN_MAX_TOKENS,
    )
    suggestion_config = PipelineConfig.model_validate(suggestion_raw)
    suggestions = llm_analyzer.suggest_sic_codes(suggestion_config)
    configured_primary = list(config.target_company.primary_sic_codes)
    configured_adjacent = list(config.target_company.adjacent_sic_codes)
    suggested_primary: list[str] = []
    suggested_adjacent: list[str] = []
    for suggestion in suggestions:
        raw_code = str(suggestion.get("sic_code") or "").strip()
        if not raw_code:
            continue
        code = raw_code.zfill(4)
        if code in configured_primary or code in configured_adjacent:
            continue
        bucket = str(suggestion.get("bucket") or "").strip().lower()
        (suggested_primary if bucket == "primary" else suggested_adjacent).append(code)
    suggested_primary = _dedup(suggested_primary)
    suggested_adjacent = [code for code in _dedup(suggested_adjacent) if code not in suggested_primary]

    suggested = suggested_primary + suggested_adjacent
    if suggested:
        try:
            counts = sic_universe_builder.preflight_sic_codes(suggested)
        except Exception as exc:
            logger.warning(f"Suggested-SIC preflight failed ({exc}); using SEC-validated suggestions")
            counts = {}
        zero_yield = {code for code, count in counts.items() if count == 0}
        suggested_primary = [code for code in suggested_primary if code not in zero_yield]
        suggested_adjacent = [code for code in suggested_adjacent if code not in zero_yield]
        if zero_yield:
            logger.info(f"Dropping zero-yield suggested SIC codes: {sorted(zero_yield)}")

    raw = config.model_dump()
    raw["target_company"]["primary_sic_codes"] = _dedup(configured_primary + suggested_primary)
    raw["target_company"]["adjacent_sic_codes"] = [
        code for code in _dedup(configured_adjacent + suggested_adjacent)
        if code not in raw["target_company"]["primary_sic_codes"]
    ]
    return PipelineConfig.model_validate(raw)


def last_discovery_snapshot() -> dict:
    return dict(_last_discovery_snapshot)


def build(config: PipelineConfig | dict) -> list[dict]:
    global _last_discovery_snapshot
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
    cfg = as_config(config)
    if cfg.universe.discovery_mode == "suggest-sic+embedding":
        cfg = _config_with_suggested_sics(cfg)
    target = cfg.target_company
    primary_sics = target.primary_sic_codes
    universe_cfg = cfg.universe
    seed_sics = _sics_for_seed_tickers(universe_cfg.seed_tickers)
    adjacent_sics = _dedup(target.adjacent_sic_codes + seed_sics)

    _preflight_sic_codes(primary_sics + adjacent_sics, universe_cfg.allow_broad_sic_codes)

    sic_clusters = universe_cfg.sic_clusters
    primary_records = _records_for_bucket(primary_sics, "primary", sic_clusters)
    adjacent_records = _records_for_bucket(adjacent_sics, "adjacent", sic_clusters)
    embedding_records = (
        embedding_universe_builder.discover(cfg)
        if universe_cfg.discovery_mode in {"sic+embedding", "suggest-sic+embedding"}
        else []
    )
    merged_records = _merge_records(primary_records, adjacent_records, sic_clusters)

    logger.info(f"Primary SIC codes: {primary_sics}")
    logger.info(f"Adjacent SIC codes: {adjacent_sics}")
    if embedding_records:
        logger.info(f"Embedding discovery candidates: {len(embedding_records)}")
    quota_records = _apply_bucket_quotas(
        merged_records, universe_cfg.max_candidates, universe_cfg.primary_allocation_pct,
    )
    if embedding_records:
        quota_records = _merge_records(quota_records, [], sic_clusters, embedding_records)

    result = _apply_analyst_ticker_overrides(
        quota_records,
        _dedup(universe_cfg.must_include_tickers + universe_cfg.seed_tickers),
        universe_cfg.exclude_tickers,
    )
    _last_discovery_snapshot = {
        "discovery_mode": universe_cfg.discovery_mode,
        "sic_filer_tickers": {record["ticker"] for record in merged_records},
        "candidate_tickers": {record["ticker"] for record in result},
        # Post-expansion codes: in suggest-sic+embedding mode the suggested
        # SICs only exist on the local cfg copy, so callers auditing what
        # discovery actually searched must read them from here.
        "primary_sic_codes": list(primary_sics),
        "adjacent_sic_codes": list(adjacent_sics),
    }
    return result
