from src import embedding_store, fetcher, get_logger, llm_analyzer, sic_codes, sic_universe_builder
from src.config_schema import PipelineConfig, as_config

logger = get_logger(__name__)
_last_discovery_trace: dict[str, str] = {}
_last_candidate_prefixes: set[str] = set()
_last_candidate_limit_reached = False


def _sic_prefixes(sics: list[str]) -> list[str]:
    prefixes = []
    for sic in sics:
        clean = str(sic or "").strip()
        if len(clean) >= 2:
            prefixes.append(clean[:2])
    return list(dict.fromkeys(prefixes))


def _expanded_sics(seed_sics: list[str]) -> list[str]:
    prefixes = _sic_prefixes(seed_sics)
    if not prefixes:
        return []
    codes = []
    for code in sic_codes.official_sic_codes():
        padded = code.zfill(4)
        if any(padded.startswith(prefix) for prefix in prefixes):
            codes.append(code)
    return codes


def _filter_broad_sics(sics: list[str]) -> list[str]:
    try:
        counts = sic_universe_builder.preflight_sic_codes(sics)
    except Exception as e:
        logger.warning(f"Embedding discovery SIC preflight failed; using unfiltered expanded SIC list: {e}")
        return sics
    broad = {
        sic: count for sic, count in counts.items()
        if count > sic_universe_builder.BROAD_SIC_CIK_THRESHOLD
    }
    if broad:
        detail = ", ".join(f"{sic} ({count})" for sic, count in sorted(broad.items()))
        logger.warning(f"Embedding discovery skipping broad expanded SIC code(s): {detail}")
    return [sic for sic in sics if sic not in broad]


def _candidate_sics(seed_sics: list[str]) -> list[str]:
    clean_seed_sics = [str(sic).strip().zfill(4) for sic in seed_sics if str(sic or "").strip()]
    return list(dict.fromkeys(clean_seed_sics + _expanded_sics(seed_sics)))


def _round_robin(sics: list[str], tickers_by_sic: dict[str, list[str]], seen: set[str]) -> list[str]:
    """One ticker per SIC per round, so no single early/large SIC can consume
    the whole candidate budget — the corpus stays a representative sample of
    every code in the tier."""
    queues = [list(tickers_by_sic.get(sic, ())) for sic in sics]
    picked: list[str] = []
    while any(queues):
        for queue in queues:
            while queue:
                ticker = queue.pop(0)
                if ticker not in seen:
                    seen.add(ticker)
                    picked.append(ticker)
                    break
    return picked


def _candidate_tickers(seed_sics: list[str], limit: int | None = None) -> tuple[list[str], bool]:
    """Stratified candidate enumeration: the seed SICs themselves (including
    suggested codes in hybrid mode) are sampled round-robin first, then the
    same-2-digit-family expansion codes fill the remaining budget, also
    round-robin. Replaces sequential enumeration, whose truncation at the
    budget was dominated by whichever SIC happened to enumerate first."""
    seed_set = {str(sic).strip().zfill(4) for sic in seed_sics if str(sic or "").strip()}
    all_sics = _filter_broad_sics(_candidate_sics(seed_sics))
    seed_tier = [sic for sic in all_sics if sic in seed_set]
    expansion_tier = [sic for sic in all_sics if sic not in seed_set]

    tickers_by_sic: dict[str, list[str]] = {}
    for sic in seed_tier + expansion_tier:
        try:
            tickers_by_sic[sic] = list(sic_universe_builder.discover_tickers_by_sic([sic]))
        except Exception as e:
            logger.warning(f"SIC {sic} — embedding discovery ticker enumeration failed: {e}")

    seen: set[str] = set()
    ordered = _round_robin(seed_tier, tickers_by_sic, seen) + _round_robin(expansion_tier, tickers_by_sic, seen)
    if limit and len(ordered) > limit:
        logger.info(
            f"Embedding discovery candidate ticker limit reached: {limit} "
            f"(round-robin across {len(tickers_by_sic)} SIC codes, {len(ordered)} enumerable)"
        )
        return ordered[:limit], True
    return ordered, False


def _description_for_ticker(ticker: str) -> tuple[str | None, dict]:
    cached = fetcher._load_cache(ticker)
    if isinstance(cached, dict) and cached.get("business_description"):
        return cached["business_description"], {
            "source_accession": cached.get("source_accession"),
            "source_date": cached.get("fetch_timestamp"),
            "description_source": cached.get("description_source"),
        }
    try:
        description, source = fetcher._fetch_business_description(ticker)
    except Exception as e:
        logger.warning(f"{ticker} — embedding discovery description fetch failed: {e}")
        return None, {}
    return description, {"description_source": source}


def _populate_store(tickers: list[str], embedding_model: str, trace: dict[str, str]) -> None:
    store = embedding_store.load()
    missing: list[tuple[str, str, dict]] = []
    for ticker in tickers:
        description, source_metadata = _description_for_ticker(ticker)
        if not description:
            trace[ticker] = "description_fetch_failed"
            continue
        if embedding_store.is_fresh(store.get(ticker.upper()), description, embedding_model):
            trace[ticker] = "embedded"
            continue
        trace[ticker] = "stale_vector"
        missing.append((ticker, description, source_metadata))

    if not missing:
        return

    vectors = llm_analyzer.embed_texts([description for _, description, _ in missing], model=embedding_model)
    if vectors is None:
        logger.warning("Embedding discovery corpus embedding failed; semantic candidates unavailable for this run")
        for ticker, _, _ in missing:
            trace[ticker] = "corpus_embedding_failed"
        return

    records = []
    for index, (ticker, description, metadata) in enumerate(missing):
        if index >= len(vectors) or not isinstance(vectors[index], list):
            trace[ticker] = "corpus_embedding_failed"
            continue
        records.append(embedding_store.make_record(ticker, description, vectors[index], embedding_model, **metadata))
        trace[ticker] = "embedded"
    if records:
        embedding_store.upsert_many(records, data=store)


def last_discovery_trace() -> dict[str, str]:
    return dict(_last_discovery_trace)


def unenumerated_stage(ticker: str) -> str:
    if _last_candidate_limit_reached:
        sic = sic_universe_builder.fetch_sic_for_ticker(ticker)
        if sic and str(sic).zfill(4)[:2] in _last_candidate_prefixes:
            return "truncated_by_embedding_candidate_limit"
    return "outside_expanded_taxonomy"


def discover(config: PipelineConfig | dict) -> list[dict]:
    global _last_candidate_limit_reached, _last_candidate_prefixes, _last_discovery_trace
    _last_discovery_trace = {}
    cfg = as_config(config)
    universe_cfg = cfg.universe
    seed_sics = list(cfg.target_company.primary_sic_codes) + list(cfg.target_company.adjacent_sic_codes)
    _last_candidate_prefixes = set(_sic_prefixes(seed_sics))
    tickers, _last_candidate_limit_reached = _candidate_tickers(seed_sics, universe_cfg.embedding_candidate_limit)
    if not tickers:
        return []

    _populate_store(tickers, cfg.llm.embedding_model, _last_discovery_trace)
    target_vectors = llm_analyzer.embed_texts([cfg.target_company.description], model=cfg.llm.embedding_model)
    if not target_vectors:
        return []

    rows = []
    candidate_set = set(tickers)
    matches = embedding_store.query(
        target_vectors[0],
        universe_cfg.embedding_top_n,
        allowed_tickers=candidate_set,
        embedding_model=cfg.llm.embedding_model,
    )
    matched_tickers = {ticker for ticker, _ in matches}
    for ticker in candidate_set:
        if _last_discovery_trace.get(ticker) == "embedded" and ticker not in matched_tickers:
            _last_discovery_trace[ticker] = "outside_candidate_set_top_n"
    for ticker, similarity in matches:
        if ticker not in candidate_set:
            continue
        if similarity < universe_cfg.embedding_min_similarity:
            _last_discovery_trace[ticker] = "below_similarity_threshold"
            continue
        _last_discovery_trace[ticker] = "recalled"
        rows.append({
            "ticker": ticker,
            "matched_sic_codes": [],
            "primary_matched_sic_codes": [],
            "adjacent_matched_sic_codes": [],
            "source_bucket": "embedding",
            "sic_2_digit": None,
            "sic_3_digit": None,
            "industry_cluster": None,
            "candidate_source": "embedding",
            "embedding_similarity": similarity,
        })
    return rows
