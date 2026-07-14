"""
Shared shape of the per-company "record" that flows through the pipeline.

This is the single authoritative description of the ~40-field dict that fetcher
produces, caches to data/cache/{ticker}.json, and every downstream stage
(feature_builder, scorer, reporter, the universe filters, eval) reads. It is a
TypedDict, not a pydantic model: the record is JSON-serialized at rest and
accessed with plain dict / .get() syntax in ~150 places, so a TypedDict
documents the contract and enables static type-checking WITHOUT changing any
runtime behavior — at runtime a CompanyRecord is exactly a dict.

total=False: the record is built up incrementally (fetcher creates the
financial/EV core; _attach_universe_metadata adds the SIC fields; FMP
enrichment fills market data later), and consumers read it defensively with
.get(), so every key is treated as optional here rather than asserting a
field is always present at a given stage.

Mypy runs a small, explicit contract-checking scope in CI. Keep widening that
scope as dynamic dict consumers are typed.
"""
from typing import TypedDict


class CompanyRecord(TypedDict, total=False):
    # Identity
    ticker: str
    company_name: str | None

    # The six standardized financial features scorer ranks on (revenue is
    # log-transformed downstream in feature_builder; the rest are used as-is).
    market_cap_usd_mm: float | None
    revenue_ttm_usd_mm: float | None
    ebitda_margin: float | None
    gross_margin: float | None
    revenue_cagr_3yr: float | None
    net_debt_ebitda: float | None
    capex_revenue: float | None

    # Enterprise-value bridge components (assembled into EV in
    # fetcher._ev_from_bridge). Absent components are treated as 0 there.
    net_debt_usd_mm: float | None
    minority_interest_usd_mm: float | None
    preferred_equity_usd_mm: float | None
    operating_lease_liability_usd_mm: float | None
    enterprise_value_usd_mm: float | None

    # Absolute-dollar figures behind the ratios above (used by reporter for
    # the additional PE metrics and the scale-reconciliation note).
    ebitda_usd_mm: float | None
    ebit_usd_mm: float | None
    net_income_usd_mm: float | None
    gross_profit_usd_mm: float | None
    free_cash_flow_usd_mm: float | None

    # Additional PE metrics derived in fetcher (see _enrich_with_fmp_data).
    fcf_conversion: float | None
    interest_coverage: float | None
    debt_to_equity: float | None
    ev_ebitda: float | None
    ev_revenue: float | None
    ev_ebit: float | None
    ev_gross_profit: float | None
    pe_ratio: float | None
    fcf_yield: float | None

    # Valuation provenance — how ev_ebitda / ev_revenue were sourced.
    valuation_source: str | None
    ev_ebitda_source_value: float | None
    ev_revenue_source_value: float | None

    # Business-model inputs (the description feeds the LLM extractor; sector is
    # FMP's). LLM-extracted attributes live in a separate llm_features dict
    # keyed by ticker, NOT on this record.
    gics_sector: str | None
    business_description: str | None
    description_source: str | None

    # Provenance carried down from universe_builder's candidate record. The
    # nested map plus the flattened SIC fields reporter/feature_builder read.
    universe_metadata: dict[str, object]
    source_bucket: str | None
    sic_2_digit: str | None
    sic_3_digit: str | None
    industry_cluster: str | None

    # Bookkeeping. raw_fmp_profile is FMP's full /profile payload (the universe
    # domicile filter reads its 'country'); missing_flags marks which fields
    # came back empty. Two timestamps drive layered cache freshness (fetcher):
    # fetch_timestamp is when the slow EDGAR fundamentals were built;
    # market_data_timestamp is when the fast-moving FMP market layer (market cap
    # and the EV multiples derived from it) was last refreshed.
    raw_fmp_profile: dict[str, object] | None
    missing_flags: dict[str, bool]
    fetch_timestamp: str
    market_data_timestamp: str | None


class LLMFeatureRecord(TypedDict, total=False):
    """Authoritative shape of llm_features[ticker].

    The extractor, feature builder, scorer, reporter, and eval harness all read
    this JSON-serializable dict. Keeping the field list here makes additions
    explicit even before every dynamic consumer is fully typed.
    """

    business_model: str | None
    revenue_recurrence: str | None
    customer_type: str | None
    capital_intensity: str | None
    primary_value_driver: str | None
    sub_sector_description: str | None
    evidence_quote: str | None
    confidence: int | None
    evidence_verified: bool
    judge_score: int | None
    judge_reason: str | None
    low_confidence_flag: bool
    profile_incomplete: bool
    extraction_failed: bool
    description_sha256: str | None
    extraction_model: str | None
