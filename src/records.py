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

NOTE: no type checker runs in CI yet (only ruff, which does not type-check), so
this currently serves as documentation + opt-in IDE/static checking. Wiring up
mypy to give it teeth is a deliberate follow-up — it will surface a batch of
pre-existing typing issues worth handling on their own.
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
    universe_metadata: dict
    source_bucket: str | None
    sic_2_digit: str | None
    sic_3_digit: str | None
    industry_cluster: str | None

    # Bookkeeping. raw_fmp_profile is FMP's full /profile payload (the universe
    # domicile filter reads its 'country'); missing_flags marks which fields
    # came back empty; fetch_timestamp is when this record was built (UTC ISO).
    raw_fmp_profile: dict | None
    missing_flags: dict[str, bool]
    fetch_timestamp: str
