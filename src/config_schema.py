from pydantic import BaseModel, ConfigDict, Field


class TargetCompanyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    primary_sic_codes: list[str]
    adjacent_sic_codes: list[str] = Field(default_factory=list)
    # Analyst-provided estimates for the private target's financial features.
    # Each one the analyst supplies is used directly as the target's value for
    # that feature in scorer's distance-to-target; each left unset (None) falls
    # back to a peer-group median and is *excluded* from the distance entirely
    # (see scorer._target_financial_row / _distance_to_target) so an imputed
    # feature can't masquerade as a real signal and pull the ranking toward
    # whatever sits near the pool median on that axis. revenue_usd_mm and
    # ebitda_margin_estimate are the two most analysts have on hand; the other
    # four (gross margin, 3yr revenue CAGR, net debt / EBITDA, capex / revenue)
    # are optional and usually available from a CIM or management accounts.
    revenue_usd_mm: float | None = None
    ebitda_margin_estimate: float | None = None
    gross_margin_estimate: float | None = None
    revenue_cagr_3yr_estimate: float | None = None
    net_debt_ebitda_estimate: float | None = None
    capex_revenue_estimate: float | None = None
    # Reference only; not read by any pipeline logic (see README).
    gics_sector: str | None = None
    gics_industry: str | None = None
    geography: str | None = None


class UniverseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_candidates: int
    primary_allocation_pct: float = 0.5
    # Optional filtering fields; pipeline code does not enforce them.
    min_revenue_usd_mm: float | None = None
    max_revenue_usd_mm: float | None = None
    min_ebitda_margin: float | None = None
    sic_clusters: dict[str, str] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extraction_model: str
    judge_model: str
    temperature: float
    max_tokens: int
    batch_size: int
    judge_threshold: int
    # Used for the sub-sector mismatch penalty (see reporter.py /
    # ScorerConfig.ranking_penalties below) — a separate model family from
    # extraction_model/judge_model since OpenAI's chat models do not expose
    # embeddings directly.
    embedding_model: str = "text-embedding-3-small"


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Both read by reporter.py. top_n_comps defaults to 15 (reporter.TOP_N)
    # when unset/0; report_formats defaults to ("csv", "html") when empty,
    # and raises ValueError for any format other than "csv"/"html".
    top_n_comps: int | None = None
    report_formats: list[str] = Field(default_factory=list)
    # Both read by reporter.py for the report footer/confidentiality banner.
    # prepared_by is omitted from the report entirely when unset; confidential
    # defaults to False so existing configs render exactly as before.
    prepared_by: str | None = None
    confidential: bool = False


class RankingPenaltiesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Soft penalties added to a candidate's financial-distance score (not
    # exclusions) in reporter.py's Top-N selection — see
    # reporter._penalty_breakdown(). These are in the SAME units as residual_abs
    # (the standardized financial-feature distance to the target), because
    # they're added to that distance, not to an ordinal rank. residual_abs
    # typically spans ~0.3-2.0 across a comp pool, so a penalty of ~0.4-0.6 is a
    # "moderate financial gap" worth of demotion — enough to drop a categorically
    # mismatched comp below a comparably-close peer, without letting one flag
    # leapfrog a much closer comp the way the old rank-unit penalties (~10-15)
    # did. eval/evaluator.py's _select_top_k reads the same values (passed
    # through from this config) so the evaluation harness and the production
    # report can't drift apart the way two independently hardcoded copies could.
    business_model_penalty: float = 0.6
    customer_type_penalty: float = 0.5
    # Below this cosine similarity between the target's and a candidate's
    # sub_sector_description, the sub-sector mismatch penalty applies.
    # Uncalibrated judgment call — no labeled "good"/"bad" pairs exist yet
    # to tune it against; revisit if it's excluding/including obviously
    # wrong companies in practice.
    subsector_similarity_threshold: float = 0.5
    subsector_mismatch_penalty: float = 0.4
    # Companies within 10x of the target's revenue (either direction) get
    # no size penalty; each further order of magnitude adds
    # size_penalty_per_extra_log10 (in residual-distance units).
    size_penalty_free_log10_range: float = 1.0
    size_penalty_per_extra_log10: float = 1.0


class ScorerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # {business_model: {financial_feature: weight}} — looked up by the
    # *target's* business_model (see scorer.py) so every candidate is
    # measured against the same ruler. Features omitted from a template
    # default to weight 1.0; a business_model with no template, or a null
    # business_model, falls back to the "default" template (or 1.0
    # everywhere if "default" itself isn't defined). Leaving this empty uses
    # unweighted Euclidean distance.
    feature_weights: dict[str, dict[str, float]] = Field(default_factory=dict)
    ranking_penalties: RankingPenaltiesConfig = Field(default_factory=RankingPenaltiesConfig)


class ValuationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Public trading comps are large-cap, liquid, minority-interest
    # multiples. Applying them straight to a small, private, mid-market
    # target overstates value: the dominant real-world effects at this size
    # are an illiquidity / size (marketability) discount, which a control
    # premium only partially offsets. This single knob is the *net* haircut
    # an analyst wants to apply to the comp-derived implied EV — e.g. 0.25
    # means "haircut the comp-implied EV by 25%". Default 0.0 is a no-op so
    # existing configs and reports render exactly as before; set it in
    # config.yaml to surface a private-company-adjusted range alongside the
    # raw comp range. Applied to the implied EV range as a whole (a
    # simplification — it does not re-derive an equity bridge).
    size_marketability_discount: float = Field(default=0.0, ge=0.0, lt=1.0)
    # Free-text rationale shown next to the adjusted range so the haircut is
    # never an unexplained number (e.g. "20% DLOM + size discount, net of an
    # assumed control premium"). Omitted from the report when unset.
    discount_note: str | None = None
    # Capitalize operating-lease liabilities into enterprise value (see
    # fetcher._include_operating_leases). Default False keeps EV/EBITDA on a
    # pre-lease-capitalization basis, internally consistent with EBITDA being
    # net of operating-lease cost under ASC 842. Set True for lease-heavy
    # targets where you want the lease-capitalized view — but note the
    # resulting EV/EBITDA is then on a different basis than the unadjusted one
    # (a true like-for-like would compare EV-incl-leases to EBITDAR).
    include_operating_leases_in_ev: bool = False


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_company: TargetCompanyConfig
    universe: UniverseConfig
    llm: LLMConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    scorer: ScorerConfig = Field(default_factory=ScorerConfig)
    valuation: ValuationConfig = Field(default_factory=ValuationConfig)
