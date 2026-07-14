from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BusinessModelExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_model: Literal["manufacturing", "services", "SaaS", "distribution", "marketplace", "other"] | None
    revenue_recurrence: Literal["high", "medium", "low"] | None
    customer_type: Literal["B2B", "B2C", "B2G", "mixed"] | None
    capital_intensity: Literal["asset_heavy", "moderate", "asset_light"] | None
    primary_value_driver: Literal["technology", "scale", "relationships", "brand", "other"] | None
    sub_sector_description: str | None
    evidence_quote: str | None
    confidence: int | None = Field(default=None, ge=1, le=5)


class CoreProfileFollowUp(BaseModel):
    """Targeted re-ask for core profile fields the first extraction left null.

    Every field carries an explicit "unknown" option so the model can decline
    per field instead of being forced to guess; "unknown" answers are kept as
    null downstream. business_model is deliberately absent — it carries the
    verbatim evidence-quote contract, which a field-only follow-up can't honor.
    """

    model_config = ConfigDict(extra="forbid")

    customer_type: Literal["B2B", "B2C", "B2G", "mixed", "unknown"]
    capital_intensity: Literal["asset_heavy", "moderate", "asset_light", "unknown"]
    primary_value_driver: Literal["technology", "scale", "relationships", "brand", "other", "unknown"]
    sub_sector_description: str | None


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=1, le=5)
    reason: str


class EndMarketVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    aligned: bool
    reason: str


class EndMarketVerdicts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[EndMarketVerdict]


class SicSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sic_code: str
    title: str
    bucket: Literal["primary", "adjacent"]
    reason: str
    confidence: Literal["high", "medium", "low"]


class SicSuggestions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suggestions: list[SicSuggestion]


class CompFitItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    score: int = Field(ge=0, le=100)
    reason: str


class CompFitReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_score: int = Field(ge=0, le=100)
    review_confidence: Literal["low", "medium", "high"]
    summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    top_fits: list[CompFitItem] = Field(default_factory=list)
    questionable_fits: list[CompFitItem] = Field(default_factory=list)
    near_miss_upgrades: list[CompFitItem] = Field(default_factory=list)


class RerankMove(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    direction: Literal["up", "down", "unchanged"]
    reason: str


class RerankResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ordered_tickers: list[str]
    moves: list[RerankMove] = Field(default_factory=list)
