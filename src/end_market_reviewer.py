"""LLM end-market review for Core-tier eligibility.

Embedding similarity between one-sentence sub-sector descriptions is the
Core tier's end-market gate, and it fails in both directions around the
threshold: generic capital-equipment phrasing can score above it for a
company serving an unrelated end market (a false Core), while a genuinely
aligned comp can land just below it on embedding noise (a false block).

This module runs one batched LLM check over the borderline candidates of a
finished Top-N — the would-be Core rows (veto direction) and the rows blocked
only by a marginal similarity shortfall (rescue direction) — and returns a
per-ticker aligned/not-aligned verdict with a one-sentence reason. The caller
(reporter._apply_end_market_review) adjusts tiers; this module only judges.

An API failure returns None and the caller changes nothing: like the
embedding-similarity fallback, an outage degrades to the coarser signals
instead of blocking report generation.
"""
from src import get_logger, llm_analyzer
from src.config_schema import PipelineConfig
from src.llm_schemas import EndMarketVerdicts

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a private equity analyst verifying end-market alignment between a
target company and candidate comparable companies.

For each candidate, decide whether it serves substantially the same end
markets as the target — the customer industries its products or services are
ultimately sold into — and give a one-sentence reason.

Rules:
- Judge end markets specifically, not business-model resemblance. Generic
  phrasing like "highly engineered products" or "capital equipment" is not
  end-market evidence: an oil-and-gas equipment maker and an automotive
  parts maker are NOT aligned however similar their language sounds.
- aligned=true only when the candidate's primary end markets substantially
  overlap the target's. Partial overlap in a secondary market is not enough.
- Base answers only on the descriptions provided.
- Return a verdict for every candidate ticker listed, exactly once."""

USER_PROMPT_TEMPLATE = """Target end-market profile: {target_description}

Candidates:
{candidate_lines}
"""


def review_end_markets(
    target_description: str,
    candidates: dict[str, str],
    config: PipelineConfig,
) -> dict[str, dict] | None:
    """{ticker: sub_sector_description} -> {ticker: {aligned, reason}}, or
    None when the call fails (caller must treat that as "no opinion")."""
    if not target_description or not candidates:
        return None

    candidate_lines = "\n".join(f"- {ticker}: {desc}" for ticker, desc in sorted(candidates.items()))
    prompt = USER_PROMPT_TEMPLATE.format(
        target_description=target_description, candidate_lines=candidate_lines,
    )

    try:
        import openai
        client = openai.OpenAI()
        result = llm_analyzer._call_openai_structured(
            client, config.llm.judge_model, SYSTEM_PROMPT, prompt,
            config.llm.temperature, config.llm.max_tokens, EndMarketVerdicts,
        )
    except Exception as e:
        logger.warning(f"End-market review API call failed: {e} — Core tier falls back to similarity threshold only")
        return None

    verdicts = {
        v.ticker: {"aligned": v.aligned, "reason": v.reason}
        for v in result.verdicts
        if v.ticker in candidates
    }
    missing = set(candidates) - set(verdicts)
    if missing:
        logger.warning(f"End-market review returned no verdict for {', '.join(sorted(missing))} — leaving unchanged")
    return verdicts
