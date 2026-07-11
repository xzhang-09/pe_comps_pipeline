import json

import openai

from src import get_logger
from src.llm_analyzer import _call_openai_structured
from src.llm_schemas import RerankResult

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a private equity analyst re-ranking a short list of
public-company comparables. Use only the provided candidate cards. You may
change order, but you must keep exactly the same tickers."""

PROMPT_TEMPLATE = """Re-rank these comparable-company candidates for fit.

Target:
{target_json}

Candidates, already sorted by deterministic financial/business fit:
{candidates_json}
"""


def _candidate_card(row: dict, llm_features: dict, companies_by_ticker: dict) -> dict:
    ticker = row["ticker"]
    company = companies_by_ticker.get(ticker, {})
    llm = llm_features.get(ticker, {})
    return {
        "ticker": ticker,
        "company_name": company.get("company_name", ticker),
        "current_rank": row.get("base_rank"),
        "adjusted_score": row.get("adjusted_score"),
        "business_model": llm.get("business_model"),
        "customer_type": llm.get("customer_type"),
        "capital_intensity": llm.get("capital_intensity"),
        "sub_sector_description": llm.get("sub_sector_description"),
        "revenue_ttm_usd_mm": company.get("revenue_ttm_usd_mm"),
        "ebitda_margin": company.get("ebitda_margin"),
        "penalty_reasons": row.get("reasons", []),
    }


def rerank(
    *,
    ranked: list[dict],
    target_profile: dict,
    llm_features: dict,
    companies_by_ticker: dict,
    model: str,
    temperature: float,
    max_tokens: int,
    rerank_window: int,
) -> tuple[list[str], list[dict]] | None:
    window = ranked[:rerank_window]
    input_tickers = [row["ticker"] for row in window]
    if len(input_tickers) < 2:
        return None

    prompt = PROMPT_TEMPLATE.format(
        target_json=json.dumps(target_profile, indent=2, default=str),
        candidates_json=json.dumps(
            [_candidate_card(row, llm_features, companies_by_ticker) for row in window],
            indent=2,
            default=str,
        ),
    )
    try:
        result = _call_openai_structured(
            openai.OpenAI(), model, SYSTEM_PROMPT, prompt, temperature, max_tokens, RerankResult,
        )
    except Exception as e:
        logger.warning(f"LLM re-ranker failed; using deterministic ranking: {e}")
        return None

    parsed = result.model_dump()
    ordered = parsed["ordered_tickers"]
    if sorted(ordered) != sorted(input_tickers) or len(ordered) != len(set(ordered)):
        logger.warning("LLM re-ranker returned an invalid ticker permutation; using deterministic ranking")
        return None
    return ordered, parsed.get("moves", [])
