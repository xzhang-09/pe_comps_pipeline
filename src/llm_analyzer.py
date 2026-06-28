import json
import re
from pathlib import Path

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from src import get_logger
from src.config_schema import PipelineConfig, as_config

logger = get_logger(__name__)

CHECKPOINT_PATH = Path("data/checkpoints/llm_checkpoint.json")

MIN_DESCRIPTION_LENGTH = 100

SYSTEM_PROMPT = """You are a private equity analyst extracting structured information from
company business descriptions to support comparable company analysis.

Rules:
- Output ONLY valid JSON. No explanation, no markdown fences, no preamble.
- Use null for any field where the description provides insufficient
  information to make a reliable determination.
- Base ALL answers strictly on the provided text.
  Do not use any outside knowledge about the company.
- confidence: integer 1-5 reflecting how explicitly the description
  supports your extraction (5 = very explicit).
- evidence_quote: if business_model is non-null, copy a short passage
  (a few words to one sentence) VERBATIM, character-for-character, from
  the description above that directly supports your business_model
  classification. Do not paraphrase or summarize. If no such passage
  exists, set business_model to null instead of guessing."""

USER_PROMPT_TEMPLATE = """Extract structured fields from this company business description.

Company: {company_name}
Description: {business_description}

Return JSON with exactly these fields:
{{
  "business_model": "manufacturing"|"services"|"SaaS"|"distribution"|"marketplace"|"other",
  "revenue_recurrence": "high"|"medium"|"low",
  "customer_type": "B2B"|"B2C"|"B2G"|"mixed",
  "capital_intensity": "asset_heavy"|"moderate"|"asset_light",
  "primary_value_driver": "technology"|"scale"|"relationships"|"brand"|"other",
  "sub_sector_description": "<one sentence describing specific sub-sector>",
  "evidence_quote": "<verbatim passage from the description supporting business_model, or null>",
  "confidence": <integer 1-5>
}}"""

JUDGE_PROMPT_TEMPLATE = """You are reviewing a business model extraction for accuracy.

Original description (first 500 chars): {description_excerpt}
Extraction result: {extraction_json}

Rate the accuracy of the extraction on a scale of 1-5:
5 = All fields are clearly supported by the text
4 = Most fields accurate, minor uncertainty on 1-2 fields
3 = Partially accurate, some fields unclear or questionable
2 = Several fields appear inaccurate or unsupported
1 = Extraction is largely inaccurate or hallucinated

Return ONLY this JSON (no other text):
{{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

EXTRACTION_FIELDS = (
    "business_model", "revenue_recurrence", "customer_type",
    "capital_intensity", "primary_value_driver", "sub_sector_description",
    "evidence_quote", "confidence",
)

SIC_SUGGESTION_SYSTEM_PROMPT = """You are an SEC filings analyst helping a private equity analyst find
plausible SIC (Standard Industrial Classification) codes for a target company,
to use as SEC EDGAR search filters for finding public comparable companies.

Rules:
- Output ONLY valid JSON. No explanation, no markdown fences, no preamble.
- SIC codes are a fixed, official 4-digit SEC list. You may misremember
  specific codes — this is a known limitation of language models, not a
  reflection of how well the company's industry is understood. Set
  "confidence" honestly: "high" only if you are confident this exact
  4-digit code is correct, "low" if you are recalling it from general
  knowledge and it should be verified before use.
- These are suggestions for a human analyst to verify against SEC's
  official SIC code list before using them — do not present them as
  certain."""

SIC_SUGGESTION_PROMPT_TEMPLATE = """Suggest 3-6 candidate SIC codes for SEC EDGAR comparable-company
discovery, based on this target company description.

Description: {description}

For each suggestion, classify it as "primary" (the company's actual
business) or "adjacent" (a related industry that could add useful
comparables but isn't the company's core business).

Return ONLY this JSON:
{{
  "suggestions": [
    {{
      "sic_code": "<4-digit code>",
      "title": "<standard SEC industry title for this code>",
      "bucket": "primary"|"adjacent",
      "reason": "<one sentence>",
      "confidence": "high"|"medium"|"low"
    }}
  ]
}}"""

SIC_SUGGESTION_FIELDS = ("sic_code", "title", "bucket", "reason", "confidence")

MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_matching(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip().lower()


def _evidence_quote_verified(evidence_quote: str | None, business_description: str) -> bool:
    """
    Programmatic check that evidence_quote is an actual verbatim substring
    of the source description (whitespace differences aside) — catches an
    LLM citing a passage that doesn't exist, which a same-pass confidence
    score wouldn't, since a hallucinated quote that "sounds right" doesn't
    lower the model's own self-reported confidence.
    """
    if not evidence_quote or not isinstance(evidence_quote, str):
        return False
    return _normalize_for_matching(evidence_quote) in _normalize_for_matching(business_description)


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    match = MARKDOWN_FENCE_RE.match(text)
    if match:
        return match.group(1).strip()
    return text


def _parse_json_response(text: str | None, ticker: str) -> dict | None:
    """Never raises — always returns a dict or None."""
    if not text:
        return None
    cleaned = _strip_markdown_fences(text)
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"{ticker} — failed to parse LLM JSON response: {e}")
        return None
    if not isinstance(parsed, dict):
        logger.warning(f"{ticker} — LLM JSON response was not an object: {cleaned!r}")
        return None
    return parsed


def _load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        return {}
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint(checkpoint: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def _log_rate_limit_retry(retry_state):
    logger.warning("Rate limit hit, waiting 60s")


@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_fixed(60),
    stop=stop_after_attempt(3),
    before_sleep=_log_rate_limit_retry,
    reraise=True,
)
def _call_openai(client, model: str, instructions: str | None, input_text: str,
                  temperature: float, max_tokens: int) -> str:
    kwargs = {
        "model": model,
        "input": input_text,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if instructions is not None:
        kwargs["instructions"] = instructions

    response = client.responses.create(**kwargs)
    return response.output_text


def _failed_result() -> dict:
    return {
        "business_model": None,
        "revenue_recurrence": None,
        "customer_type": None,
        "capital_intensity": None,
        "primary_value_driver": None,
        "sub_sector_description": None,
        "evidence_quote": None,
        "confidence": None,
        "evidence_verified": False,
        "judge_score": None,
        "judge_reason": None,
        "low_confidence_flag": False,
        "extraction_failed": True,
    }


def _extract_and_judge(client, ticker: str, company_name: str, business_description: str, config: PipelineConfig) -> dict:
    llm_config = config.llm

    user_prompt = USER_PROMPT_TEMPLATE.format(
        company_name=company_name, business_description=business_description,
    )

    try:
        extraction_text = _call_openai(
            client, llm_config.extraction_model, SYSTEM_PROMPT, user_prompt,
            llm_config.temperature, llm_config.max_tokens,
        )
    except Exception as e:
        logger.warning(f"{ticker} — extraction API call failed: {e}")
        return _failed_result()

    extraction = _parse_json_response(extraction_text, ticker)
    if extraction is None:
        return _failed_result()

    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        description_excerpt=business_description[:500],
        extraction_json=json.dumps(extraction),
    )

    judge_result = None
    try:
        judge_text = _call_openai(
            client, llm_config.judge_model, None, judge_prompt,
            llm_config.temperature, llm_config.max_tokens,
        )
        judge_result = _parse_json_response(judge_text, ticker)
    except Exception as e:
        logger.warning(f"{ticker} — judge API call failed: {e}")

    judge_score = judge_result.get("score") if judge_result else None
    judge_reason = judge_result.get("reason") if judge_result else None
    judge_threshold = llm_config.judge_threshold

    result = {field: extraction.get(field) for field in EXTRACTION_FIELDS}

    # Only require a verifiable citation when business_model was actually
    # extracted — a null business_model (model declined to guess) doesn't
    # need evidence behind it.
    evidence_verified = (
        result["business_model"] is None
        or _evidence_quote_verified(result.get("evidence_quote"), business_description)
    )
    if not evidence_verified:
        logger.warning(f"{ticker} — evidence_quote not found verbatim in source description")

    result["evidence_verified"] = evidence_verified
    result["judge_score"] = judge_score
    result["judge_reason"] = judge_reason
    result["low_confidence_flag"] = (
        not evidence_verified
        or (judge_score is not None and judge_score < judge_threshold)
    )
    result["extraction_failed"] = False
    return result


def analyze_batch(companies: list[dict], config: PipelineConfig | dict) -> dict[str, dict]:
    """
    Extract business model features for each company.

    Args:
        companies: list of company dicts from fetcher (must have 'ticker'
                   and 'business_description' fields)
        config: full config dict

    Returns:
        dict mapping ticker -> extraction result dict
        Tickers that were skipped or failed have extraction_failed=True

    Side effects:
        Writes/updates data/checkpoints/llm_checkpoint.json
        Logs progress to pipeline.log
    """
    cfg = as_config(config)
    checkpoint = _load_checkpoint()
    batch_size = cfg.llm.batch_size
    client = openai.OpenAI()

    processed_since_save = 0

    for company in companies:
        ticker = company["ticker"]

        if ticker in checkpoint:
            logger.debug(f"{ticker} — skipped (already in checkpoint)")
            continue

        description = company.get("business_description")
        if description is None:
            logger.debug(f"{ticker} — skipped (no description)")
            checkpoint[ticker] = _failed_result()
            continue
        if len(description) < MIN_DESCRIPTION_LENGTH:
            logger.debug(f"{ticker} — skipped (description too short)")
            checkpoint[ticker] = _failed_result()
            continue

        logger.info(f"Analyzing {ticker}...")
        checkpoint[ticker] = _extract_and_judge(
            client, ticker, company.get("company_name", ticker), description, cfg,
        )

        processed_since_save += 1
        if processed_since_save >= batch_size:
            _save_checkpoint(checkpoint)
            processed_since_save = 0

    _save_checkpoint(checkpoint)
    return checkpoint


def analyze_target(config: PipelineConfig | dict) -> dict:
    """
    Run LLM extraction on the target company description from config.
    Returns the same extraction result dict format.
    Does NOT use the checkpoint (target company is always re-analyzed).
    """
    cfg = as_config(config)
    target = cfg.target_company
    client = openai.OpenAI()
    return _extract_and_judge(client, "TARGET", target.name, target.description, cfg)


def suggest_sic_codes(config: PipelineConfig | dict) -> list[dict]:
    """
    Suggest candidate SIC codes for the target's primary/adjacent_sic_codes
    config fields, based on its free-text description — advisory only, does
    not write to config. LLMs are known to misremember specific 4-digit SIC
    codes (a fixed, official SEC list, not something inferable from
    business reasoning alone), so every suggestion carries a self-reported
    confidence and an explicit instruction to verify against SEC's own SIC
    code list (https://www.sec.gov/info/edgar/siccodes.htm) before adding it
    to config.yaml. Never raises — returns [] on any failure.
    """
    cfg = as_config(config)
    target = cfg.target_company
    llm_config = cfg.llm
    client = openai.OpenAI()

    prompt = SIC_SUGGESTION_PROMPT_TEMPLATE.format(description=target.description)
    try:
        text = _call_openai(
            client, llm_config.extraction_model, SIC_SUGGESTION_SYSTEM_PROMPT, prompt,
            llm_config.temperature, llm_config.max_tokens,
        )
    except Exception as e:
        logger.warning(f"SIC code suggestion API call failed: {e}")
        return []

    parsed = _parse_json_response(text, "SIC_SUGGESTION")
    if parsed is None:
        return []

    suggestions = parsed.get("suggestions")
    if not isinstance(suggestions, list):
        logger.warning(f"SIC code suggestion response missing 'suggestions' list: {parsed!r}")
        return []

    return [
        {field: s.get(field) for field in SIC_SUGGESTION_FIELDS}
        for s in suggestions if isinstance(s, dict)
    ]


# Fallback only for direct calls that don't go through config (e.g. tests,
# eval scripts run standalone) — config.yaml's llm.embedding_model is the
# source of truth for real pipeline runs.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def embed_texts(texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]] | None:
    """
    Embed a list of strings in a single batched OpenAI call (cheap — this
    model costs a small fraction of a cent per 1K tokens, and the inputs
    here are one-sentence sub_sector_description strings for a few dozen
    companies at most, not a full-market index). This catches comps that pass
    the coarse business_model/customer_type categorical match but are actually
    a different end market entirely (e.g. semiconductor process equipment vs.
    automotive parts both tagged "manufacturing"/B2B) — see
    reporter._subsector_similarities(). Returns None on any failure; callers
    should treat that as "similarity unknown" and skip the penalty rather
    than block the report.
    """
    if not texts:
        return []
    client = openai.OpenAI()
    try:
        response = client.embeddings.create(model=model, input=texts)
    except Exception as e:
        logger.warning(f"Embedding call failed for {len(texts)} texts: {e}")
        return None
    return [item.embedding for item in response.data]
