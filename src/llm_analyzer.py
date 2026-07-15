import json
import re
from pathlib import Path

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from src import get_logger

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
  supports your extraction (5 = very explicit)."""

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
    "capital_intensity", "primary_value_driver", "sub_sector_description", "confidence",
)

MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


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
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
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
        "confidence": None,
        "judge_score": None,
        "judge_reason": None,
        "low_confidence_flag": False,
        "extraction_failed": True,
    }


def _extract_and_judge(client, ticker: str, company_name: str, business_description: str, config: dict) -> dict:
    llm_config = config["llm"]

    user_prompt = USER_PROMPT_TEMPLATE.format(
        company_name=company_name, business_description=business_description,
    )

    try:
        extraction_text = _call_openai(
            client, llm_config["extraction_model"], SYSTEM_PROMPT, user_prompt,
            llm_config["temperature"], llm_config["max_tokens"],
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
            client, llm_config["judge_model"], None, judge_prompt,
            llm_config["temperature"], llm_config["max_tokens"],
        )
        judge_result = _parse_json_response(judge_text, ticker)
    except Exception as e:
        logger.warning(f"{ticker} — judge API call failed: {e}")

    judge_score = judge_result.get("score") if judge_result else None
    judge_reason = judge_result.get("reason") if judge_result else None
    judge_threshold = llm_config["judge_threshold"]

    result = {field: extraction.get(field) for field in EXTRACTION_FIELDS}
    result["judge_score"] = judge_score
    result["judge_reason"] = judge_reason
    result["low_confidence_flag"] = judge_score is not None and judge_score < judge_threshold
    result["extraction_failed"] = False
    return result


def analyze_batch(companies: list[dict], config: dict) -> dict[str, dict]:
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
    checkpoint = _load_checkpoint()
    batch_size = config["llm"]["batch_size"]
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
            client, ticker, company.get("company_name", ticker), description, config,
        )

        processed_since_save += 1
        if processed_since_save >= batch_size:
            _save_checkpoint(checkpoint)
            processed_since_save = 0

    _save_checkpoint(checkpoint)
    return checkpoint


def analyze_target(config: dict) -> dict:
    """
    Run LLM extraction on the target company description from config.
    Returns the same extraction result dict format.
    Does NOT use the checkpoint (target company is always re-analyzed).
    """
    target = config["target_company"]
    client = openai.OpenAI()
    return _extract_and_judge(client, "TARGET", target["name"], target["description"], config)
