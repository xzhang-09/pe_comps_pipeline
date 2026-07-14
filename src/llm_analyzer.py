import hashlib
import json
import re

import openai
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from src import get_logger, json_store, sic_codes
from src.config_schema import PipelineConfig, as_config
from src.llm_schemas import BusinessModelExtraction, JudgeVerdict, SicSuggestions
from src.paths import project_path

logger = get_logger(__name__)

CHECKPOINT_PATH = project_path("data", "checkpoints", "llm_checkpoint.json")
PROMPT_VERSION = "structured_outputs_v1"

MIN_DESCRIPTION_LENGTH = 100

SYSTEM_PROMPT = """You are a private equity analyst extracting structured information from
company business descriptions to support comparable company analysis.

Rules:
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
"""

JUDGE_PROMPT_TEMPLATE = """You are reviewing a business model extraction for accuracy.

Original description (first 500 chars): {description_excerpt}
Extraction result: {extraction_json}

Rate the accuracy of the extraction on a scale of 1-5:
5 = All fields are clearly supported by the text
4 = Most fields accurate, minor uncertainty on 1-2 fields
3 = Partially accurate, some fields unclear or questionable
2 = Several fields appear inaccurate or unsupported
1 = Extraction is largely inaccurate or hallucinated
"""

EXTRACTION_FIELDS = (
    "business_model", "revenue_recurrence", "customer_type",
    "capital_intensity", "primary_value_driver", "sub_sector_description",
    "evidence_quote", "confidence",
)

CORE_PROFILE_FIELDS = (
    "business_model",
    "customer_type",
    "capital_intensity",
    "primary_value_driver",
    "sub_sector_description",
)

SIC_SUGGESTION_SYSTEM_PROMPT = """You are an SEC filings analyst helping a private equity analyst find
plausible SIC (Standard Industrial Classification) codes for a target company,
to use as SEC EDGAR search filters for finding public comparable companies.

Rules:
- SIC codes are a fixed, official 4-digit SEC list. You may misremember
  specific codes — this is a known limitation of language models, not a
  reflection of how well the company's industry is understood. Set
  "confidence" honestly: "high" only if you are confident this exact
  4-digit code is correct, "low" if you are recalling it from general
  knowledge and it should be verified before use.
- Suggestions are programmatically validated against SEC's official SIC
  code list after you respond, but the analyst still decides whether each
  validated code is a good business fit."""

SIC_SUGGESTION_PROMPT_TEMPLATE = """Suggest 6-12 candidate SIC codes for SEC EDGAR comparable-company
discovery, based on this target company description.

Description: {description}

For each suggestion, classify it as "primary" (the company's actual
business) or "adjacent" (a related industry that could add useful
comparables but isn't the company's core business).
"""

SIC_SUGGESTION_FIELDS = ("sic_code", "title", "bucket", "reason", "confidence")

MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_matching(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip().lower()


def _description_fingerprint(description: str | None) -> str | None:
    """Content key for checkpoint invalidation: hash of the whitespace-
    normalized description, so an extraction result is only reused while the
    text it was extracted FROM is unchanged. Normalized the same way as
    evidence-quote matching, so cosmetic whitespace churn in a refetched
    filing doesn't needlessly re-extract every company."""
    if not description:
        return None
    return hashlib.sha256(_normalize_for_matching(description).encode("utf-8")).hexdigest()[:16]


def _usable_description(description) -> bool:
    return isinstance(description, str) and len(description) >= MIN_DESCRIPTION_LENGTH


def _checkpoint_entry_reusable(entry: dict, description: str | None, extraction_model: str) -> bool:
    """
    Whether a checkpoint entry may stand in for re-extracting this company.

    - A failed entry is reusable only while the input is still unusable
      (missing/too-short description) — the failure would just repeat. Once a
      usable description exists, retry: a company whose description was
      missing last quarter (or whose extraction hit an API blip) must not be
      excluded from every future comp pool by a permanently cached failure.
    - A successful entry is reusable only if it was extracted from the same
      description text (content hash) by the same extraction model. A company
      that rewrote its 10-K business section, or a config that switched
      models, invalidates the entry — previously it was reused forever, so
      reports could silently rank comps on stale business-model tags.
    - A legacy entry with no stamps (pre-invalidation / pre-structured-output
      checkpoint) is re-extracted once so old free-form JSON results do not
      silently survive an output-contract change.
    """
    if entry.get("extraction_failed"):
        return not _usable_description(description)

    if not _usable_description(description):
        # A good extraction whose source description is now missing/short is a
        # fetch-side data gap, not a content change — keep the prior result
        # rather than downgrading the company to a permanent failure.
        return True

    stored_hash = entry.get("description_sha256")
    if stored_hash is None:
        return False
    if stored_hash != _description_fingerprint(description):
        return False
    stored_model = entry.get("extraction_model")
    stored_prompt_version = entry.get("prompt_version")
    return (stored_model is None or stored_model == extraction_model) and stored_prompt_version == PROMPT_VERSION


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


def _core_profile_complete(result: dict) -> bool:
    return all(result.get(field) is not None for field in CORE_PROFILE_FIELDS)


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
    # Corrupt checkpoint -> {} (json_store logs it): the run re-extracts from
    # scratch, which costs API calls but keeps the run alive — strictly better
    # than crashing on json.load with no hint of which file is broken.
    return json_store.load_json(CHECKPOINT_PATH, default={})


def _save_checkpoint(checkpoint: dict) -> None:
    json_store.write_json_atomic(CHECKPOINT_PATH, checkpoint)


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


@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_fixed(60),
    stop=stop_after_attempt(3),
    before_sleep=_log_rate_limit_retry,
    reraise=True,
)
def _call_openai_structured(
    client,
    model: str,
    instructions: str | None,
    input_text: str,
    temperature: float,
    max_tokens: int,
    schema: type[BaseModel],
) -> BaseModel:
    kwargs = {
        "model": model,
        "input": input_text,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "text_format": schema,
    }
    if instructions is not None:
        kwargs["instructions"] = instructions

    response = client.responses.parse(**kwargs)
    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        raise ValueError("structured OpenAI response did not include output_parsed")
    return parsed


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
        "profile_incomplete": True,
        "extraction_failed": True,
    }


def _extract_and_judge(client, ticker: str, company_name: str, business_description: str, config: PipelineConfig) -> dict:
    llm_config = config.llm

    user_prompt = USER_PROMPT_TEMPLATE.format(
        company_name=company_name, business_description=business_description,
    )

    try:
        extraction_model = _call_openai_structured(
            client, llm_config.extraction_model, SYSTEM_PROMPT, user_prompt,
            llm_config.temperature, llm_config.max_tokens, BusinessModelExtraction,
        )
    except Exception as e:
        logger.warning(f"{ticker} — extraction API call failed: {e}")
        return _failed_result()

    extraction = extraction_model.model_dump()

    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        description_excerpt=business_description[:500],
        extraction_json=json.dumps(extraction),
    )

    judge_result = None
    try:
        judge_model = _call_openai_structured(
            client, llm_config.judge_model, None, judge_prompt,
            llm_config.temperature, llm_config.max_tokens, JudgeVerdict,
        )
        judge_result = judge_model.model_dump()
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

    core_profile_complete = _core_profile_complete(result)
    if not core_profile_complete:
        logger.warning(f"{ticker} — LLM extraction missing one or more core business-profile fields")

    result["evidence_verified"] = evidence_verified
    result["judge_score"] = judge_score
    result["judge_reason"] = judge_reason
    # An incomplete profile (some fields extracted as null) is a coverage
    # problem, not an extraction-quality problem: the candidate stays in the
    # eligible pool with the missing fields treated as unknown (no mismatch
    # penalty, blocked from the Core tier). low_confidence_flag — which hard
    # excludes from the pool — is reserved for signals that the extraction
    # itself may be wrong: unverifiable evidence or a failing judge score.
    result["low_confidence_flag"] = (
        not evidence_verified
        or (judge_score is not None and judge_score < judge_threshold)
    )
    result["profile_incomplete"] = not core_profile_complete
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
        dict mapping ticker -> extraction result dict, covering exactly the
        requested companies. (Previously this returned the entire checkpoint,
        so tickers left over from earlier runs / other targets leaked into
        the caller's counts — e.g. the report's low-confidence tally.)
        Tickers that were skipped or failed have extraction_failed=True

    Side effects:
        Writes/updates data/checkpoints/llm_checkpoint.json. Checkpoint
        entries are keyed by ticker but invalidated by content — see
        _checkpoint_entry_reusable for the description-hash/model/retry rules.
        Logs progress to pipeline.log
    """
    cfg = as_config(config)
    checkpoint = _load_checkpoint()
    batch_size = cfg.llm.batch_size
    extraction_model = cfg.llm.extraction_model
    client = openai.OpenAI()

    results: dict[str, dict] = {}
    processed_since_save = 0
    checkpoint_dirty = False

    for company in companies:
        ticker = company["ticker"]
        description = company.get("business_description")

        entry = checkpoint.get(ticker)
        if entry is not None and _checkpoint_entry_reusable(entry, description, extraction_model):
            if not entry.get("extraction_failed") and entry.get("description_sha256") is None:
                # Legacy entry from before content-keyed invalidation: stamp it
                # so future description/model changes are detectable.
                entry["description_sha256"] = _description_fingerprint(description)
                entry["extraction_model"] = extraction_model
                entry["prompt_version"] = PROMPT_VERSION
                checkpoint_dirty = True
            if not entry.get("extraction_failed"):
                # Recompute both flags from entry content on reuse: entries
                # written before the profile_incomplete/low_confidence split
                # (including ones stamped low-confidence purely for missing
                # fields) get corrected without a re-extraction.
                judge_score = entry.get("judge_score")
                low_confidence = (
                    not entry.get("evidence_verified", True)
                    or (judge_score is not None and judge_score < cfg.llm.judge_threshold)
                )
                profile_incomplete = not _core_profile_complete(entry)
                if (entry.get("low_confidence_flag") != low_confidence
                        or entry.get("profile_incomplete") != profile_incomplete):
                    entry["low_confidence_flag"] = low_confidence
                    entry["profile_incomplete"] = profile_incomplete
                    checkpoint_dirty = True
            logger.debug(f"{ticker} — skipped (already in checkpoint)")
            results[ticker] = entry
            continue

        if not _usable_description(description):
            logger.debug(f"{ticker} — skipped (no description)" if description is None
                         else f"{ticker} — skipped (description too short)")
            checkpoint[ticker] = _failed_result()
            results[ticker] = checkpoint[ticker]
            checkpoint_dirty = True
            continue

        if entry is not None:
            logger.info(f"{ticker} — checkpoint entry stale (description/model changed, or retryable failure); re-analyzing")
        logger.info(f"Analyzing {ticker}...")
        result = _extract_and_judge(
            client, ticker, company.get("company_name", ticker), description, cfg,
        )
        result["description_sha256"] = _description_fingerprint(description)
        result["extraction_model"] = extraction_model
        result["prompt_version"] = PROMPT_VERSION
        checkpoint[ticker] = result
        results[ticker] = result
        checkpoint_dirty = True

        processed_since_save += 1
        if processed_since_save >= batch_size:
            _save_checkpoint(checkpoint)
            processed_since_save = 0

    if checkpoint_dirty:
        _save_checkpoint(checkpoint)
    return results


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


SIC_SUGGESTION_MIN_MAX_TOKENS = 1200


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
        parsed = _call_openai_structured(
            client, llm_config.extraction_model, SIC_SUGGESTION_SYSTEM_PROMPT, prompt,
            llm_config.temperature, llm_config.max_tokens, SicSuggestions,
        )
    except Exception as e:
        logger.warning(f"SIC code suggestion API call failed: {e}")
        return []

    suggestions = parsed.model_dump()["suggestions"]

    raw_suggestions = [
        {field: s.get(field) for field in SIC_SUGGESTION_FIELDS}
        for s in suggestions if isinstance(s, dict)
    ]
    validated = [sic_codes.validate_sic_suggestion(s) for s in raw_suggestions]
    return [s for s in validated if s is not None]


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
