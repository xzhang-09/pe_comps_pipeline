import hashlib
import json
from datetime import datetime, timezone

import openai

from src import get_logger, json_store
from src.config_schema import PipelineConfig, TargetCompanyConfig, as_config
from src.llm_analyzer import _call_openai_structured, _strip_markdown_fences
from src.llm_schemas import CompFitReview
from src.paths import project_path

logger = get_logger(__name__)

REVIEW_PATH = project_path("outputs", "comp_fit_review.json")
REVIEW_MAX_OUTPUT_TOKENS = 2500

REVIEW_PROMPT_VERSION = "analyst_memo_v8_score_calibrated"

SYSTEM_PROMPT = """You are a private equity analyst reviewing whether a selected
public-company comparable set is a good fit for valuing a private target.

Rules:
- Use only the data provided in the prompt. Do not use outside knowledge.
- Write in concise, investment-professional language for a PE analyst.
- Treat this as a directional qualitative review, not a final diligence view.
- Penalize end-market mismatch, customer-type mismatch, extreme scale mismatch,
  weak financial comparability, and low data confidence.
- If discussing confidence flags, specify whether the issue applies to selected
  comps or to excluded candidates.
- Keep reasons concise and specific.
- Before writing a blanket claim like "all selected comps share X" in
  strengths/summary, check whether any comp is named as an exception to X
  anywhere else in your own response (e.g. in weaknesses or questionable_fits).
  If so, say "most" instead of "all" and name the exception, or say "all
  except <ticker>" — do not let the strengths/summary contradict a specific
  exception you yourself raise elsewhere in the same response.
- Only credit a comp with "good"/"strong" end-market fit if its actual end
  markets or sub-sector description substantively overlap with the
  target's described end markets. A shared high-level business_model label
  (e.g. both tagged "manufacturing") is not by itself evidence of end-market
  fit — a comp serving a clearly different end market (e.g. oil & gas
  piping vs. automotive OEM parts) should not be praised for end-market fit
  even if other dimensions (scale, financials) are strong; credit those
  other dimensions by name instead.
- If any selected comp's revenue exceeds 5x the target's (or the target's
  exceeds 5x a comp's), the summary must explicitly state the largest such
  ratio (e.g. "up to 9x larger") rather than a vague qualifier like
  "reasonable range" or "generally comparable" — a vague characterization
  next to a double-digit size gap reads as inconsistent to the reader.
- The target JSON includes a pre-computed "max_revenue_ratio_among_selected_comps"
  field (ticker and ratio). When stating the largest revenue-scale ratio,
  use that exact number — do not compute your own ratio from the individual
  comp revenue figures, which is error-prone across a list of this size.
- Score-band calibration:
  - 80-100 means strong comp-set support with only minor caveats.
  - 70-79 means useful support, but not banker-grade without analyst review.
  - 60-69 means mixed directional support with material caveats.
  - 50-59 means weak/mixed support; use only as a rough screen.
  - Below 50 means weak support.
- Do not call a set good if several selected comps have clearly different end
  markets, if 25% or more of selected comps would require review/exclusion, or
  if multiple selected comps exceed 5x target revenue. In those cases, scores
  in the 60-69 range are usually more appropriate unless there are unusually
  strong offsetting reasons.
- top_fits and questionable_fits must reference only tickers listed under
  Selected Top Comps. near_miss_upgrades must reference only tickers listed
  under Near-Miss Candidates. Never describe a near-miss ticker as a selected
  comp."""

PROMPT_TEMPLATE = """Review this comparable-company selection.

Target:
{target_json}

Selected Top Comps:
{top_comps_json}

Near-Miss Candidates:
{near_misses_json}

Score the Top Comps as a set and call out strongest/weakest fits. Consider:
- business_model_fit (0-25)
- end_market_fit (0-25)
- customer_type_fit (0-15)
- revenue_scale_fit (0-15)
- financial_profile_fit (0-15)
- data_confidence (0-5)
"""


def _canonical_json(data) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def _signature(target_config: TargetCompanyConfig, top_comps: list[dict], near_misses: list[dict]) -> str:
    payload = {
        "target": {
            "name": target_config.name,
            "description": target_config.description,
            "revenue_usd_mm": target_config.revenue_usd_mm,
            "ebitda_margin_estimate": target_config.ebitda_margin_estimate,
        },
        "review_prompt_version": REVIEW_PROMPT_VERSION,
        "top_comps": top_comps,
        "near_misses": near_misses,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _load_cached(signature: str) -> dict | None:
    cached = json_store.load_json(REVIEW_PATH)
    if not isinstance(cached, dict):
        return None
    if cached.get("input_signature") != signature:
        return None
    review = cached.get("review")
    return _normalize_review(review) if isinstance(review, dict) else None


def _save_cached(signature: str, review: dict) -> None:
    record = {
        "input_signature": signature,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review": review,
    }
    json_store.write_json_atomic(REVIEW_PATH, record)


def _parse_review(text: str | None) -> dict | None:
    if not text:
        return None
    cleaned = _strip_markdown_fences(text)
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _clean_analyst_text(value):
    replacements = {
        "No low-confidence flags on selected comps": "No source-support issues among selected comps",
        "no low-confidence flags on selected comps": "no source-support issues among selected comps",
        "no low-confidence flags noted": "no source-support issues noted among selected comps",
        "low-confidence flags": "source-support issues",
        "low confidence flags": "source-support issues",
        "low data confidence": "weak source support",
    }
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_clean_analyst_text(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_analyst_text(item) for key, item in value.items()}
    return value


def _normalize_review(review: dict) -> dict:
    return {
        "status": "available",
        "overall_score": review.get("overall_score"),
        "review_confidence": review.get("review_confidence"),
        "summary": _clean_analyst_text(review.get("summary")),
        "strengths": _clean_analyst_text(review.get("strengths")) if isinstance(review.get("strengths"), list) else [],
        "weaknesses": _clean_analyst_text(review.get("weaknesses")) if isinstance(review.get("weaknesses"), list) else [],
        "top_fits": _clean_analyst_text(review.get("top_fits")) if isinstance(review.get("top_fits"), list) else [],
        "questionable_fits": (
            _clean_analyst_text(review.get("questionable_fits"))
            if isinstance(review.get("questionable_fits"), list)
            else []
        ),
        "near_miss_upgrades": (
            _clean_analyst_text(review.get("near_miss_upgrades"))
            if isinstance(review.get("near_miss_upgrades"), list)
            else []
        ),
    }


def _max_revenue_ratio(target_config: TargetCompanyConfig, top_comps: list[dict]) -> dict | None:
    """
    Precomputes the largest comp/target revenue ratio deterministically and
    hands it to the LLM as a given fact (see SYSTEM_PROMPT's instruction to
    use this value verbatim) instead of asking the LLM to compute it from
    the raw revenue figures itself — LLMs are unreliable at this kind of
    multi-step arithmetic across a list (observed: it has stated "12x" when
    the actual max was 19x, apparently anchoring on a different comp than
    the true max), and a wrong ratio in the summary directly contradicts
    the report's own deterministic scale-reconciliation note elsewhere.
    """
    target_revenue = target_config.revenue_usd_mm
    if not target_revenue or target_revenue <= 0:
        return None
    candidates = [
        (c.get("revenue_ttm_usd_mm"), c.get("ticker")) for c in top_comps
        if c.get("revenue_ttm_usd_mm")
    ]
    if not candidates:
        return None
    max_revenue, max_ticker = max(candidates, key=lambda c: c[0])
    return {"ticker": max_ticker, "ratio": round(max_revenue / target_revenue, 1)}


def review_comp_fit(
    target_config: TargetCompanyConfig | dict,
    top_comps: list[dict],
    near_misses: list[dict],
    config: PipelineConfig | dict,
) -> dict:
    target = (
        target_config if isinstance(target_config, TargetCompanyConfig)
        else TargetCompanyConfig.model_validate(target_config)
    )
    cfg = as_config(config)
    signature = _signature(target, top_comps, near_misses)
    cached = _load_cached(signature)
    if cached is not None:
        return cached

    llm_config = cfg.llm
    prompt = PROMPT_TEMPLATE.format(
        target_json=json.dumps({
            "name": target.name,
            "description": target.description,
            "revenue_usd_mm": target.revenue_usd_mm,
            "ebitda_margin_estimate": target.ebitda_margin_estimate,
            "max_revenue_ratio_among_selected_comps": _max_revenue_ratio(target, top_comps),
        }, indent=2),
        top_comps_json=json.dumps(top_comps, indent=2),
        near_misses_json=json.dumps(near_misses, indent=2),
    )

    try:
        parsed = _call_openai_structured(
            openai.OpenAI(), llm_config.judge_model, SYSTEM_PROMPT, prompt,
            llm_config.temperature, max(REVIEW_MAX_OUTPUT_TOKENS, llm_config.max_tokens), CompFitReview,
        )
    except Exception as e:
        logger.warning(f"Comp-fit review API call failed: {e}")
        return {"status": "unavailable", "reason": "LLM comp-fit review API call failed"}

    review = _normalize_review(parsed.model_dump())
    _save_cached(signature, review)
    return review
