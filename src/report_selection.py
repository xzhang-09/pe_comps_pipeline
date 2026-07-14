"""Top-N comp selection: eligibility, soft penalties, ranking, tiering, audit.

Selection semantics live apart from presentation/rendering because changes
here move the report's actual numbers. reporter.py re-exports the shared
helpers so existing callers and tests can keep addressing reporter.<name>.
"""
import math

import numpy as np
import pandas as pd

from src import get_logger, llm_analyzer, llm_reranker

logger = get_logger(__name__)

TOP_N = 15

# business_model/customer_type values exempt from their respective soft
# penalties (too generic/ambiguous to penalize a mismatch against). The
# actual penalty magnitudes, the sub-sector similarity threshold, and the
# size-penalty constants all live in config.yaml's scorer.ranking_penalties
# (see src/config_schema.py's RankingPenaltiesConfig for what each one does
# and why the defaults are what they are) — not hardcoded here, so
# eval/evaluator.py's _select_top_k can read the exact same values instead
# of maintaining its own copy that can drift out of sync.
EXEMPT_BUSINESS_MODELS = (None, "other")
EXEMPT_CUSTOMER_TYPES = (None, "mixed")

# How many near-miss candidates (ranked just below the Top-N cutoff) to
# surface in the report's audit trail.
AUDIT_SIZE = 5

# Reframes the Top-N's flat rank order into three named confidence tiers a
# reader can act on directly, instead of leaving them to infer "which of
# these 15 should I actually trust" from a fit_flag/outlier_flag border
# color buried in a 10-column table. A comp lands in review_exclude if
# *either* signal (LLM qualitative judgment or the statistical outlier
# check) flags it — the two checks catch different failure modes, so
# either one firing is reason enough for a second look.
TIER_LABELS = {"core": "Core Comps", "secondary": "Secondary Comps", "review_exclude": "Review / Exclude"}

# Ordering used to lay out the final report: tier first, financial-distance
# rank within tier. A review_exclude comp can otherwise land at rank 1 purely
# on financial closeness (observed in validation), which reads as an
# endorsement the tier label then has to walk back.
TIER_ORDER = {"core": 0, "secondary": 1, "review_exclude": 2}


def _assign_tier(fit_flag: str | None, outlier_flag: bool, has_mismatch: bool) -> str:
    """
    has_mismatch is any core-blocking signal (see reporter._annotate_top_rows):
    a business-model or customer-type mismatch, any end-market similarity
    shortfall, an incomplete business profile (missing extraction fields —
    unknown isn't penalized as a mismatch but shouldn't anchor the Core
    tier), or a revenue gap beyond CORE_MAX_LOG10_REVENUE_GAP. It caps a
    comp at secondary even when the LLM review calls it a strongest fit. A
    comp with none of those blockers is core even if the qualitative review
    did not include it in the limited "top_fits" callouts; otherwise the Core
    tier can stay empty purely because the review named only three candidates.
    """
    if fit_flag == "weak" or outlier_flag:
        return "review_exclude"
    if not has_mismatch:
        return "core"
    return "secondary"


def _size_log10_gap(candidate_revenue: float | None, target_revenue: float | None) -> float | None:
    """Absolute log10 revenue ratio between candidate and target, or None
    when either side is missing/non-positive (no size signal available)."""
    if not candidate_revenue or not target_revenue or candidate_revenue <= 0 or target_revenue <= 0:
        return None
    return abs(math.log10(candidate_revenue / target_revenue))


def _size_mismatch_penalty(candidate_revenue: float | None, target_revenue: float | None, penalties: dict) -> float:
    log_ratio = _size_log10_gap(candidate_revenue, target_revenue)
    if log_ratio is None:
        return 0.0
    excess = max(0.0, log_ratio - penalties["size_penalty_free_log10_range"])
    return excess * penalties["size_penalty_per_extra_log10"]


def _subsector_mismatch_penalty(similarity: float | None, penalties: dict) -> float:
    """
    Graded end-market penalty: 0 at/above the similarity threshold, ramping
    linearly to the configured maximum at similarity 0. See
    RankingPenaltiesConfig — a cliff at the threshold made rankings flip on
    embedding-noise-sized similarity differences.
    """
    threshold = penalties["subsector_similarity_threshold"]
    if similarity is None or similarity >= threshold or threshold <= 0:
        return 0.0
    return penalties["subsector_mismatch_penalty"] * (threshold - similarity) / threshold


def _subsector_similarities(
    target_description: str | None,
    llm_features: dict,
    eligible_tickers: list[str],
    embedding_model: str,
) -> dict[str, float]:
    """
    Cosine similarity between the target's sub_sector_description and each
    eligible candidate's, via OpenAI embeddings. Returns {} (no penalty
    applied to anyone) if the target has no sub_sector_description, no
    candidate has one either, or the embedding call fails — an embeddings
    outage shouldn't block report generation, it should just fall back to
    the coarser business_model/customer_type/size penalties alone.
    """
    if not target_description:
        return {}

    tickers_with_text = [t for t in eligible_tickers if llm_features.get(t, {}).get("sub_sector_description")]
    if not tickers_with_text:
        return {}

    texts = [target_description] + [llm_features[t]["sub_sector_description"] for t in tickers_with_text]
    vectors = llm_analyzer.embed_texts(texts, model=embedding_model)
    if vectors is None:
        logger.warning("Sub-sector embedding lookup failed; skipping sub-sector mismatch penalty for this run")
        return {}

    target_vec = np.array(vectors[0])
    target_norm = np.linalg.norm(target_vec)
    if target_norm == 0:
        return {}

    similarities = {}
    for ticker, vec in zip(tickers_with_text, vectors[1:]):
        candidate_vec = np.array(vec)
        candidate_norm = np.linalg.norm(candidate_vec)
        if candidate_norm == 0:
            continue
        similarities[ticker] = float(np.dot(target_vec, candidate_vec) / (target_norm * candidate_norm))
    return similarities


def _penalty_breakdown(
    ticker: str,
    base_rank: int,
    residual: float,
    llm_features: dict,
    companies_by_ticker: dict,
    target_business_model: str | None,
    target_customer_type: str | None,
    target_revenue: float | None,
    apply_business_model_penalty: bool,
    apply_customer_type_penalty: bool,
    subsector_similarities: dict[str, float],
    penalties: dict,
) -> dict:
    llm = llm_features[ticker]
    candidate_business_model = llm.get("business_model")
    candidate_customer_type = llm.get("customer_type")
    candidate_revenue = companies_by_ticker.get(ticker, {}).get("revenue_ttm_usd_mm")
    subsector_similarity = subsector_similarities.get(ticker)
    subsector_threshold = penalties["subsector_similarity_threshold"]

    # A candidate-side None is "unknown", not a mismatch: penalizing it would
    # equate a terse 10-K description with an actual B2C/asset-light profile.
    # Unknown fields instead surface via profile_incomplete below (a Core-tier
    # blocker with a fit note, no distance penalty).
    business_model_penalty = (
        penalties["business_model_penalty"]
        if apply_business_model_penalty and candidate_business_model is not None
        and candidate_business_model != target_business_model
        else 0.0
    )
    customer_type_penalty = (
        penalties["customer_type_penalty"]
        if apply_customer_type_penalty and candidate_customer_type is not None
        and candidate_customer_type != target_customer_type
        else 0.0
    )
    size_penalty = _size_mismatch_penalty(candidate_revenue, target_revenue, penalties)
    subsector_penalty = _subsector_mismatch_penalty(subsector_similarity, penalties)

    profile_incomplete = bool(llm.get("profile_incomplete"))
    missing_fields = [f for f in llm_analyzer.CORE_PROFILE_FIELDS if llm.get(f) is None]

    reasons = []
    if profile_incomplete and missing_fields:
        reasons.append(
            f"business profile incomplete ({', '.join(missing_fields)} unavailable; kept in pool, Core tier blocked)"
        )
    if business_model_penalty:
        reasons.append(f"business model mismatch ({candidate_business_model} vs target's {target_business_model})")
    if customer_type_penalty:
        reasons.append(f"customer type mismatch ({candidate_customer_type} vs target's {target_customer_type})")
    if size_penalty:
        reasons.append(f"revenue scale mismatch (+{size_penalty:.1f} rank penalty)")
    if subsector_penalty:
        reasons.append(
            f"end-market similarity below threshold ({subsector_similarity:.2f} vs. {subsector_threshold}; "
            f"+{subsector_penalty:.2f} rank penalty)"
        )

    # Penalties are added to the *continuous* financial distance (residual_abs),
    # not to the ordinal base_rank. Adding to an ordinal rank threw away how much
    # closer one comp was than another — the gap between residual 0.40 and 0.42
    # counted the same single rank-step as the gap between 0.40 and 1.33 — so a
    # fixed penalty leapfrogged a company across the whole list regardless of how
    # strong its financial fit actually was. In distance units a penalty demotes
    # a comp by a comparable amount of financial distance, so a much closer comp
    # can't be unseated by one categorical flag. base_rank is retained purely for
    # the audit trail's "financial-only rank" display.
    return {
        "ticker": ticker,
        "base_rank": base_rank,
        "adjusted_score": residual + business_model_penalty + customer_type_penalty + size_penalty + subsector_penalty,
        "business_model_penalty": business_model_penalty,
        "customer_type_penalty": customer_type_penalty,
        "size_penalty": size_penalty,
        "size_log10_gap": _size_log10_gap(candidate_revenue, target_revenue),
        "subsector_penalty": subsector_penalty,
        "profile_incomplete": profile_incomplete,
        "reasons": reasons,
    }


def _eligible_candidates(
    company_scores: pd.DataFrame,
    llm_features: dict,
    companies_by_ticker: dict,
    excluded_tickers: set[str] | None = None,
    exclude_training: bool = True,
) -> list[str]:
    """Every ticker that survives the hard low-confidence/training filter —
    the pool Top-N is actually selected from. Shared by ranking, the audit
    trail, and the relative-dispersion diagnostic (which needs the same
    pool as its "before selection" baseline)."""
    excluded_tickers = excluded_tickers or set()
    return [
        ticker for ticker in company_scores.index
        if ticker not in excluded_tickers
        if llm_features.get(ticker) is not None
        and not llm_features[ticker].get("low_confidence_flag")
        and (not exclude_training or companies_by_ticker.get(ticker, {}).get("source_bucket") != "training")
    ]


def _ranked_candidates(
    company_scores: pd.DataFrame,
    llm_features: dict,
    companies_by_ticker: dict,
    target_business_model: str | None,
    target_customer_type: str | None,
    target_revenue: float | None,
    subsector_similarities: dict[str, float],
    penalties: dict,
    excluded_tickers: set[str] | None = None,
    exclude_training: bool = True,
) -> list[dict]:
    """
    Every candidate that survives the hard low-confidence/training filter,
    with its full penalty breakdown, sorted best-to-worst by adjusted score
    (financial distance + penalties, all in distance units).
    Shared by _select_top_15 (which just takes the head) and the report's
    audit trail (which looks at what fell just outside the cutoff and why).
    """
    apply_business_model_penalty = target_business_model not in EXEMPT_BUSINESS_MODELS
    apply_customer_type_penalty = target_customer_type not in EXEMPT_CUSTOMER_TYPES

    candidates = _eligible_candidates(
        company_scores, llm_features, companies_by_ticker,
        excluded_tickers=excluded_tickers, exclude_training=exclude_training,
    )

    base_rank = {
        ticker: rank
        for rank, ticker in enumerate(
            sorted(candidates, key=lambda t: company_scores.loc[t, "residual_abs"]), start=1,
        )
    }

    breakdowns = [
        _penalty_breakdown(
            ticker, base_rank[ticker], float(company_scores.loc[ticker, "residual_abs"]),
            llm_features, companies_by_ticker,
            target_business_model, target_customer_type, target_revenue,
            apply_business_model_penalty, apply_customer_type_penalty,
            subsector_similarities, penalties,
        )
        for ticker in candidates
    ]
    breakdowns.sort(key=lambda b: (b["adjusted_score"], b["base_rank"]))
    return breakdowns


def _select_ranked_tickers(
    ranked: list[dict],
    k: int,
    *,
    llm_rerank: dict | None = None,
    rerank_context: dict | None = None,
) -> list[str]:
    if not llm_rerank or not llm_rerank.get("enabled"):
        return [b["ticker"] for b in ranked[:k]]

    context = rerank_context or {}
    reranked = llm_reranker.rerank(
        ranked=ranked,
        target_profile=context.get("target_profile", {}),
        llm_features=context.get("llm_features", {}),
        companies_by_ticker=context.get("companies_by_ticker", {}),
        model=llm_rerank.get("model", "gpt-4.1"),
        temperature=llm_rerank.get("temperature", 0),
        max_tokens=llm_rerank.get("max_tokens", 1200),
        rerank_window=int(llm_rerank.get("rerank_window", 30)),
    )
    if reranked is None:
        return [b["ticker"] for b in ranked[:k]]

    ordered_window, _moves = reranked
    window_set = set(ordered_window)
    tail = [b["ticker"] for b in ranked if b["ticker"] not in window_set]
    return (ordered_window + tail)[:k]


def _select_top_15(
    company_scores: pd.DataFrame,
    llm_features: dict,
    companies_by_ticker: dict,
    target_business_model: str | None,
    target_customer_type: str | None,
    target_revenue: float | None,
    subsector_similarities: dict[str, float] | None,
    penalties: dict,
    llm_rerank: dict | None = None,
    rerank_context: dict | None = None,
    k: int = TOP_N,
) -> list[str]:
    """
    A hard filter on low_confidence_flag, then residual_abs ranking with
    soft penalties (not exclusions, magnitudes from `penalties` — see
    config.yaml's scorer.ranking_penalties) for:
    - business_model mismatch
    - customer_type mismatch — catches cases business_model alone
      doesn't, e.g. a government-contractor "manufacturer" vs. a B2B
      commercial one, both tagged "manufacturing" by the LLM
    - revenue-scale mismatch (continuous, see _size_mismatch_penalty) —
      prevents financially close but severely size-mismatched companies
      from ranking too highly
    - sub-sector mismatch (continuous, via embeddings — see
      _subsector_similarities) — catches "right categorical tags, wrong end
      market" cases business_model/customer_type can't see
    """
    ranked = _ranked_candidates(
        company_scores, llm_features, companies_by_ticker,
        target_business_model, target_customer_type, target_revenue,
        subsector_similarities or {}, penalties,
        excluded_tickers=None,
        exclude_training=True,
    )
    if len(ranked) < k:
        logger.warning(f"Only {len(ranked)} companies available after low-confidence filter (wanted {k})")
    default_context = {
        "target_profile": {
            "business_model": target_business_model,
            "customer_type": target_customer_type,
            "revenue_usd_mm": target_revenue,
        },
        "llm_features": llm_features,
        "companies_by_ticker": companies_by_ticker,
    }
    return _select_ranked_tickers(
        ranked,
        k,
        llm_rerank=llm_rerank,
        rerank_context=rerank_context or default_context,
    )


def _audit_trail(
    company_scores: pd.DataFrame,
    llm_features: dict,
    companies_by_ticker: dict,
    target_business_model: str | None,
    target_customer_type: str | None,
    target_revenue: float | None,
    top_n: int,
    subsector_similarities: dict[str, float] | None,
    penalties: dict,
    audit_size: int = AUDIT_SIZE,
) -> list[dict]:
    """
    The `audit_size` candidates ranked just outside the Top-N cutoff, with
    why they didn't make it — answers the "why isn't competitor X in the
    comps table" question an IC partner tends to ask. Only candidates with
    a non-zero penalty are interesting here; a candidate that just barely
    lost on financial distance alone doesn't need a "reason" call-out.
    """
    ranked = _ranked_candidates(
        company_scores, llm_features, companies_by_ticker,
        target_business_model, target_customer_type, target_revenue,
        subsector_similarities or {}, penalties,
        excluded_tickers=None,
        exclude_training=True,
    )
    near_misses = ranked[top_n:top_n + audit_size]

    rows = []
    for b in near_misses:
        ticker = b["ticker"]
        company = companies_by_ticker.get(ticker, {})
        rows.append({
            "ticker": ticker,
            "company_name": company.get("company_name", ticker),
            "base_rank": b["base_rank"],
            "reasons": b["reasons"] or ["no soft-penalty mismatch — ranked just below the cutoff on financial distance alone"],
        })
    rows.sort(key=lambda row: row["base_rank"])
    return rows
