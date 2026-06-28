import csv
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from scipy import stats

from src import comp_fit_reviewer, feature_builder, get_logger, llm_analyzer, scorer
from src.config_schema import PipelineConfig, as_config

logger = get_logger(__name__)

# Plain-language labels for the report's "Top features by distance to target"
# table — model_diagnostics.top_features otherwise shows raw column names
# (e.g. "revenue_ttm_log", "business_model_manufacturing") straight out of
# the feature matrix, which means nothing to a non-technical reader.
FINANCIAL_FEATURE_LABELS = {
    "revenue_ttm_log": "Revenue (TTM)",
    "ebitda_margin": "EBITDA Margin",
    "gross_margin": "Gross Margin",
    "revenue_cagr_3yr": "Revenue Growth (3yr CAGR)",
    "net_debt_ebitda": "Net Debt / EBITDA",
    "capex_revenue": "Capex / Revenue",
}

LLM_FIELD_LABELS = {
    "business_model": "Business Model",
    "revenue_recurrence": "Revenue Recurrence",
    "customer_type": "Customer Type",
    "capital_intensity": "Capital Intensity",
    "primary_value_driver": "Primary Value Driver",
}

# Values are LLM-extracted strings (see llm_analyzer.USER_PROMPT_TEMPLATE);
# listed explicitly rather than auto-titlecased so "SaaS"/"B2B"/"B2C"/"B2G"
# don't get mangled into "Saas"/"B2b".
LLM_VALUE_LABELS = {
    "manufacturing": "Manufacturing", "services": "Services", "SaaS": "SaaS",
    "distribution": "Distribution", "marketplace": "Marketplace", "other": "Other",
    "high": "High", "medium": "Medium", "low": "Low",
    "B2B": "B2B", "B2C": "B2C", "B2G": "B2G", "mixed": "Mixed",
    "asset_heavy": "Asset-Heavy", "moderate": "Moderate", "asset_light": "Asset-Light",
    "technology": "Technology", "scale": "Scale", "relationships": "Relationships", "brand": "Brand",
    "unknown": "Unknown",
}


def _humanize_feature_name(feature: str) -> str:
    if feature in FINANCIAL_FEATURE_LABELS:
        return FINANCIAL_FEATURE_LABELS[feature]

    for field, field_label in LLM_FIELD_LABELS.items():
        prefix = f"{field}_"
        if feature.startswith(prefix):
            value = feature[len(prefix):]
            return f"{field_label}: {LLM_VALUE_LABELS.get(value, value.replace('_', ' ').capitalize())}"

    return feature.replace("_", " ").capitalize()


def _ordinal(value: float | int | None) -> str:
    if value is None:
        return "N/A"

    n = int(round(value))
    suffix = "th"
    if not 10 <= n % 100 <= 20:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# Substantive weaknesses (scale mismatch) outrank cosmetic ones (sub-sector
# diversity) when picking the executive summary's single "key caveat" —
# otherwise whichever weakness the LLM happened to list first wins, which
# isn't necessarily the one that most affects the valuation conclusion.
CAVEAT_PRIORITY_KEYWORDS = ("scale mismatch", "size mismatch", "revenue scale")


def _select_top_caveat(weaknesses: list[str] | None) -> str | None:
    if not weaknesses:
        return None
    for keyword in CAVEAT_PRIORITY_KEYWORDS:
        for weakness in weaknesses:
            if keyword in weakness.lower():
                return weakness
    return weaknesses[0]


def _scale_reconciliation_note(companies_by_ticker: dict, top15: list[str], target_revenue: float | None) -> str | None:
    """
    A deterministic, data-grounded sentence placed right under the LLM's
    free-text summary in Section 3 — that summary and its own weaknesses
    list can describe the same scale mismatch inconsistently (e.g. "within
    a reasonable range for mid-market benchmarking" next to "some comps
    >10x target size"), and editing the LLM's prose to fix that would be
    fragile across runs. Adding a fact-based anchor next to it lets the
    reader resolve the inconsistency themselves instead of taking either
    LLM sentence at face value.
    """
    if target_revenue is None or target_revenue <= 0:
        return None
    revenues = [
        companies_by_ticker[t]["revenue_ttm_usd_mm"] for t in top15
        if companies_by_ticker.get(t, {}).get("revenue_ttm_usd_mm")
    ]
    if not revenues:
        return None
    max_ratio = max(revenues) / target_revenue
    return (
        f"For reference: Top {len(top15)} comp revenue ranges ${min(revenues):.0f}mm–${max(revenues):.0f}mm against "
        f"the target's ${target_revenue:.0f}mm (up to {max_ratio:.1f}x larger) — see the revenue-screened sensitivity "
        f"case in Section 4 for a scale-controlled range."
    )


# Reframes the Top-N's flat rank order into three named confidence tiers a
# reader can act on directly, instead of leaving them to infer "which of
# these 15 should I actually trust" from a fit_flag/outlier_flag border
# color buried in a 10-column table. A comp lands in review_exclude if
# *either* signal (LLM qualitative judgment or the statistical outlier
# check) flags it — the two checks catch different failure modes, so
# either one firing is reason enough for a second look.
TIER_LABELS = {"core": "Core Comps", "secondary": "Secondary Comps", "review_exclude": "Review / Exclude"}

# fetcher.py's description_source values, spelled out for the report —
# "edgar" alone doesn't tell a reader whether that means structured EDGAR
# financial data or just the filing's free-text business description (it's
# the latter; see fetcher._fetch_business_description).
DESCRIPTION_SOURCE_LABELS = {
    "edgar": "SEC EDGAR (filing text)",
    "fmp": "FMP company profile",
}


def _assign_tier(fit_flag: str | None, outlier_flag: bool) -> str:
    if fit_flag == "weak" or outlier_flag:
        return "review_exclude"
    if fit_flag == "strong":
        return "core"
    return "secondary"


# A score that just clears the "Strong" threshold next to a weakness the
# LLM itself called material shouldn't read as confidently as a score that
# clears it cleanly — downgrade_band is the range where one materially
# worded weakness is enough to pull the label down a tier; above it, a
# single weakness isn't treated as disqualifying.
FIT_LABEL_DOWNGRADE_BAND = (80, 90)
CAVEAT_SEVERITY_KEYWORDS = ("significant", "material", "substantial")
# Catches phrasing like "10x target size" or "5.5x larger" — a concrete
# multiple is at least as strong a severity signal as the literal word
# "significant", and LLM-written weaknesses don't reliably use that word
# even when describing an equally severe mismatch (e.g. "skewed larger
# than target, with some comps >10x target size" — no severity keyword,
# but >10x is a stronger claim than "significant" on its own).
SCALE_MAGNITUDE_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*x\b")


def _has_severe_scale_caveat(weaknesses: list[str] | None) -> bool:
    if not weaknesses:
        return False
    for weakness in weaknesses:
        lowered = weakness.lower()
        if not any(k in lowered for k in CAVEAT_PRIORITY_KEYWORDS):
            continue
        if any(s in lowered for s in CAVEAT_SEVERITY_KEYWORDS) or SCALE_MAGNITUDE_PATTERN.search(lowered):
            return True
    return False


# Thresholds shared by the executive summary and the Comparable Fit Review
# section so both describe the same overall_score with the same words.
def _fit_label(score: float | int | None, weaknesses: list[str] | None = None) -> str | None:
    if score is None:
        return None
    lower, upper = FIT_LABEL_DOWNGRADE_BAND
    if lower <= score < upper and _has_severe_scale_caveat(weaknesses):
        return "Good, capped for scale mismatch"
    if score >= 80:
        return "Strong"
    if score >= 65:
        return "Good / directionally supportive"
    if score >= 50:
        return "Mixed"
    return "Weak"


OUTPUTS_DIR = Path("outputs")
CSV_PATH = OUTPUTS_DIR / "comps_report.csv"
HTML_PATH = OUTPUTS_DIR / "comps_report.html"
TEMPLATE_DIR = Path("src/templates")
FAILED_TICKERS_PATH = Path("outputs/failed_tickers.csv")

TOP_N = 15
DEFAULT_REPORT_FORMATS = ("csv", "html")
SUPPORTED_REPORT_FORMATS = set(DEFAULT_REPORT_FORMATS)

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

DISCLAIMER = "Analysis based on public company data via FMP and SEC EDGAR. For reference purposes only."

# (display label, source field, format) — target estimate for ebitda_margin
# comes from config; the other three have no explicit target estimate, so
# they fall back to the same imputation_medians used to fill the target's
# feature vector in scorer._target_financial_row().
BENCHMARK_METRICS = (
    ("EBITDA Margin", "ebitda_margin", "percent"),
    ("Revenue Growth", "revenue_cagr_3yr", "percent"),
    ("Gross Margin", "gross_margin", "percent"),
    ("Capex/Revenue", "capex_revenue", "percent"),
    ("FCF Conversion", "fcf_conversion", "percent"),
    ("Net Debt/EBITDA", "net_debt_ebitda", "multiple"),
    ("Interest Coverage", "interest_coverage", "multiple"),
    ("Debt/Equity", "debt_to_equity", "multiple"),
)

CSV_COLUMNS = (
    "rank", "ticker", "company_name", "ev_ebitda_actual",
    "residual_abs", "ebitda_margin", "gross_margin", "revenue_ttm_usd_mm",
    "revenue_cagr_3yr", "net_debt_ebitda", "business_model", "customer_type",
    "capital_intensity", "sub_sector_description", "judge_score", "low_confidence_flag",
)


def _size_mismatch_penalty(candidate_revenue: float | None, target_revenue: float | None, penalties: dict) -> float:
    if not candidate_revenue or not target_revenue or candidate_revenue <= 0 or target_revenue <= 0:
        return 0.0
    log_ratio = abs(math.log10(candidate_revenue / target_revenue))
    excess = max(0.0, log_ratio - penalties["size_penalty_free_log10_range"])
    return excess * penalties["size_penalty_per_extra_log10"]


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

    business_model_penalty = (
        penalties["business_model_penalty"]
        if apply_business_model_penalty and candidate_business_model != target_business_model
        else 0.0
    )
    customer_type_penalty = (
        penalties["customer_type_penalty"]
        if apply_customer_type_penalty and candidate_customer_type != target_customer_type
        else 0.0
    )
    size_penalty = _size_mismatch_penalty(candidate_revenue, target_revenue, penalties)
    subsector_penalty = (
        penalties["subsector_mismatch_penalty"]
        if subsector_similarity is not None and subsector_similarity < subsector_threshold
        else 0.0
    )

    reasons = []
    if business_model_penalty:
        reasons.append(f"business model mismatch ({candidate_business_model} vs target's {target_business_model})")
    if customer_type_penalty:
        reasons.append(f"customer type mismatch ({candidate_customer_type} vs target's {target_customer_type})")
    if size_penalty:
        reasons.append(f"revenue scale mismatch (+{size_penalty:.1f} rank penalty)")
    if subsector_penalty:
        reasons.append(f"end-market similarity below threshold ({subsector_similarity:.2f} vs. {subsector_threshold} required)")

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
        "subsector_penalty": subsector_penalty,
        "reasons": reasons,
    }


def _eligible_candidates(
    company_scores: pd.DataFrame,
    llm_features: dict,
    companies_by_ticker: dict,
) -> list[str]:
    """Every ticker that survives the hard low-confidence/training filter —
    the pool Top-N is actually selected from. Shared by ranking, the audit
    trail, and the relative-dispersion diagnostic (which needs the same
    pool as its "before selection" baseline)."""
    return [
        ticker for ticker in company_scores.index
        if llm_features.get(ticker) is not None
        and not llm_features[ticker].get("low_confidence_flag")
        and companies_by_ticker.get(ticker, {}).get("source_bucket") != "training"
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

    candidates = _eligible_candidates(company_scores, llm_features, companies_by_ticker)

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


def _select_top_15(
    company_scores: pd.DataFrame,
    llm_features: dict,
    companies_by_ticker: dict,
    target_business_model: str | None,
    target_customer_type: str | None,
    target_revenue: float | None,
    subsector_similarities: dict[str, float] | None,
    penalties: dict,
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
    )
    if len(ranked) < k:
        logger.warning(f"Only {len(ranked)} companies available after low-confidence filter (wanted {k})")
    return [b["ticker"] for b in ranked[:k]]


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


def _distribution_stats(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "mean": float(np.mean(arr)),
    }


MIN_VALUES_FOR_DISPERSION = 2


def _iqr(values: list[float]) -> float | None:
    if len(values) < MIN_VALUES_FOR_DISPERSION:
        return None
    dist = _distribution_stats(values)
    return dist["p75"] - dist["p25"]


def _relative_dispersion(company_scores: pd.DataFrame, eligible_tickers: list[str], top_n_tickers: list[str]) -> dict:
    """
    Does selecting the Top-N actually narrow the spread of EV/EBITDA
    multiples relative to the full eligible pool it was drawn from? A good
    comp set should converge on a usable multiple; a low ratio means the
    selection is doing real work, a ratio near/above 1.0 means the Top-N is
    about as scattered as picking from the pool at random; worth treating
    as a signal to revisit feature_weights or the soft-penalty constants,
    not a hard pass/fail threshold (no validated "good" cutoff is defined,
    see eval/results.md for why an external ground truth is hard to get
    right for this pipeline).

    Reported relative to the eligible pool's own dispersion rather than as
    an absolute CV/IQR, since a target's industry can be intrinsically
    volatile (e.g. cyclicals) — an absolute number conflates "this industry
    is just dispersed" with "the selection isn't converging."
    """
    pool_values = [
        company_scores.loc[t, "ev_ebitda_actual"] for t in eligible_tickers
        if t in company_scores.index and company_scores.loc[t, "ev_ebitda_actual"] is not None
    ]
    selected_values = [company_scores.loc[t, "ev_ebitda_actual"] for t in top_n_tickers]

    pool_iqr = _iqr(pool_values)
    selected_iqr = _iqr(selected_values)

    ratio = None
    if pool_iqr is not None and selected_iqr is not None and pool_iqr > 0:
        ratio = selected_iqr / pool_iqr

    return {
        "pool_iqr": pool_iqr,
        "selected_iqr": selected_iqr,
        "ratio": ratio,
        "n_pool": len(pool_values),
        "n_selected": len(selected_values),
    }


TUKEY_FENCE_MULTIPLIER = 1.5

# A dispersion ratio (Top-N EV/EBITDA IQR ÷ eligible-pool IQR) is graded in
# bands so Section 6's verdict can't contradict Section 4's absolute-spread
# caveat. The ratio is RELATIVE (narrowed vs. the pool the Top-N was drawn
# from); a low ratio doesn't guarantee the resulting band is absolutely tight,
# so only a comfortably large reduction earns "meaningfully narrowed":
#   < 0.60  meaningfully narrowed (>40% tighter than the pool)
#   0.60–0.85 modestly narrowed (still a wide band — defer to §4's range caveat)
#   0.85–1.0  barely narrowed (about as scattered as the pool)
#   >= 1.0    did not narrow
MEANINGFUL_NARROWING_RATIO = 0.60
MODEST_NARROWING_RATIO = 0.85

# When the Top-N's own EV/EBITDA P75÷P25 exceeds this, the multiple spread
# is wide enough that the headline P25–P75 range is more "noise" than
# "signal" and the report says so explicitly rather than presenting the
# range as if it were a tight, usable band.
WIDE_MULTIPLE_SPREAD_RATIO = 2.0

# Top-N comps whose EBITDA margin is below this get a company-specific
# caution flag: a depressed margin usually drags the EV/EBITDA multiple
# (and EV/Revenue) down for reasons specific to that company, so the comp's
# low multiple shouldn't be read as a clean market data point.
LOW_MARGIN_CAUTION_THRESHOLD = 0.10

# An EV/EBITDA outlier guard that does not depend on the set's own IQR.
# When the Top-N is itself very dispersed, the Tukey fence widens enough to
# wave through multiples that are plainly extreme relative to the set's
# center (e.g. a 41x next to a 14x median). Anything above this multiple of
# the set's median is flagged regardless of where the Tukey fence lands.
ABSOLUTE_OUTLIER_MEDIAN_MULTIPLE = 2.5


def _ev_ebitda_outlier_tickers(company_scores: pd.DataFrame, top15: list[str]) -> set[str]:
    """
    Tukey-fence outliers within the Top-N's own EV/EBITDA distribution —
    deliberately independent of the Comparable Fit Review's qualitative
    weaker-fit judgment (comp_fit_review.questionable_fits). A company can
    be a perfectly good business-model/customer-type match and still have a
    valuation multiple that's a statistical outlier relative to its own
    peer set (e.g. a richly-valued growth name); relying solely on the LLM
    to catch that would mean an outlier multiple only gets flagged when the
    LLM's qualitative read happens to agree.
    """
    if len(top15) < MIN_VALUES_FOR_DISPERSION:
        return set()

    values = {t: float(company_scores.loc[t, "ev_ebitda_actual"]) for t in top15}
    dist = _distribution_stats(list(values.values()))
    iqr = dist["p75"] - dist["p25"]
    lower_fence = dist["p25"] - TUKEY_FENCE_MULTIPLIER * iqr
    upper_fence = dist["p75"] + TUKEY_FENCE_MULTIPLIER * iqr
    # Absolute guard (see ABSOLUTE_OUTLIER_MEDIAN_MULTIPLE): on a very
    # dispersed set the Tukey upper fence can sit above multiples that are
    # obviously extreme relative to the median, so flag those too.
    absolute_upper = dist["median"] * ABSOLUTE_OUTLIER_MEDIAN_MULTIPLE
    return {
        t for t, v in values.items()
        if v < lower_fence or v > upper_fence or v > absolute_upper
    }


SCATTER_WIDTH = 720
SCATTER_HEIGHT = 420
SCATTER_MARGIN = {"left": 70, "right": 30, "top": 30, "bottom": 60}
SCATTER_X_TICK_CANDIDATES_USD_MM = (10, 30, 100, 300, 1000, 3000, 10000)


SCATTER_Y_PADDING_MULTIPLE = 1.15


def _revenue_multiple_scatter_svg(
    company_scores: pd.DataFrame, companies_by_ticker: dict, eligible_candidates: list[str],
    top15_rows: list[dict], target_revenue: float | None,
) -> dict | None:
    """
    Hand-built SVG (no charting library dependency — the rest of the report
    generates fine without one, and this is the only chart) plotting
    EV/EBITDA against revenue, log-scaled on the x-axis since revenue spans
    roughly two orders of magnitude across the eligible pool. This turns
    what the rest of the report only describes in prose ("wide range",
    "scale mismatch", "outlier") into something a reader can see in one
    glance: where the target's own revenue sits relative to the comp
    cloud, and which Top-N comps are flagged strong/weak/outlier. The
    eligible pool renders as faint background dots for context; the Top-N
    are the colored, labeled foreground dots.

    The y-axis scales off the Top-N's own EV/EBITDA range (with padding),
    not the full eligible pool's — a single extreme pool outlier (the pool
    can be 4-5x the size of the Top-N) would otherwise stretch the axis so
    far that every Top-N point gets squeezed into the bottom of the chart.
    Pool points above the resulting y_max are dropped rather than rendered
    off-chart or distorting the scale; the count of dropped points is
    returned so the caller can disclose how many aren't shown.
    """
    pool_points = []
    for ticker in eligible_candidates:
        revenue = companies_by_ticker.get(ticker, {}).get("revenue_ttm_usd_mm")
        ev_ebitda = company_scores.loc[ticker, "ev_ebitda_actual"] if ticker in company_scores.index else None
        if revenue and revenue > 0 and ev_ebitda is not None:
            pool_points.append((revenue, float(ev_ebitda)))

    if len(pool_points) < MIN_VALUES_FOR_DISPERSION:
        return None

    top15_points = [
        (row["revenue_ttm_usd_mm"], row["ev_ebitda_actual"], row["ticker"], row.get("fit_flag"), row.get("outlier_flag"))
        for row in top15_rows
        if row.get("revenue_ttm_usd_mm") and row["revenue_ttm_usd_mm"] > 0 and row.get("ev_ebitda_actual") is not None
    ]
    if not top15_points:
        return None

    all_revenues_log = [math.log10(r) for r, _ in pool_points] + ([math.log10(target_revenue)] if target_revenue else [])

    x_min, x_max = min(all_revenues_log), max(all_revenues_log)
    x_pad = (x_max - x_min) * 0.08 or 0.5
    x_min, x_max = x_min - x_pad, x_max + x_pad

    y_min = 0.0
    y_max = max(m for _, m, *_ in top15_points) * SCATTER_Y_PADDING_MULTIPLE
    n_pool_clipped = sum(1 for _, m in pool_points if m > y_max)
    pool_points = [(r, m) for r, m in pool_points if m <= y_max]

    left, right = SCATTER_MARGIN["left"], SCATTER_WIDTH - SCATTER_MARGIN["right"]
    top, bottom = SCATTER_MARGIN["top"], SCATTER_HEIGHT - SCATTER_MARGIN["bottom"]

    def x_pos(revenue: float) -> float:
        frac = (math.log10(revenue) - x_min) / (x_max - x_min)
        return left + frac * (right - left)

    def y_pos(multiple: float) -> float:
        frac = (multiple - y_min) / (y_max - y_min) if y_max > y_min else 0.0
        return bottom - frac * (bottom - top)

    parts = [
        f'<svg viewBox="0 0 {SCATTER_WIDTH} {SCATTER_HEIGHT}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="11">',
        f'<rect x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" fill="none" stroke="#dddddd"/>',
    ]

    n_y_ticks = 5
    for i in range(n_y_ticks + 1):
        val = y_min + (y_max - y_min) * i / n_y_ticks
        y = y_pos(val)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#555555">{val:.0f}x</text>')

    for revenue in SCATTER_X_TICK_CANDIDATES_USD_MM:
        log_revenue = math.log10(revenue)
        if x_min <= log_revenue <= x_max:
            x = x_pos(revenue)
            parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#eeeeee"/>')
            label = f"${revenue}mm" if revenue < 1000 else f"${revenue / 1000:.0f}bn"
            parts.append(f'<text x="{x:.1f}" y="{bottom + 18}" text-anchor="middle" fill="#555555">{label}</text>')

    parts.append(
        f'<text x="{(left + right) / 2:.1f}" y="{SCATTER_HEIGHT - 10}" text-anchor="middle" '
        f'fill="#333333" font-weight="bold">Revenue (log scale)</text>',
    )
    mid_y = (top + bottom) / 2
    parts.append(
        f'<text x="16" y="{mid_y:.1f}" text-anchor="middle" fill="#333333" font-weight="bold" '
        f'transform="rotate(-90 16 {mid_y:.1f})">EV/EBITDA</text>',
    )

    if target_revenue:
        tx = x_pos(target_revenue)
        parts.append(f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{bottom}" stroke="#1a3a5c" stroke-width="2" stroke-dasharray="4,3"/>')
        parts.append(f'<text x="{tx:.1f}" y="{top - 8}" text-anchor="middle" fill="#1a3a5c" font-weight="bold">Target (${target_revenue:.0f}mm)</text>')

    for revenue, multiple in pool_points:
        parts.append(f'<circle cx="{x_pos(revenue):.1f}" cy="{y_pos(multiple):.1f}" r="3" fill="#cccccc" opacity="0.6"/>')

    fit_colors = {"strong": "#2e7d32", "weak": "#c0392b"}
    for revenue, multiple, ticker, fit_flag, outlier_flag in top15_points:
        color = "#c0392b" if outlier_flag else fit_colors.get(fit_flag, "#1a3a5c")
        cx, cy = x_pos(revenue), y_pos(multiple)
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}" stroke="#ffffff" stroke-width="1"/>')
        parts.append(f'<text x="{cx + 7:.1f}" y="{cy + 3:.1f}" fill="#333333">{ticker}</text>')

    legend_items = (
        ("#1a3a5c", "Top-N, no flag"),
        ("#2e7d32", "Strongest fit"),
        ("#c0392b", "Weaker fit / outlier"),
        ("#cccccc", "Eligible pool"),
    )
    legend_x = right - 165
    for i, (color, label) in enumerate(legend_items):
        ly = top + 4 + i * 16
        parts.append(f'<circle cx="{legend_x}" cy="{ly}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 10}" y="{ly + 4}" fill="#333333">{label}</text>')

    parts.append("</svg>")
    return {"svg": "".join(parts), "n_pool_clipped": n_pool_clipped}


FOOTBALL_WIDTH = 720
FOOTBALL_MARGIN = {"left": 185, "right": 110, "top": 22, "bottom": 44}
FOOTBALL_ROW_HEIGHT = 34
FOOTBALL_BAR_HEIGHT = 14


def _football_field_rows(
    implied_valuation: dict, revenue_screened_valuation: dict | None,
    discounted_valuation: dict | None, size_adjusted_valuation: dict | None, size_anchor: dict | None,
) -> list[dict]:
    """Assembles the football-field rows, most decision-relevant first for a
    small private target: the size anchor and the discounted ("working") range
    lead, the raw Top-N comp ranges and the regression point follow. Each range
    row carries low/mid/high (P25/median/P75); each point row a single value.
    Ranges use whichever basis the rest of the report leads with (EV/EBITDA,
    else EV/Revenue) so a bar isn't shown on a different footing than the
    headline."""
    def basis_range(block: dict | None) -> dict | None:
        block = block or {}
        return block.get("by_ebitda") or block.get("by_revenue")

    rows: list[dict] = []
    if size_anchor and size_anchor.get("implied_ev"):
        rows.append({"label": f"Size anchor (n={size_anchor.get('n')})", "kind": "point",
                     "value": size_anchor["implied_ev"], "color": "#b8860b", "emphasis": True})
    discounted_range = basis_range(discounted_valuation)
    if discounted_range:
        pct = round((discounted_valuation.get("discount") or 0) * 100)
        rows.append({"label": f"After {pct}% discount", "kind": "range", "color": "#2e7d32", "emphasis": True,
                     "low": discounted_range["p25"], "mid": discounted_range["median"], "high": discounted_range["p75"]})
    screened_range = basis_range(revenue_screened_valuation)
    if screened_range:
        rows.append({"label": f"Size-comparable (n={revenue_screened_valuation.get('n')})", "kind": "range",
                     "color": "#1a3a5c", "low": screened_range["p25"], "mid": screened_range["median"], "high": screened_range["p75"]})
    by_ebitda = implied_valuation.get("by_ebitda")
    if by_ebitda:
        rows.append({"label": "EV/EBITDA (Top-N)", "kind": "range", "color": "#1a3a5c",
                     "low": by_ebitda["p25"], "mid": by_ebitda["median"], "high": by_ebitda["p75"]})
    by_revenue = implied_valuation.get("by_revenue")
    if by_revenue:
        rows.append({"label": "EV/Revenue (Top-N)", "kind": "range", "color": "#1a3a5c",
                     "low": by_revenue["p25"], "mid": by_revenue["median"], "high": by_revenue["p75"]})
    if size_adjusted_valuation and size_adjusted_valuation.get("implied_ev"):
        rows.append({"label": "Size-adjusted (regr.)", "kind": "point",
                     "value": size_adjusted_valuation["implied_ev"], "color": "#888888"})
    return rows


def _football_field_svg(
    implied_valuation: dict, revenue_screened_valuation: dict | None,
    discounted_valuation: dict | None, size_adjusted_valuation: dict | None, size_anchor: dict | None,
) -> str | None:
    """Horizontal 'football field' of implied enterprise value by valuation
    method — the canonical one-glance comps output. Returns None when no comp
    range is available (target revenue/EBITDA missing), so the template can
    skip it."""
    rows = _football_field_rows(
        implied_valuation, revenue_screened_valuation, discounted_valuation, size_adjusted_valuation, size_anchor,
    )
    if not any(r["kind"] == "range" for r in rows):
        return None

    values: list[float] = []
    for r in rows:
        values += [r["low"], r["high"]] if r["kind"] == "range" else [r["value"]]
    x_min, x_max = min(values), max(values)
    if x_max <= x_min:
        x_max = x_min + 1.0
    pad = (x_max - x_min) * 0.08 or 1.0
    x_min, x_max = max(0.0, x_min - pad), x_max + pad

    left = FOOTBALL_MARGIN["left"]
    right = FOOTBALL_WIDTH - FOOTBALL_MARGIN["right"]
    top = FOOTBALL_MARGIN["top"]
    plot_bottom = top + len(rows) * FOOTBALL_ROW_HEIGHT
    height = plot_bottom + FOOTBALL_MARGIN["bottom"]

    def x_pos(v: float) -> float:
        return left + (v - x_min) / (x_max - x_min) * (right - left)

    parts = [
        f'<svg viewBox="0 0 {FOOTBALL_WIDTH} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="11">',
    ]

    n_ticks = 5
    for i in range(n_ticks + 1):
        v = x_min + (x_max - x_min) * i / n_ticks
        x = x_pos(v)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{plot_bottom:.1f}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{x:.1f}" y="{plot_bottom + 16:.1f}" text-anchor="middle" fill="#555555">${v:.0f}mm</text>')

    for idx, r in enumerate(rows):
        cy = top + idx * FOOTBALL_ROW_HEIGHT + FOOTBALL_ROW_HEIGHT / 2
        weight = "bold" if r.get("emphasis") else "normal"
        parts.append(
            f'<text x="{left - 10}" y="{cy + 4:.1f}" text-anchor="end" fill="#333333" font-weight="{weight}">{r["label"]}</text>',
        )
        if r["kind"] == "range":
            x0, x1, xm = x_pos(r["low"]), x_pos(r["high"]), x_pos(r["mid"])
            bar_top = cy - FOOTBALL_BAR_HEIGHT / 2
            parts.append(
                f'<rect x="{x0:.1f}" y="{bar_top:.1f}" width="{max(x1 - x0, 1):.1f}" height="{FOOTBALL_BAR_HEIGHT}" '
                f'fill="{r["color"]}" opacity="0.30"/>',
            )
            parts.append(
                f'<line x1="{xm:.1f}" y1="{bar_top:.1f}" x2="{xm:.1f}" y2="{bar_top + FOOTBALL_BAR_HEIGHT:.1f}" '
                f'stroke="{r["color"]}" stroke-width="2"/>',
            )
            parts.append(
                f'<text x="{right + 6}" y="{cy + 4:.1f}" text-anchor="start" fill="#333333">'
                f'${r["low"]:.0f}–{r["high"]:.0f}mm</text>',
            )
        else:
            x = x_pos(r["value"])
            d = 5
            parts.append(
                f'<path d="M{x:.1f} {cy - d:.1f} L{x + d:.1f} {cy:.1f} L{x:.1f} {cy + d:.1f} L{x - d:.1f} {cy:.1f} Z" '
                f'fill="{r["color"]}"/>',
            )
            parts.append(
                f'<text x="{right + 6}" y="{cy + 4:.1f}" text-anchor="start" fill="#333333">${r["value"]:.0f}mm</text>',
            )

    parts.append(
        f'<text x="{(left + right) / 2:.1f}" y="{height - 6}" text-anchor="middle" '
        f'fill="#333333" font-weight="bold">Implied Enterprise Value ($mm) — bar = P25–P75, tick = median</text>',
    )
    parts.append("</svg>")
    return "".join(parts)


MIN_POOL_FOR_SIZE_REGRESSION = 10


def _size_adjusted_valuation(
    company_scores: pd.DataFrame, companies_by_ticker: dict, eligible_candidates: list[str],
    target_revenue: float | None, target_ebitda: float | None,
) -> dict | None:
    """
    Regresses EV/EBITDA on log10(revenue) across the full eligible pool
    (typically 4-5x the size of the Top-N, so the slope estimate is less
    noisy than anything computed on just 15 comps) to quantify whether this
    industry's multiples scale with company size — and if so, what that
    implies for a target as small/large as this one. This is the
    quantitative counterpart to the report's qualitative "many comps are
    larger than the target" observations elsewhere.
    """
    if target_revenue is None or target_ebitda is None or target_revenue <= 0:
        return None

    xs, ys = [], []
    for ticker in eligible_candidates:
        revenue = companies_by_ticker.get(ticker, {}).get("revenue_ttm_usd_mm")
        ev_ebitda = company_scores.loc[ticker, "ev_ebitda_actual"] if ticker in company_scores.index else None
        if revenue and revenue > 0 and ev_ebitda is not None:
            xs.append(math.log10(revenue))
            ys.append(float(ev_ebitda))

    if len(xs) < MIN_POOL_FOR_SIZE_REGRESSION or len(set(xs)) < 2:
        return None

    slope, intercept, r_value, p_value, _ = stats.linregress(xs, ys)
    predicted_multiple = slope * math.log10(target_revenue) + intercept
    if predicted_multiple <= 0:
        return None

    return {
        "n": len(xs),
        "slope": slope,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "is_significant": p_value < 0.05,
        "predicted_multiple": predicted_multiple,
        "implied_ev": predicted_multiple * target_ebitda,
    }


# How far (in either direction) a Top-N comp's revenue may sit from the
# target's before it's excluded from the revenue-screened sensitivity case.
# 10x is generous on purpose — narrow bands (e.g. 0.5x-2x) leave too few
# Top-N comps to form a meaningful distribution for most targets. This is
# a coarse filter on revenue only — it does not exclude comps already
# flagged as weaker fits or valuation outliers elsewhere in the report, so
# a comp can appear here even if the other sensitivity case excludes it.
SIZE_BAND_MULTIPLE = 10


def _revenue_screened_valuation(
    company_scores: pd.DataFrame, companies_by_ticker: dict, top15: list[str],
    target_config: dict, imputation_medians: dict, target_business_model: str | None,
) -> dict | None:
    """
    A sensitivity case using only the Top-N comps within SIZE_BAND_MULTIPLE x
    of the target's own revenue — directly answers "what does the comp set
    imply if I only look at companies closer to my actual size", which is a
    more direct response to a scale-mismatch caveat than the regression-based
    _size_adjusted_valuation (which answers a related but different question:
    how the multiple trends with size across the *entire* eligible pool).
    Returns None if too few comps fall within the band to be meaningful.
    """
    target_revenue = target_config.get("revenue_usd_mm")
    if target_revenue is None or target_revenue <= 0:
        return None

    lower = target_revenue / SIZE_BAND_MULTIPLE
    upper = target_revenue * SIZE_BAND_MULTIPLE
    screened_tickers = [
        t for t in top15
        if (revenue := companies_by_ticker.get(t, {}).get("revenue_ttm_usd_mm")) is not None and lower <= revenue <= upper
    ]
    if len(screened_tickers) < MIN_VALUES_FOR_DISPERSION:
        return None

    valuation_multiples = _valuation_multiple_distribution(company_scores, companies_by_ticker, screened_tickers)
    implied = _implied_valuation(target_config, imputation_medians, target_business_model, valuation_multiples)
    implied["n"] = len(screened_tickers)
    implied["tickers"] = screened_tickers
    return implied


# A much tighter band than SIZE_BAND_MULTIPLE (0.5x-5x vs. 10x either way) —
# answers "what if I only look at comps genuinely close to my size." Almost
# always leaves too few Top-N comps for a standalone valuation case, so it's
# reported as a single directional median rather than a full P25/Median/P75
# distribution that would imply more precision than a 3-5 company sample
# can support.
STRICT_SIZE_BAND_LOWER_MULTIPLE = 0.5
STRICT_SIZE_BAND_UPPER_MULTIPLE = 5


def _strict_size_screen(
    company_scores: pd.DataFrame, companies_by_ticker: dict, top15: list[str], target_revenue: float | None,
) -> dict | None:
    if target_revenue is None or target_revenue <= 0:
        return None

    lower = target_revenue * STRICT_SIZE_BAND_LOWER_MULTIPLE
    upper = target_revenue * STRICT_SIZE_BAND_UPPER_MULTIPLE
    tickers = [
        t for t in top15
        if (revenue := companies_by_ticker.get(t, {}).get("revenue_ttm_usd_mm")) is not None and lower <= revenue <= upper
    ]
    if len(tickers) < MIN_VALUES_FOR_DISPERSION:
        return None

    values = [float(company_scores.loc[t, "ev_ebitda_actual"]) for t in tickers if t in company_scores.index]
    if not values:
        return None

    return {
        "n": len(tickers),
        "tickers": tickers,
        "median_ev_ebitda": float(np.median(values)),
    }


def _multiple_values(companies_by_ticker: dict, top15: list[str], field: str) -> list[float]:
    return [
        companies_by_ticker[t][field] for t in top15
        if companies_by_ticker.get(t, {}).get(field) is not None
    ]


def _valuation_multiple_distribution(company_scores: pd.DataFrame, companies_by_ticker: dict, top15: list[str]) -> dict:
    ev_ebitda_values = [company_scores.loc[t, "ev_ebitda_actual"] for t in top15]

    def dist(field: str) -> dict:
        values = _multiple_values(companies_by_ticker, top15, field)
        # Empty -> a zero-distribution placeholder so the template's fixed
        # rows always have p25/median/p75/mean to format (matches how
        # ev_revenue was already handled before more multiples were added).
        return _distribution_stats(values) if values else _distribution_stats([0.0])

    return {
        "ev_ebitda": _distribution_stats(ev_ebitda_values),
        "ev_revenue": dist("ev_revenue"),
        "ev_ebit": dist("ev_ebit"),
        "ev_gross_profit": dist("ev_gross_profit"),
        "pe_ratio": dist("pe_ratio"),
        "fcf_yield": dist("fcf_yield"),
    }


def _target_estimate(field: str, target_config: dict, imputation_medians: dict, target_business_model: str | None) -> tuple[float | None, bool]:
    """Returns (value, is_estimated). is_estimated is True when value is a
    peer-median fallback rather than a number the analyst actually entered
    for the target — only ebitda_margin currently has a real config input
    (target_config.ebitda_margin_estimate); the other BENCHMARK_METRICS
    fields always fall back to imputation_medians, since the target has no
    public filings to source them from. Report consumers need this flag so
    "Target Est." isn't read as company-reported data when it's actually
    just the comp pool's own median.
    """
    if field == "ebitda_margin":
        value = target_config.get("ebitda_margin_estimate")
        if value is not None:
            return value, False
    return feature_builder.median_for(imputation_medians, field, target_business_model), True


def _financial_benchmarks(
    companies_by_ticker: dict, top15: list[str], target_config: dict,
    imputation_medians: dict, target_business_model: str | None,
) -> list[dict]:
    rows = []
    for label, field, fmt in BENCHMARK_METRICS:
        values = [
            companies_by_ticker[t][field] for t in top15
            if companies_by_ticker.get(t, {}).get(field) is not None
        ]
        if not values:
            continue

        target_value, target_est_is_estimated = _target_estimate(field, target_config, imputation_medians, target_business_model)
        percentile = float(stats.percentileofscore(values, target_value)) if target_value is not None else None

        dist = _distribution_stats(values)
        rows.append({
            "metric": label,
            "format": fmt,
            "target_est": target_value,
            "target_est_is_estimated": target_est_is_estimated,
            "p25": dist["p25"],
            "median": dist["median"],
            "p75": dist["p75"],
            "target_percentile": percentile,
            "target_percentile_label": _ordinal(percentile),
        })
    return rows


def _implied_valuation(
    target_config: dict, imputation_medians: dict, target_business_model: str | None, valuation_multiples: dict,
) -> dict:
    """
    Translates the Top-N's valuation-multiple distribution (already computed
    by _valuation_multiple_distribution) into an implied enterprise value
    range for the target — the step the rest of Section 4 builds up to but
    previously never stated outright. target_ebitda_margin follows the same
    real-input-vs-peer-median-fallback rule as _target_estimate(); revenue
    has no fallback since target_config.revenue_usd_mm is a required input.
    """
    target_revenue = target_config.get("revenue_usd_mm")
    target_ebitda_margin = target_config.get("ebitda_margin_estimate")
    target_ebitda_margin_is_estimated = target_ebitda_margin is None
    if target_ebitda_margin is None:
        target_ebitda_margin = feature_builder.median_for(imputation_medians, "ebitda_margin", target_business_model)

    target_ebitda = (
        target_revenue * target_ebitda_margin
        if target_revenue is not None and target_ebitda_margin is not None
        else None
    )

    by_ebitda = None
    if target_ebitda is not None:
        ev_ebitda = valuation_multiples["ev_ebitda"]
        by_ebitda = {k: ev_ebitda[k] * target_ebitda for k in ("p25", "median", "p75")}

    by_revenue = None
    if target_revenue is not None:
        ev_revenue = valuation_multiples["ev_revenue"]
        by_revenue = {k: ev_revenue[k] * target_revenue for k in ("p25", "median", "p75")}

    # How much the two bases agree at the median — lets the report say
    # whether EV/EBITDA and EV/Revenue point to the same number (mutual
    # support) or diverge (a flag that the comp set's margin profile isn't
    # representative of the target's), instead of showing both ranges with
    # no comment on whether they're consistent.
    median_convergence_pct = None
    if by_ebitda is not None and by_revenue is not None:
        midpoint = (by_ebitda["median"] + by_revenue["median"]) / 2
        if midpoint:
            median_convergence_pct = abs(by_ebitda["median"] - by_revenue["median"]) / midpoint * 100

    # Median agreement alone overstates how much the two bases corroborate
    # each other if one has a much wider P25-P75 spread — a close median
    # next to a much noisier range isn't "mutual support", it's one basis
    # carrying more uncertainty than the other.
    revenue_basis_is_wider = None
    if by_ebitda is not None and by_revenue is not None:
        ebitda_width = by_ebitda["p75"] - by_ebitda["p25"]
        revenue_width = by_revenue["p75"] - by_revenue["p25"]
        revenue_basis_is_wider = ebitda_width > 0 and revenue_width > ebitda_width * 1.2

    return {
        "target_revenue": target_revenue,
        "target_ebitda": target_ebitda,
        "target_ebitda_margin_is_estimated": target_ebitda_margin_is_estimated,
        "by_ebitda": by_ebitda,
        "by_revenue": by_revenue,
        "median_convergence_pct": median_convergence_pct,
        "revenue_basis_is_wider": revenue_basis_is_wider,
    }


def _discounted_valuation(implied_valuation: dict, discount: float) -> dict | None:
    """
    Applies a single net private-company / size-marketability haircut to the
    comp-derived implied EV range. Public trading comps are large-cap,
    liquid, minority-interest multiples; applied straight to a small private
    mid-market target they overstate value (the dominant effects at this
    size are an illiquidity/size discount that a control premium only
    partially offsets — see ValuationConfig.size_marketability_discount).
    Returns None when no discount is configured (the raw comp range stands
    on its own) or when there's no implied range to discount. Mirrors the
    by_ebitda/by_revenue shape of _implied_valuation so the template can
    render it with the same markup.
    """
    if not discount or discount <= 0:
        return None
    factor = 1.0 - discount
    out: dict = {"discount": discount, "factor": factor}
    for basis in ("by_ebitda", "by_revenue"):
        block = implied_valuation.get(basis)
        out[basis] = {k: block[k] * factor for k in ("p25", "median", "p75")} if block else None
    if out["by_ebitda"] is None and out["by_revenue"] is None:
        return None
    return out


def _size_anchor(strict_size_screen: dict | None, target_ebitda: float | None) -> dict | None:
    """
    Turns the strictest size screen (comps genuinely close to the target's
    own revenue — see _strict_size_screen) into a single implied-EV anchor
    for the executive summary. A target this small should be anchored to
    size-comparable comps, not to the full Top-N range that's dominated by
    much larger companies; surfacing that anchor up top — rather than
    leaving it buried as a one-line footnote in Section 4 — is the direct
    answer to the headline scale-mismatch caveat. Point estimate only (the
    sample is too small for a defensible P25/P75), with the comp count
    carried through so the report can disclose how thin it is.
    """
    if not strict_size_screen or target_ebitda is None:
        return None
    median_multiple = strict_size_screen.get("median_ev_ebitda")
    if median_multiple is None:
        return None
    return {
        "n": strict_size_screen.get("n"),
        "tickers": strict_size_screen.get("tickers", []),
        "median_ev_ebitda": median_multiple,
        "implied_ev": median_multiple * target_ebitda,
    }


def _executive_summary(
    top_n: int, implied_valuation: dict, comp_fit_review: dict, implied_valuation_excl_flagged: dict | None,
    discounted_valuation: dict | None = None, size_anchor: dict | None = None,
) -> dict:
    """
    Front-loads the "bottom line" an IC reader looks for before reading the
    rest of the report's methodology — implied valuation range, overall fit
    score, the single biggest caveat, and (when available) the sensitivity
    range excluding flagged comps. Deliberately excludes the size-adjusted
    regression estimate (see Section 4) — its R² is typically too weak to
    sit next to the other two ranges on equal footing, and surfacing it in
    a one-paragraph "bottom line" gives a low-confidence number more visual
    weight than it can support.
    """
    implied_range = implied_valuation.get("by_ebitda") or implied_valuation.get("by_revenue")
    fit_available = comp_fit_review.get("status") == "available"

    excl_flagged_range = None
    if implied_valuation_excl_flagged:
        excl_flagged_range = implied_valuation_excl_flagged.get("by_ebitda") or implied_valuation_excl_flagged.get("by_revenue")

    # Removing flagged comps changes which percentile values land at P25/P75
    # — on a set this small, that reshuffling isn't guaranteed to pull both
    # ends inward (e.g. dropping a low-multiple comp can lower the new P25
    # below the original one even as P75 holds steady). "Narrows" is only
    # accurate when the resulting range is actually tighter; otherwise say
    # "changes" so the claim matches the numbers shown right next to it.
    excl_flagged_is_narrower = None
    if implied_range and excl_flagged_range:
        original_width = implied_range["p75"] - implied_range["p25"]
        excl_flagged_width = excl_flagged_range["p75"] - excl_flagged_range["p25"]
        excl_flagged_is_narrower = excl_flagged_width < original_width

    # The discount applies to whichever basis the headline range uses, so
    # the adjusted low/high stay on the same EV/EBITDA-or-EV/Revenue footing
    # as the raw headline rather than mixing bases.
    discounted_range = None
    discount_fraction = discounted_valuation.get("discount") if discounted_valuation else None
    if discounted_valuation:
        discounted_range = discounted_valuation.get("by_ebitda") or discounted_valuation.get("by_revenue")

    # The size anchor is a comp-implied (public, undiscounted) figure; for a
    # small private target the same private-company haircut that applies to the
    # full-set range applies to it too. Carry a discounted anchor so the
    # headline can lead with the decision-relevant (post-discount) number and
    # cite the pre-discount anchor for transparency, rather than presenting the
    # undiscounted figure as the bottom line.
    size_anchor_ev = size_anchor.get("implied_ev") if size_anchor else None
    size_anchor_ev_discounted = (
        size_anchor_ev * (1 - discount_fraction)
        if (size_anchor_ev is not None and discount_fraction is not None)
        else None
    )

    return {
        "n_comps": top_n,
        "implied_ev_low": implied_range["p25"] if implied_range else None,
        "implied_ev_high": implied_range["p75"] if implied_range else None,
        "implied_basis": "EV/EBITDA" if implied_valuation.get("by_ebitda") else "EV/Revenue",
        "excl_flagged_ev_low": excl_flagged_range["p25"] if excl_flagged_range else None,
        "excl_flagged_ev_high": excl_flagged_range["p75"] if excl_flagged_range else None,
        "excl_flagged_is_narrower": excl_flagged_is_narrower,
        "discount_pct": discounted_valuation.get("discount") * 100 if discounted_valuation else None,
        "adjusted_ev_low": discounted_range["p25"] if discounted_range else None,
        "adjusted_ev_high": discounted_range["p75"] if discounted_range else None,
        "size_anchor_ev": size_anchor_ev,
        "size_anchor_ev_discounted": size_anchor_ev_discounted,
        "size_anchor_n": size_anchor.get("n") if size_anchor else None,
        "size_anchor_multiple": size_anchor.get("median_ev_ebitda") if size_anchor else None,
        "fit_score": comp_fit_review.get("overall_score") if fit_available else None,
        "fit_label": comp_fit_review.get("fit_label") if fit_available else None,
        "top_caveat": _select_top_caveat(comp_fit_review.get("weaknesses")) if fit_available else None,
    }


def _net_debt_usd_mm(company: dict) -> float | None:
    """net_debt_usd_mm wasn't persisted by fetcher.py until recently, so
    cached records fetched before that change won't have it — but it's
    exactly recoverable from two fields that were always cached:
    enterprise_value_usd_mm = market_cap_usd_mm + net_debt_usd_mm. Avoids
    needing to invalidate the fetch cache (and re-hit FMP/EDGAR) just to
    backfill a number that's already implied by what's on disk.
    """
    net_debt = company.get("net_debt_usd_mm")
    if net_debt is not None:
        return net_debt
    ev = company.get("enterprise_value_usd_mm")
    market_cap = company.get("market_cap_usd_mm")
    if ev is not None and market_cap is not None:
        return ev - market_cap
    return None


def _top15_table(companies_by_ticker: dict, llm_features: dict, company_scores: pd.DataFrame, top15: list[str]) -> list[dict]:
    rows = []
    for rank, ticker in enumerate(top15, start=1):
        company = companies_by_ticker.get(ticker, {})
        llm = llm_features.get(ticker, {})
        scores = company_scores.loc[ticker]
        rows.append({
            "rank": rank,
            "ticker": ticker,
            "company_name": company.get("company_name", ticker),
            "ev_ebitda_actual": float(scores["ev_ebitda_actual"]),
            "residual_abs": float(scores["residual_abs"]),
            "ebitda_margin": company.get("ebitda_margin"),
            "gross_margin": company.get("gross_margin"),
            "revenue_ttm_usd_mm": company.get("revenue_ttm_usd_mm"),
            "revenue_cagr_3yr": company.get("revenue_cagr_3yr"),
            "net_debt_ebitda": company.get("net_debt_ebitda"),
            "business_model": llm.get("business_model"),
            "customer_type": llm.get("customer_type"),
            "capital_intensity": llm.get("capital_intensity"),
            "sub_sector_description": llm.get("sub_sector_description"),
            "judge_score": llm.get("judge_score"),
            "low_confidence_flag": bool(llm.get("low_confidence_flag", False)),
            "fit_flag": None,
            "market_cap_usd_mm": company.get("market_cap_usd_mm"),
            "net_debt_usd_mm": _net_debt_usd_mm(company),
            "ebitda_usd_mm": company.get("ebitda_usd_mm"),
            "enterprise_value_usd_mm": company.get("enterprise_value_usd_mm"),
            "ev_revenue": company.get("ev_revenue"),
            "ev_ebit": company.get("ev_ebit"),
            "description_source": company.get("description_source"),
        })
    return rows


# Numeric columns shown in the Section 2 comps matrix that also get a median
# footer row — the standard "median" line at the bottom of a comps tearsheet,
# so the analyst reads central tendency without eyeballing the column.
TOP15_MEDIAN_FIELDS = (
    "revenue_ttm_usd_mm", "ev_ebitda_actual", "ev_revenue", "ev_ebit",
    "ebitda_margin", "gross_margin", "revenue_cagr_3yr", "net_debt_ebitda",
)


def _top15_medians(rows: list[dict]) -> dict:
    medians = {}
    for field in TOP15_MEDIAN_FIELDS:
        values = [row[field] for row in rows if row.get(field) is not None]
        medians[field] = float(np.median(values)) if values else None
    return medians


def _review_candidate_payload(row: dict, candidate_type: str, reasons: list[str] | None = None) -> dict:
    return {
        "candidate_type": candidate_type,
        "rank": row.get("rank"),
        "ticker": row.get("ticker"),
        "company_name": row.get("company_name"),
        "ev_ebitda": row.get("ev_ebitda_actual"),
        "financial_distance": row.get("residual_abs"),
        "ebitda_margin": row.get("ebitda_margin"),
        "gross_margin": row.get("gross_margin"),
        "revenue_ttm_usd_mm": row.get("revenue_ttm_usd_mm"),
        "revenue_cagr_3yr": row.get("revenue_cagr_3yr"),
        "net_debt_ebitda": row.get("net_debt_ebitda"),
        "business_model": row.get("business_model"),
        "customer_type": row.get("customer_type"),
        "capital_intensity": row.get("capital_intensity"),
        "sub_sector_description": row.get("sub_sector_description"),
        "extraction_judge_score": row.get("judge_score"),
        "low_confidence_flag": row.get("low_confidence_flag"),
        "near_miss_reasons": reasons or [],
    }


def _near_miss_review_payload(audit_trail: list[dict], companies_by_ticker: dict, llm_features: dict, company_scores: pd.DataFrame) -> list[dict]:
    rows = []
    for audit in audit_trail:
        ticker = audit["ticker"]
        company = companies_by_ticker.get(ticker, {})
        llm = llm_features.get(ticker, {})
        score = company_scores.loc[ticker] if ticker in company_scores.index else {}
        row = {
            "ticker": ticker,
            "company_name": company.get("company_name", ticker),
            "ev_ebitda_actual": float(score["ev_ebitda_actual"]) if ticker in company_scores.index else None,
            "residual_abs": float(score["residual_abs"]) if ticker in company_scores.index else None,
            "ebitda_margin": company.get("ebitda_margin"),
            "gross_margin": company.get("gross_margin"),
            "revenue_ttm_usd_mm": company.get("revenue_ttm_usd_mm"),
            "revenue_cagr_3yr": company.get("revenue_cagr_3yr"),
            "net_debt_ebitda": company.get("net_debt_ebitda"),
            "business_model": llm.get("business_model"),
            "customer_type": llm.get("customer_type"),
            "capital_intensity": llm.get("capital_intensity"),
            "sub_sector_description": llm.get("sub_sector_description"),
            "judge_score": llm.get("judge_score"),
            "low_confidence_flag": bool(llm.get("low_confidence_flag", False)),
        }
        rows.append(_review_candidate_payload(row, "near_miss", audit.get("reasons")))
    return rows


def _model_diagnostics(
    scorer_results: dict, n_scored_companies: int, top15: list[str], eligible_candidates: list[str],
) -> dict:
    # Keep this section focused on diagnostics computed from the current run.
    company_scores = scorer_results["company_scores"]
    mean_distance = float(company_scores.loc[top15, "residual_abs"].mean()) if top15 else 0.0

    breakdown = scorer.feature_distance_breakdown(scorer_results["feature_distance_sq_diff"], top15)
    top_features = breakdown.to_dict(orient="records")
    for row in top_features:
        row["label"] = _humanize_feature_name(row["feature"])

    return {
        "mean_distance": mean_distance,
        "n_scored_companies": n_scored_companies,
        "top_features": top_features,
        "relative_dispersion": _relative_dispersion(company_scores, eligible_candidates, top15),
    }


def _data_notes(llm_features: dict) -> dict:
    low_confidence_count = sum(1 for v in llm_features.values() if v.get("low_confidence_flag"))

    failed_fetch_count = 0
    if FAILED_TICKERS_PATH.exists():
        with open(FAILED_TICKERS_PATH, encoding="utf-8") as f:
            failed_fetch_count = max(sum(1 for _ in f) - 1, 0)  # minus header row

    return {
        "low_confidence_count": low_confidence_count,
        "failed_fetch_count": failed_fetch_count,
        "disclaimer": DISCLAIMER,
    }


def _selection_summary(top15: list[str], eligible_candidates: list[str], model_diagnostics: dict, data_notes: dict) -> dict:
    dispersion = model_diagnostics["relative_dispersion"]
    return {
        "n_top_comps": len(top15),
        "n_scored_companies": model_diagnostics["n_scored_companies"],
        "n_eligible_candidates": len(eligible_candidates),
        "n_dispersion_pool": dispersion["n_pool"],
        "dispersion_ratio": dispersion["ratio"],
        "low_confidence_count": data_notes["low_confidence_count"],
        "failed_fetch_count": data_notes["failed_fetch_count"],
    }


def _write_csv(rows: list[dict]) -> str:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in CSV_COLUMNS})
    return str(CSV_PATH)


def _format_usd_mm(value: float | None) -> str:
    """Rounds to the nearest whole $mm for display. Deliberately uses
    Python's round() (returns int for a float input with no ndigits) rather
    than "%.0f" % value or Jinja's `round` filter (which both keep the
    result as a float) — formatting a small negative float like -0.013
    that way prints "-0", which reads as a data error rather than what it
    actually is (a value close enough to zero to round there)."""
    if value is None:
        return "N/A"
    return str(int(round(value)))


def _render_html(context: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), trim_blocks=True, lstrip_blocks=True)
    env.filters["usd_mm"] = _format_usd_mm
    template = env.get_template("report.html")
    return template.render(**context)


def _write_html(context: dict) -> str:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    html_text = _render_html(context)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_text)
    return str(HTML_PATH)


def _top_n_from_config(config: PipelineConfig) -> int:
    configured = config.output.top_n_comps
    return int(configured) if configured else TOP_N


def _report_formats_from_config(config: PipelineConfig) -> list[str]:
    formats = config.output.report_formats or list(DEFAULT_REPORT_FORMATS)
    unsupported = sorted(set(formats) - SUPPORTED_REPORT_FORMATS)
    if unsupported:
        raise ValueError(f"Unsupported report format(s): {', '.join(unsupported)}")
    return formats


def generate(
    scorer_results: dict,
    companies: list[dict],
    llm_features: dict[str, dict],
    target_llm_features: dict,
    imputation_medians: dict,
    config: PipelineConfig | dict,
) -> dict:
    """
    Select Top 15 comps and generate reports.

    target_llm_features supplies the target business_model for soft penalties;
    imputation_medians supplies target estimates for financial-benchmark fields
    not specified directly in config.yaml.

    Returns:
        {"csv": "outputs/comps_report.csv", "html": "outputs/comps_report.html"}
    """
    cfg = as_config(config)
    companies_by_ticker = {c["ticker"]: c for c in companies}
    company_scores = scorer_results["company_scores"]
    # target_config and penalties stay plain dicts: both are threaded into a deep
    # tree of valuation/benchmark/selection helpers, and penalties is the dict
    # currency reporter shares with eval/evaluator and the selection tests — so
    # they're flattened from the validated model here rather than retyping that
    # whole surface. Every other config read below goes through the typed model.
    target_config = cfg.target_company.model_dump()

    target_business_model = target_llm_features.get("business_model")
    target_customer_type = target_llm_features.get("customer_type")
    target_revenue = target_config.get("revenue_usd_mm")
    top_n = _top_n_from_config(cfg)

    penalties = cfg.scorer.ranking_penalties.model_dump()
    embedding_model = cfg.llm.embedding_model

    eligible_candidates = _eligible_candidates(company_scores, llm_features, companies_by_ticker)
    subsector_similarities = _subsector_similarities(
        target_llm_features.get("sub_sector_description"), llm_features, eligible_candidates, embedding_model,
    )

    top15 = _select_top_15(
        company_scores, llm_features, companies_by_ticker,
        target_business_model, target_customer_type, target_revenue, subsector_similarities, penalties, k=top_n,
    )

    top15_rows = _top15_table(companies_by_ticker, llm_features, company_scores, top15)
    audit_trail = _audit_trail(
        company_scores, llm_features, companies_by_ticker,
        target_business_model, target_customer_type, target_revenue, top_n, subsector_similarities, penalties,
    )
    model_diagnostics = _model_diagnostics(scorer_results, len(company_scores), top15, eligible_candidates)
    data_notes = _data_notes(llm_features)
    top_review_payload = [_review_candidate_payload(row, "selected") for row in top15_rows]
    near_miss_review_payload = _near_miss_review_payload(audit_trail, companies_by_ticker, llm_features, company_scores)
    comp_fit_review = comp_fit_reviewer.review_comp_fit(target_config, top_review_payload, near_miss_review_payload, cfg)
    comp_fit_review["fit_label"] = (
        _fit_label(comp_fit_review.get("overall_score"), comp_fit_review.get("weaknesses"))
        if comp_fit_review.get("status") == "available" else None
    )
    if comp_fit_review.get("status") == "available":
        # Worst/best first: a reader scanning "Weaker Fits to Review" should
        # see the most concerning comp at the top, not whatever order the
        # LLM happened to return.
        comp_fit_review["top_fits"] = sorted(comp_fit_review.get("top_fits", []), key=lambda r: r.get("score") or 0, reverse=True)
        comp_fit_review["questionable_fits"] = sorted(comp_fit_review.get("questionable_fits", []), key=lambda r: r.get("score") or 0)
        comp_fit_review["near_miss_upgrades"] = sorted(comp_fit_review.get("near_miss_upgrades", []), key=lambda r: r.get("score") or 0, reverse=True)

    # Tag each Top-N row with how the Comparable Fit Review judged it, so
    # Section 2's table and Section 3's strongest/weaker-fit call-outs are
    # visually linked instead of requiring the reader to cross-reference
    # tickers by hand.
    strong_tickers = {row.get("ticker") for row in comp_fit_review.get("top_fits", [])}
    weak_tickers = {row.get("ticker") for row in comp_fit_review.get("questionable_fits", [])}
    # Statistical outliers are flagged independently of the LLM's fit
    # judgment (see _ev_ebitda_outlier_tickers) — a comp can be a good
    # qualitative match and still be a multiple outlier, or vice versa.
    outlier_tickers = _ev_ebitda_outlier_tickers(company_scores, top15)
    flagged_tickers = weak_tickers | outlier_tickers
    for row in top15_rows:
        if row["ticker"] in strong_tickers:
            row["fit_flag"] = "strong"
        elif row["ticker"] in weak_tickers:
            row["fit_flag"] = "weak"
        row["outlier_flag"] = row["ticker"] in outlier_tickers
        row["tier"] = _assign_tier(row["fit_flag"], row["outlier_flag"])

    tier_summary = [
        {
            "tier": tier,
            "label": label,
            "n": sum(1 for row in top15_rows if row["tier"] == tier),
            "tickers": [
                # Review/Exclude mixes two independent checks (LLM
                # qualitative judgment and the statistical outlier test —
                # see _ev_ebitda_outlier_tickers); without this annotation
                # a reader sees e.g. AAON in this group and goes looking
                # for it in the qualitative Weaker Fits table, where it
                # won't be if it only tripped the outlier check.
                (
                    f"{row['ticker']} ("
                    + ("weaker fit + outlier" if row["fit_flag"] == "weak" and row["outlier_flag"]
                       else "weaker fit" if row["fit_flag"] == "weak"
                       else "statistical outlier")
                    + ")"
                )
                if tier == "review_exclude" else row["ticker"]
                for row in top15_rows if row["tier"] == tier
            ],
        }
        for tier, label in TIER_LABELS.items()
    ]

    valuation_multiples = _valuation_multiple_distribution(company_scores, companies_by_ticker, top15)
    implied_valuation = _implied_valuation(target_config, imputation_medians, target_business_model, valuation_multiples)

    # Section 3 flags certain comps as weaker fits, and the EV/EBITDA
    # distribution itself flags statistical outliers — but the Implied
    # Enterprise Value table otherwise treats all Top-N comps as equally
    # weighted. This sensitivity range shows what the valuation looks like
    # with both kinds of flagged comps dropped, so the qualitative review
    # and the outlier check actually inform the quantitative conclusion
    # instead of sitting next to it unreconciled.
    top15_excl_flagged = [t for t in top15 if t not in flagged_tickers]
    implied_valuation_excl_flagged = None
    if flagged_tickers and len(top15_excl_flagged) >= MIN_VALUES_FOR_DISPERSION:
        valuation_multiples_excl_flagged = _valuation_multiple_distribution(company_scores, companies_by_ticker, top15_excl_flagged)
        implied_valuation_excl_flagged = _implied_valuation(
            target_config, imputation_medians, target_business_model, valuation_multiples_excl_flagged,
        )

    size_adjusted_valuation = _size_adjusted_valuation(
        company_scores, companies_by_ticker, eligible_candidates, target_revenue, implied_valuation.get("target_ebitda"),
    )
    revenue_screened_valuation = _revenue_screened_valuation(
        company_scores, companies_by_ticker, top15, target_config, imputation_medians, target_business_model,
    )
    strict_size_screen = _strict_size_screen(company_scores, companies_by_ticker, top15, target_revenue)
    scatter_data = _revenue_multiple_scatter_svg(company_scores, companies_by_ticker, eligible_candidates, top15_rows, target_revenue)
    scale_reconciliation_note = _scale_reconciliation_note(companies_by_ticker, top15, target_revenue)

    # Private-company / size-marketability adjustment: the comp-derived
    # ranges above are large-cap, liquid, minority trading multiples; a
    # small private target is realistically worth less than they imply. The
    # discount is config-driven (default 0.0 = no-op), applied to both the
    # headline range and the flagged-excluded range so the report can show
    # an adjusted range next to every raw one. size_anchor surfaces the
    # strictest size-comparable subset as the executive summary's anchor.
    discount = cfg.valuation.size_marketability_discount or 0.0
    discounted_valuation = _discounted_valuation(implied_valuation, discount)
    discounted_valuation_excl_flagged = (
        _discounted_valuation(implied_valuation_excl_flagged, discount) if implied_valuation_excl_flagged else None
    )
    size_anchor = _size_anchor(strict_size_screen, implied_valuation.get("target_ebitda"))
    football_field = _football_field_svg(
        implied_valuation, revenue_screened_valuation, discounted_valuation, size_adjusted_valuation, size_anchor,
    )

    # Company-specific caution: Top-N comps whose depressed EBITDA margin
    # likely distorts their own multiple (see LOW_MARGIN_CAUTION_THRESHOLD),
    # so a reader doesn't take e.g. a 4.9x next to a low margin as a clean
    # market multiple.
    low_margin_comps = [
        {"ticker": row["ticker"], "ebitda_margin": row["ebitda_margin"]}
        for row in top15_rows
        if row.get("ebitda_margin") is not None and row["ebitda_margin"] < LOW_MARGIN_CAUTION_THRESHOLD
    ]

    # The Data Appendix only earns its "Description Source" column when
    # sources actually differ across the Top-N — with one fetch run all
    # landing on EDGAR (the common case), a column that's the same value
    # 15 times running is dead weight; collapse it into a single sentence
    # instead, the same treatment "Valuation Source" got (see its removal
    # above the Top-N row dicts) once it was found to never vary at all.
    description_sources = {row.get("description_source") for row in top15_rows if row.get("description_source")}
    description_sources_vary = len(description_sources) > 1
    description_source_common = next(iter(description_sources)) if len(description_sources) == 1 else None

    context = {
        "target_name": target_config.get("name"),
        "target_description": target_config.get("description"),
        "valuation_multiples": valuation_multiples,
        "implied_valuation": implied_valuation,
        "implied_valuation_excl_flagged": implied_valuation_excl_flagged,
        "flagged_tickers": sorted(flagged_tickers),
        "size_adjusted_valuation": size_adjusted_valuation,
        "revenue_screened_valuation": revenue_screened_valuation,
        "strict_size_screen": strict_size_screen,
        "scatter_data": scatter_data,
        "football_field": football_field,
        "scale_reconciliation_note": scale_reconciliation_note,
        "description_sources_vary": description_sources_vary,
        "description_source_common": description_source_common,
        "tier_summary": tier_summary,
        "TIER_LABELS": TIER_LABELS,
        "DESCRIPTION_SOURCE_LABELS": DESCRIPTION_SOURCE_LABELS,
        "executive_summary": _executive_summary(
            top_n, implied_valuation, comp_fit_review, implied_valuation_excl_flagged,
            discounted_valuation, size_anchor,
        ),
        "discounted_valuation": discounted_valuation,
        "discounted_valuation_excl_flagged": discounted_valuation_excl_flagged,
        "size_anchor": size_anchor,
        "discount_note": cfg.valuation.discount_note,
        "low_margin_comps": low_margin_comps,
        "financial_benchmarks": _financial_benchmarks(
            companies_by_ticker, top15, target_config, imputation_medians, target_business_model,
        ),
        "top15": top15_rows,
        "top15_medians": _top15_medians(top15_rows),
        "audit_trail": audit_trail,
        "model_diagnostics": model_diagnostics,
        "data_notes": data_notes,
        "selection_summary": _selection_summary(top15, eligible_candidates, model_diagnostics, data_notes),
        "comp_fit_review": comp_fit_review,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "n_comps": len(top15_rows),
        "tukey_fence_multiplier": TUKEY_FENCE_MULTIPLIER,
        "meaningful_narrowing_ratio": MEANINGFUL_NARROWING_RATIO,
        "modest_narrowing_ratio": MODEST_NARROWING_RATIO,
        "wide_multiple_spread_ratio": WIDE_MULTIPLE_SPREAD_RATIO,
        "SIZE_BAND_MULTIPLE": SIZE_BAND_MULTIPLE,
        "STRICT_SIZE_BAND_LOWER_MULTIPLE": STRICT_SIZE_BAND_LOWER_MULTIPLE,
        "STRICT_SIZE_BAND_UPPER_MULTIPLE": STRICT_SIZE_BAND_UPPER_MULTIPLE,
        "prepared_by": cfg.output.prepared_by,
        "confidential": cfg.output.confidential,
    }

    output_paths = {}
    report_formats = _report_formats_from_config(cfg)
    if "csv" in report_formats:
        output_paths["csv"] = _write_csv(top15_rows)
    if "html" in report_formats:
        output_paths["html"] = _write_html(context)
    output_paths["n_comps"] = len(top15_rows)

    return output_paths
