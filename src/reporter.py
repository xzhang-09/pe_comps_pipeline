import csv
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from src import comp_fit_reviewer, get_logger, llm_analyzer, provenance, scorer
from src.config_schema import PipelineConfig, as_config
from src.paths import project_path
from src.records import CompanyRecord

# report_selection owns Top-N ranking semantics, report_valuation owns pure
# valuation math, and report_charts owns SVG builders. This module keeps report
# assembly: context building, review scoping, CSV/HTML writing, and provenance.
# Shared helpers are re-imported below (and listed in __all__ as intentional
# re-exports) so existing callers can keep addressing reporter.<name>.
from src.report_charts import (
    _football_field_rows,
    _football_field_svg,
    _revenue_multiple_scatter_svg,
)
from src.report_selection import (
    AUDIT_SIZE,
    EXEMPT_BUSINESS_MODELS,
    EXEMPT_CUSTOMER_TYPES,
    TIER_LABELS,
    TIER_ORDER,
    TOP_N,
    _assign_tier,
    _audit_trail,
    _eligible_candidates,
    _penalty_breakdown,
    _ranked_candidates,
    _select_top_15,
    _size_mismatch_penalty,
    _subsector_similarities,
)
from src.report_valuation import (
    ABSOLUTE_OUTLIER_MEDIAN_MULTIPLE,
    BENCHMARK_METRICS,
    LOW_MARGIN_CAUTION_THRESHOLD,
    MEANINGFUL_NARROWING_RATIO,
    MIN_POOL_FOR_SIZE_REGRESSION,
    MIN_VALUES_FOR_DISPERSION,
    MODEST_NARROWING_RATIO,
    SIZE_BAND_MULTIPLE,
    STRICT_SIZE_BAND_LOWER_MULTIPLE,
    STRICT_SIZE_BAND_UPPER_MULTIPLE,
    TUKEY_FENCE_MULTIPLIER,
    WIDE_MULTIPLE_SPREAD_RATIO,
    _discounted_valuation,
    _distribution_stats,
    _ev_ebitda_outlier_tickers,
    _financial_benchmarks,
    _implied_valuation,
    _iqr,
    _multiple_values,
    _ordinal,
    _relative_dispersion,
    _revenue_screened_valuation,
    _size_adjusted_valuation,
    _size_anchor,
    _strict_size_screen,
    _target_estimate,
    _valuation_multiple_distribution,
)

__all__ = [
    # kept here
    "generate",
    # re-exports: report_charts
    "_football_field_rows", "_football_field_svg", "_revenue_multiple_scatter_svg",
    # re-exports: report_selection
    "AUDIT_SIZE", "EXEMPT_BUSINESS_MODELS", "EXEMPT_CUSTOMER_TYPES", "TIER_LABELS",
    "TIER_ORDER", "TOP_N", "_assign_tier", "_audit_trail", "_eligible_candidates",
    "_penalty_breakdown", "_ranked_candidates", "_select_top_15",
    "_size_mismatch_penalty", "_subsector_similarities",
    # re-exports: report_valuation
    "ABSOLUTE_OUTLIER_MEDIAN_MULTIPLE", "BENCHMARK_METRICS", "LOW_MARGIN_CAUTION_THRESHOLD",
    "MEANINGFUL_NARROWING_RATIO", "MIN_POOL_FOR_SIZE_REGRESSION", "MIN_VALUES_FOR_DISPERSION",
    "MODEST_NARROWING_RATIO", "SIZE_BAND_MULTIPLE", "STRICT_SIZE_BAND_LOWER_MULTIPLE",
    "STRICT_SIZE_BAND_UPPER_MULTIPLE", "TUKEY_FENCE_MULTIPLIER", "WIDE_MULTIPLE_SPREAD_RATIO",
    "_discounted_valuation", "_distribution_stats", "_ev_ebitda_outlier_tickers",
    "_financial_benchmarks", "_implied_valuation", "_iqr", "_multiple_values", "_ordinal",
    "_relative_dispersion", "_revenue_screened_valuation", "_size_adjusted_valuation",
    "_size_anchor", "_strict_size_screen", "_target_estimate", "_valuation_multiple_distribution",
    # module attributes tests patch (e.g. src.reporter.llm_analyzer.embed_texts)
    "llm_analyzer",
]

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


# fetcher.py's description_source values, spelled out for the report —
# "edgar" alone doesn't tell a reader whether that means structured EDGAR
# financial data or just the filing's free-text business description (it's
# the latter; see fetcher._fetch_business_description).
DESCRIPTION_SOURCE_LABELS = {
    "edgar": "SEC EDGAR (filing text)",
    "fmp": "FMP company profile",
}


# A score that just clears the "Strong" threshold next to a weakness the
# LLM itself called material shouldn't read as confidently as a score that
# clears it cleanly — downgrade_band is the range where one materially
# worded weakness is enough to pull the label down a tier; above it, a
# single weakness isn't treated as disqualifying.
FIT_LABEL_DOWNGRADE_BAND = (80, 90)
CAVEAT_SEVERITY_KEYWORDS = ("significant", "material", "substantial")
# Core comps must be within one order of magnitude of the target's revenue
# (|log10(candidate/target)| <= 1.0, i.e. 10x either way). The size penalty
# already demotes bigger comps in the ranking, but without a cap a $2bn
# company could still carry the "Core" label against a $150mm target — a
# label an IC reader takes as "use this multiple as-is". Beyond 10x it stays
# usable but is capped at Secondary. Comps with no revenue data are not
# size-blocked (the missing-data signal is handled by low-confidence checks).
CORE_MAX_LOG10_REVENUE_GAP = 1.0
# Catches phrasing like "10x target size" or "5.5x larger" — a concrete
# multiple is at least as strong a severity signal as the literal word
# "significant", and LLM-written weaknesses don't reliably use that word
# even when describing an equally severe mismatch (e.g. "skewed larger
# than target, with some comps >10x target size" — no severity keyword,
# but >10x is a stronger claim than "significant" on its own).
SCALE_MAGNITUDE_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*x\b")
TICKER_REFERENCE_PATTERN = re.compile(r"\b[A-Z]{2,5}\b")
NARRATIVE_TICKER_ALLOWLIST = {
    "AI",
    "B2B",
    "B2C",
    "EBITDA",
    "EV",
    "FMP",
    "LLM",
    "OEM",
    "SEC",
}

VALUATION_ROLE_LABELS = {
    "valuation_anchor": "Valuation Anchor",
    "broad_reference": "Broad Reference",
    "review_sensitivity": "Review / Sensitivity Only",
}


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


def _has_material_fit_caveat(weaknesses: list[str] | None) -> bool:
    if not weaknesses:
        return False
    joined = " ".join(weaknesses).lower()
    material_terms = (
        "significant",
        "weak for several",
        "several comps",
        "scale mismatch",
        "revenue scale",
        "end-market fit is weak",
        "customer type mismatch",
        "review / exclude",
    )
    return any(term in joined for term in material_terms)


def _fit_quality_diagnostics(
    rows: list[dict],
    target_revenue: float | int | None,
    model_diagnostics: dict | None,
) -> dict[str, float | int | None]:
    total = len(rows)
    review_exclude_count = sum(1 for row in rows if row.get("tier") == "review_exclude")
    secondary_count = sum(1 for row in rows if row.get("tier") == "secondary")
    review_exclude_share = review_exclude_count / total if total else 0.0
    secondary_or_review_share = (secondary_count + review_exclude_count) / total if total else 0.0

    max_revenue_ratio = None
    if target_revenue:
        ratios = []
        for row in rows:
            revenue = row.get("revenue_ttm_usd_mm")
            if revenue and revenue > 0:
                ratios.append(max(float(revenue) / float(target_revenue), float(target_revenue) / float(revenue)))
        max_revenue_ratio = max(ratios) if ratios else None

    dispersion = None
    if model_diagnostics:
        dispersion = model_diagnostics.get("selected_ev_ebitda_iqr_vs_pool")
        if dispersion is None and isinstance(model_diagnostics.get("relative_dispersion"), dict):
            dispersion = model_diagnostics["relative_dispersion"].get("ratio")

    return {
        "review_exclude_count": review_exclude_count,
        "review_exclude_share": review_exclude_share,
        "secondary_or_review_share": secondary_or_review_share,
        "max_revenue_ratio": max_revenue_ratio,
        "selected_ev_ebitda_iqr_vs_pool": dispersion,
    }


def _selection_quality_summary(rows: list[dict], target_revenue: float | int | None) -> dict[str, float | int | None]:
    revenues = [
        float(row["revenue_ttm_usd_mm"])
        for row in rows
        if row.get("revenue_ttm_usd_mm") is not None and row.get("revenue_ttm_usd_mm") > 0
    ]
    revenue_ratios = []
    if target_revenue:
        revenue_ratios = [
            max(revenue / float(target_revenue), float(target_revenue) / revenue)
            for revenue in revenues
        ]

    return {
        "n_total": len(rows),
        "n_core": sum(1 for row in rows if row.get("tier") == "core"),
        "n_secondary": sum(1 for row in rows if row.get("tier") == "secondary"),
        "n_review_exclude": sum(1 for row in rows if row.get("tier") == "review_exclude"),
        "usable_count": sum(1 for row in rows if row.get("tier") != "review_exclude"),
        "max_revenue_ratio": max(revenue_ratios) if revenue_ratios else None,
        "median_revenue_ratio": float(np.median(revenue_ratios)) if revenue_ratios else None,
    }


def _has_material_quant_caveat(diagnostics: dict[str, float | int | None] | None) -> bool:
    if not diagnostics:
        return False
    review_exclude_share = diagnostics.get("review_exclude_share") or 0
    secondary_or_review_share = diagnostics.get("secondary_or_review_share") or 0
    max_revenue_ratio = diagnostics.get("max_revenue_ratio")
    dispersion = diagnostics.get("selected_ev_ebitda_iqr_vs_pool")
    return (
        review_exclude_share >= 0.25
        or secondary_or_review_share >= 0.65
        or (max_revenue_ratio is not None and max_revenue_ratio >= 5.0)
        or (dispersion is not None and dispersion >= 0.85)
    )


# Thresholds shared by the executive summary and the Comparable Fit Review
# section so both describe the same overall_score with the same words.
def _fit_label(
    score: float | int | None,
    weaknesses: list[str] | None = None,
    diagnostics: dict[str, float | int | None] | None = None,
) -> str | None:
    if score is None:
        return None
    if score >= 80:
        return "Strong"
    if score >= 65:
        if (
            _has_severe_scale_caveat(weaknesses)
            or _has_material_fit_caveat(weaknesses)
            or _has_material_quant_caveat(diagnostics)
        ):
            return "Mixed / directionally useful with material caveats"
        return "Good / directionally supportive"
    if score >= 50:
        return "Mixed / directionally useful with material caveats"
    return "Weak"


OUTPUTS_DIR = project_path("outputs")
CSV_PATH = OUTPUTS_DIR / "comps_report.csv"
HTML_PATH = OUTPUTS_DIR / "comps_report.html"
TEMPLATE_DIR = project_path("src", "templates")
FAILED_TICKERS_PATH = OUTPUTS_DIR / "failed_tickers.csv"

DEFAULT_REPORT_FORMATS = ("csv", "html")
SUPPORTED_REPORT_FORMATS = set(DEFAULT_REPORT_FORMATS)

DISCLAIMER = "Analysis based on public company data via FMP and SEC EDGAR. For reference purposes only."

CSV_COLUMNS = (
    "rank", "ticker", "company_name", "ev_ebitda_actual",
    "residual_abs", "ebitda_margin", "gross_margin", "revenue_ttm_usd_mm",
    "revenue_cagr_3yr", "net_debt_ebitda", "business_model", "customer_type",
    "capital_intensity", "sub_sector_description", "tier", "fit_notes",
    "judge_score", "low_confidence_flag", "profile_incomplete", "candidate_source",
)


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
        candidate_source = company.get("candidate_source") or (company.get("universe_metadata") or {}).get("candidate_source")
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
            "profile_incomplete": bool(llm.get("profile_incomplete", False)),
            "fit_flag": None,
            "market_cap_usd_mm": company.get("market_cap_usd_mm"),
            "net_debt_usd_mm": _net_debt_usd_mm(company),
            "ebitda_usd_mm": company.get("ebitda_usd_mm"),
            "enterprise_value_usd_mm": company.get("enterprise_value_usd_mm"),
            "ev_revenue": company.get("ev_revenue"),
            "ev_ebit": company.get("ev_ebit"),
            "description_source": company.get("description_source"),
            "candidate_source": candidate_source,
            "candidate_source_label": _candidate_source_label(candidate_source),
            "analyst_approved": False,
        })
    return rows


def _candidate_source_label(source: str | None) -> str:
    if source == "analyst_specified":
        return "Analyst-specified"
    return "SEC SIC"


def _must_include_exclusion_notes(cfg: PipelineConfig, top_tickers: list[str], llm_features: dict, company_scores: pd.DataFrame) -> list[str]:
    selected = set(top_tickers)
    notes = []
    for ticker in cfg.universe.must_include_tickers:
        if ticker in selected:
            continue
        llm = llm_features.get(ticker)
        if llm and llm.get("low_confidence_flag"):
            notes.append(f"{ticker} was analyst-specified but excluded by the low-confidence source-support filter.")
        elif ticker not in company_scores.index:
            notes.append(f"{ticker} was analyst-specified but did not reach scoring after fetch/data filters.")
    return notes


def _analyst_approval_notes(cfg: PipelineConfig, top_rows: list[dict]) -> list[str]:
    if not cfg.universe.analyst_approved_tickers:
        return []

    rows_by_ticker = {row["ticker"]: row for row in top_rows}
    notes = []
    for ticker in cfg.universe.analyst_approved_tickers:
        row = rows_by_ticker.get(ticker)
        if row is None:
            notes.append(f"{ticker} was analyst-approved but is not in the selected comp set.")
            continue
        row["analyst_approved"] = True
        notes.append(
            f"{ticker} retained as analyst-approved ({TIER_LABELS[row['tier']]}); model tier/caveats still shown."
        )
    return notes


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


def _business_model_alignment_summary(rows: list[dict], target_business_model: str | None, target_customer_type: str | None) -> dict | None:
    if not rows or (not target_business_model and not target_customer_type):
        return None
    business_matches = [row for row in rows if row.get("business_model") == target_business_model] if target_business_model else []
    business_exceptions = (
        [
            {"ticker": row["ticker"], "business_model": row.get("business_model") or "unknown"}
            for row in rows if row.get("business_model") != target_business_model
        ]
        if target_business_model else []
    )
    customer_matches = [row for row in rows if row.get("customer_type") == target_customer_type] if target_customer_type else []
    customer_exceptions = (
        [
            {"ticker": row["ticker"], "customer_type": row.get("customer_type") or "unknown"}
            for row in rows if row.get("customer_type") != target_customer_type
        ]
        if target_customer_type else []
    )
    fit_note_exceptions = [
        {"ticker": row["ticker"], "tier": row.get("tier"), "fit_notes": row.get("fit_notes")}
        for row in rows
        if row.get("fit_notes") and row.get("fit_notes") != "No material fit flags"
    ]
    return {
        "target_business_model": target_business_model,
        "n_matching": len(business_matches),
        "n_total": len(rows),
        "exceptions": business_exceptions,
        "target_customer_type": target_customer_type,
        "n_customer_matching": len(customer_matches),
        "customer_exceptions": customer_exceptions,
        "fit_note_exceptions": fit_note_exceptions,
    }


def _filter_review_scope(review: dict, selected_tickers: set[str], near_miss_tickers: set[str]) -> dict:
    if review.get("status") != "available":
        return review

    removed_selected = 0
    removed_near_miss = 0

    for key in ("top_fits", "questionable_fits"):
        scoped = []
        for row in review.get(key, []):
            if row.get("ticker") in selected_tickers:
                scoped.append(row)
            else:
                removed_selected += 1
        review[key] = scoped

    scoped_near_misses = []
    for row in review.get("near_miss_upgrades", []):
        if row.get("ticker") in near_miss_tickers:
            scoped_near_misses.append(row)
        else:
            removed_near_miss += 1
    review["near_miss_upgrades"] = scoped_near_misses

    notes = []
    if removed_selected:
        plural = "s" if removed_selected != 1 else ""
        notes.append(
            f"Review scope check removed {removed_selected} selected-comp callout{plural} "
            "that referenced a non-selected ticker."
        )
    if removed_near_miss:
        plural = "s" if removed_near_miss != 1 else ""
        notes.append(
            f"Review scope check removed {removed_near_miss} near-miss callout{plural} "
            "that referenced a selected or unavailable ticker."
        )
    invalid_narrative_tickers = _invalid_review_ticker_references(
        review,
        allowed_tickers=selected_tickers | near_miss_tickers,
    )
    for ticker in invalid_narrative_tickers:
        notes.append(f"Review narrative referenced unavailable ticker {ticker}.")
    review["scope_notes"] = notes
    return review


def _review_narrative_text(review: dict) -> str:
    parts = [str(review.get("summary") or "")]
    for key in ("strengths", "weaknesses"):
        parts.extend(str(item) for item in review.get(key, []) if item)
    return " ".join(parts)


def _invalid_review_ticker_references(review: dict, allowed_tickers: set[str]) -> list[str]:
    candidates = set(TICKER_REFERENCE_PATTERN.findall(_review_narrative_text(review)))
    invalid = candidates - allowed_tickers - NARRATIVE_TICKER_ALLOWLIST
    return sorted(invalid)


def _fit_notes(reasons: list[str], fit_flag: str | None, outlier_flag: bool, review_reason: str | None) -> str:
    notes = list(reasons)
    if fit_flag == "strong":
        notes.append("qualitative review: strongest fit")
    elif fit_flag == "weak":
        notes.append(f"qualitative review: {review_reason}" if review_reason else "qualitative review: weaker fit")
    if outlier_flag:
        notes.append("EV/EBITDA statistical outlier")
    return "; ".join(notes) if notes else "No material fit flags"


def _valuation_role(row: dict) -> str:
    if row.get("tier") == "review_exclude":
        return "review_sensitivity"
    if row.get("tier") == "core":
        return "valuation_anchor"
    return "broad_reference"


def _annotate_valuation_roles(rows: list[dict]) -> dict:
    for row in rows:
        row["valuation_role"] = _valuation_role(row)
        row["valuation_role_label"] = VALUATION_ROLE_LABELS[row["valuation_role"]]

    return {
        role: {
            "label": label,
            "n": sum(1 for row in rows if row.get("valuation_role") == role),
            "tickers": [row["ticker"] for row in rows if row.get("valuation_role") == role],
        }
        for role, label in VALUATION_ROLE_LABELS.items()
    }


def _replacement_rejection_reason(candidate_row: dict) -> str | None:
    if candidate_row["tier"] == "review_exclude":
        return "candidate also classified as Review / Exclude"
    return None


def _secondary_replacement_is_better(candidate_row: dict, review_rows: list[dict]) -> bool:
    if candidate_row["tier"] != "secondary" or not review_rows:
        return False
    if candidate_row.get("fit_flag") == "weak" or candidate_row.get("outlier_flag"):
        return False
    notes = candidate_row.get("fit_notes") or ""
    hard_mismatches = ("business model mismatch", "customer type mismatch")
    return not any(mismatch in notes for mismatch in hard_mismatches)


def _replacement_acceptance_reason(candidate_row: dict, review_rows: list[dict]) -> str | None:
    if candidate_row["tier"] == "core":
        return "accepted as Core replacement for a Review / Exclude slot"
    if _secondary_replacement_is_better(candidate_row, review_rows):
        replaced = ", ".join(row["ticker"] for row in review_rows[:3])
        suffix = f" over {replaced}" if replaced else ""
        return f"accepted as better Secondary replacement{suffix}"
    return None


def _substitution_audit_summary(substitution_audit: list[dict], rejected_limit: int = 5) -> dict | None:
    if not substitution_audit:
        return None
    accepted = [item for item in substitution_audit if item.get("decision") == "accepted"]
    rejected = [item for item in substitution_audit if item.get("decision") == "rejected"]
    displayed_rejected = rejected[:rejected_limit]
    return {
        "n_total": len(substitution_audit),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "displayed": accepted + displayed_rejected,
        "hidden_rejected_count": max(0, len(rejected) - len(displayed_rejected)),
    }


def _annotate_top_rows(
    rows: list[dict],
    company_scores: pd.DataFrame,
    llm_features: dict,
    companies_by_ticker: dict,
    target_business_model: str | None,
    target_customer_type: str | None,
    target_revenue: float | None,
    subsector_similarities: dict[str, float],
    penalties: dict,
    strong_tickers: set[str],
    weak_tickers: set[str],
    review_reasons: dict,
) -> set[str]:
    tickers = [row["ticker"] for row in rows]
    outlier_tickers = _ev_ebitda_outlier_tickers(company_scores, tickers)
    apply_business_model_penalty = target_business_model not in EXEMPT_BUSINESS_MODELS
    apply_customer_type_penalty = target_customer_type not in EXEMPT_CUSTOMER_TYPES

    for row in rows:
        row["fit_flag"] = None
        if row["ticker"] in strong_tickers:
            row["fit_flag"] = "strong"
        elif row["ticker"] in weak_tickers:
            row["fit_flag"] = "weak"
        row["outlier_flag"] = row["ticker"] in outlier_tickers
        penalty = _penalty_breakdown(
            ticker=row["ticker"],
            base_rank=row["rank"],
            residual=float(company_scores.loc[row["ticker"], "residual_abs"]),
            llm_features=llm_features,
            companies_by_ticker=companies_by_ticker,
            target_business_model=target_business_model,
            target_customer_type=target_customer_type,
            target_revenue=target_revenue,
            apply_business_model_penalty=apply_business_model_penalty,
            apply_customer_type_penalty=apply_customer_type_penalty,
            subsector_similarities=subsector_similarities,
            penalties=penalties,
        )
        # Core blockers: any categorical mismatch, any end-market similarity
        # shortfall (the graded penalty is > 0 only below the threshold), or a
        # >10x revenue gap. The graded subsector penalty replaced the old
        # near-threshold + LLM-strong exception: the exception existed to
        # soften a cliff penalty, and with the cliff gone a comp either clears
        # the similarity bar or it doesn't.
        size_blocks_core = (
            penalty["size_log10_gap"] is not None
            and penalty["size_log10_gap"] > CORE_MAX_LOG10_REVENUE_GAP
        )
        has_mismatch = bool(
            penalty["business_model_penalty"]
            or penalty["customer_type_penalty"]
            or penalty["subsector_penalty"]
            or penalty["profile_incomplete"]
            or size_blocks_core
        )
        row["tier"] = _assign_tier(row["fit_flag"], row["outlier_flag"], has_mismatch)
        row["fit_notes"] = _fit_notes(
            penalty["reasons"], row["fit_flag"], row["outlier_flag"], review_reasons.get(row["ticker"])
        )

    return weak_tickers | outlier_tickers


def _fill_usable_comp_slots(
    selected_tickers: list[str],
    top_n: int,
    company_scores: pd.DataFrame,
    llm_features: dict,
    companies_by_ticker: dict,
    target_business_model: str | None,
    target_customer_type: str | None,
    target_revenue: float | None,
    subsector_similarities: dict[str, float],
    penalties: dict,
    strong_tickers: set[str],
    weak_tickers: set[str],
    review_reasons: dict,
) -> tuple[list[dict], list[str], set[str], list[dict]]:
    ranked = _ranked_candidates(
        company_scores, llm_features, companies_by_ticker,
        target_business_model, target_customer_type, target_revenue,
        subsector_similarities, penalties,
        excluded_tickers=None,
        exclude_training=True,
    )
    ranked_tickers = [row["ticker"] for row in ranked]
    selected = list(selected_tickers)
    next_idx = 0
    substitution_audit = []

    while True:
        rows = _top15_table(companies_by_ticker, llm_features, company_scores, selected)
        flagged_tickers = _annotate_top_rows(
            rows, company_scores, llm_features, companies_by_ticker,
            target_business_model, target_customer_type, target_revenue,
            subsector_similarities, penalties,
            strong_tickers, weak_tickers, review_reasons,
        )
        usable_count = sum(1 for row in rows if row["tier"] != "review_exclude")
        if usable_count >= top_n:
            return rows, selected, flagged_tickers, substitution_audit

        review_rows = [row for row in rows if row["tier"] == "review_exclude"]
        selected_set = set(selected)
        accepted_replacement = None
        while next_idx < len(ranked_tickers):
            candidate = ranked_tickers[next_idx]
            next_idx += 1
            if candidate in selected_set:
                continue

            tentative = selected + [candidate]
            tentative_rows = _top15_table(companies_by_ticker, llm_features, company_scores, tentative)
            _annotate_top_rows(
                tentative_rows, company_scores, llm_features, companies_by_ticker,
                target_business_model, target_customer_type, target_revenue,
                subsector_similarities, penalties,
                strong_tickers, weak_tickers, review_reasons,
            )
            candidate_row = next(row for row in tentative_rows if row["ticker"] == candidate)
            rejection_reason = _replacement_rejection_reason(candidate_row)
            if rejection_reason:
                substitution_audit.append({
                    "ticker": candidate,
                    "company_name": candidate_row.get("company_name", candidate),
                    "decision": "rejected",
                    "tier": candidate_row["tier"],
                    "reason": rejection_reason,
                })
                continue

            acceptance_reason = _replacement_acceptance_reason(candidate_row, review_rows)
            if not acceptance_reason:
                substitution_audit.append({
                    "ticker": candidate,
                    "company_name": candidate_row.get("company_name", candidate),
                    "decision": "rejected",
                    "tier": candidate_row["tier"],
                    "reason": f"candidate not clearly better than the Review / Exclude slot: {candidate_row.get('fit_notes')}",
                })
                continue

            accepted_replacement = candidate
            substitution_audit.append({
                "ticker": candidate,
                "company_name": candidate_row.get("company_name", candidate),
                "decision": "accepted",
                "tier": candidate_row["tier"],
                "reason": acceptance_reason,
            })
            break
        if accepted_replacement is None:
            return rows, selected, flagged_tickers, substitution_audit
        selected.append(accepted_replacement)


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
            "profile_incomplete": bool(llm.get("profile_incomplete", False)),
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


def _data_notes(llm_features: dict, companies_by_ticker: dict) -> dict:
    # Count only companies in the current run's universe: llm_features can
    # carry entries from a reloaded artifact, and the report's "excluded for
    # weak source support" figure should not include non-candidates.
    low_confidence_count = sum(
        1 for ticker, v in llm_features.items()
        if ticker in companies_by_ticker and v.get("low_confidence_flag")
    )

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


def _git_commit() -> str:
    """Short HEAD SHA of the code that produced this report, or 'unknown' if git
    isn't available or this isn't a checkout — provenance, never fatal."""
    return provenance.git_commit()


def _config_hash(cfg: PipelineConfig) -> str:
    """Stable 12-char fingerprint of the config that produced this report, so two
    reports can be told apart (or confirmed identical) by inputs at a glance —
    same canonical-JSON-then-sha256 approach comp_fit_reviewer uses for its cache
    key."""
    return provenance.config_hash(cfg)


def _oldest_timestamp(records: list[CompanyRecord], field: str) -> str | None:
    """Oldest value of `field` across records (missing values ignored). ISO 8601
    UTC timestamps sort lexicographically in chronological order, so min() is the
    earliest — i.e. the worst-case staleness of that data layer."""
    stamps = [r.get(field) for r in records if r.get(field)]
    return min(stamps) if stamps else None


def _newest_timestamp(records: list[CompanyRecord], field: str) -> str | None:
    stamps = [r.get(field) for r in records if r.get(field)]
    return max(stamps) if stamps else None


def _market_data_warning(start: str | None, end: str | None) -> str | None:
    if not start or not end or start == end:
        return None
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    if end_dt - start_dt <= timedelta(days=1):
        return None
    return "Selected comps' market data timestamps span more than 1 day; multiples may not share one exact market close."


def _provenance(cfg: PipelineConfig, comp_records: list[CompanyRecord]) -> dict:
    """Self-attestation block: which config, which code, and how old the
    underlying data actually is. With the layered cache (fetcher), a comp's data
    is no longer necessarily fetched at run time — fundamentals can be up to their
    TTL old and market cap up to its shorter TTL — so the report states the real
    vintage rather than claiming everything is live."""
    market_data_as_of = _oldest_timestamp(comp_records, "market_data_timestamp")
    market_data_through = _newest_timestamp(comp_records, "market_data_timestamp")
    return {
        "config_hash": _config_hash(cfg),
        "code_version": _git_commit(),
        "fundamentals_as_of": _oldest_timestamp(comp_records, "fetch_timestamp"),
        "market_data_as_of": market_data_as_of,
        "market_data_through": market_data_through,
        "market_data_warning": _market_data_warning(market_data_as_of, market_data_through),
    }


def generate(
    scorer_results: dict,
    companies: list[CompanyRecord],
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
    small_sample_warning = None
    if len(eligible_candidates) < cfg.output.min_comps_warning:
        small_sample_warning = (
            f"Small comp pool: only {len(eligible_candidates)} eligible companies survived filtering "
            f"(warning threshold: {cfg.output.min_comps_warning}). Distance-based ranking and the multiple "
            "distributions below are unstable at this sample size — treat them as directional at best, and "
            "consider widening SIC codes or relaxing universe filters."
        )
        logger.warning(small_sample_warning)
    subsector_similarities = _subsector_similarities(
        target_llm_features.get("sub_sector_description"), llm_features, eligible_candidates, embedding_model,
    )

    top15 = _select_top_15(
        company_scores, llm_features, companies_by_ticker,
        target_business_model, target_customer_type, target_revenue, subsector_similarities, penalties,
        llm_rerank={
            **cfg.scorer.llm_rerank.model_dump(),
            "temperature": cfg.llm.temperature,
            "max_tokens": cfg.llm.max_tokens,
        },
        rerank_context={
            "target_profile": {
                "name": target_config.get("name"),
                "description": target_config.get("description"),
                **target_llm_features,
                "revenue_usd_mm": target_revenue,
            },
            "llm_features": llm_features,
            "companies_by_ticker": companies_by_ticker,
        },
        k=top_n,
    )

    top15_rows = _top15_table(companies_by_ticker, llm_features, company_scores, top15)
    must_include_exclusion_notes = _must_include_exclusion_notes(cfg, top15, llm_features, company_scores)
    audit_trail = _audit_trail(
        company_scores, llm_features, companies_by_ticker,
        target_business_model, target_customer_type, target_revenue, top_n, subsector_similarities, penalties,
    )
    model_diagnostics = _model_diagnostics(scorer_results, len(company_scores), top15, eligible_candidates)
    data_notes = _data_notes(llm_features, companies_by_ticker)
    top_review_payload = [_review_candidate_payload(row, "selected") for row in top15_rows]
    near_miss_review_payload = _near_miss_review_payload(audit_trail, companies_by_ticker, llm_features, company_scores)
    comp_fit_review = comp_fit_reviewer.review_comp_fit(target_config, top_review_payload, near_miss_review_payload, cfg)
    comp_fit_review = _filter_review_scope(
        comp_fit_review,
        selected_tickers=set(top15),
        near_miss_tickers={row.get("ticker") for row in near_miss_review_payload},
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
    review_reasons = {
        row.get("ticker"): row.get("reason")
        for row in [*comp_fit_review.get("top_fits", []), *comp_fit_review.get("questionable_fits", [])]
    }
    # Statistical outliers are flagged independently of the LLM's fit
    # judgment (see _ev_ebitda_outlier_tickers) — a comp can be a good
    # qualitative match and still be a multiple outlier, or vice versa. When
    # that moves a row into Review / Exclude, pull in the next ranked candidate
    # so the report still has top_n usable comps instead of letting review rows
    # consume effective comp slots.
    top15_rows, top15, flagged_tickers, substitution_audit = _fill_usable_comp_slots(
        top15, top_n, company_scores, llm_features, companies_by_ticker,
        target_business_model, target_customer_type, target_revenue,
        subsector_similarities, penalties,
        strong_tickers, weak_tickers, review_reasons,
    )
    selection_quality = _selection_quality_summary(top15_rows, target_revenue)
    fit_quality_diagnostics = _fit_quality_diagnostics(top15_rows, target_revenue, model_diagnostics)
    comp_fit_review["fit_quality_diagnostics"] = fit_quality_diagnostics
    comp_fit_review["fit_label"] = (
        _fit_label(
            comp_fit_review.get("overall_score"),
            comp_fit_review.get("weaknesses"),
            fit_quality_diagnostics,
        )
        if comp_fit_review.get("status") == "available" else None
    )

    # Lay the report out tier-first (stable sort preserves adjusted-score
    # order within each tier) and renumber ranks to match — see TIER_ORDER.
    # top15 (the ticker list) keeps selection order; every downstream use of
    # it (valuation distributions, screens) is order-insensitive.
    top15_rows.sort(key=lambda row: TIER_ORDER[row["tier"]])
    for rank, row in enumerate(top15_rows, start=1):
        row["rank"] = rank
    valuation_role_summary = _annotate_valuation_roles(top15_rows)
    analyst_approval_notes = _analyst_approval_notes(cfg, top15_rows)
    substitution_audit_summary = _substitution_audit_summary(substitution_audit)

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
                       else "statistical outlier" if row["outlier_flag"]
                       else "fit flag")
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
        "small_sample_warning": small_sample_warning,
        "must_include_exclusion_notes": must_include_exclusion_notes,
        "analyst_approval_notes": analyst_approval_notes,
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
        "valuation_role_summary": valuation_role_summary,
        "selection_quality": selection_quality,
        "substitution_audit": substitution_audit,
        "substitution_audit_summary": substitution_audit_summary,
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
        "business_model_alignment": _business_model_alignment_summary(top15_rows, target_business_model, target_customer_type),
        "audit_trail": audit_trail,
        "model_diagnostics": model_diagnostics,
        "data_notes": data_notes,
        "selection_summary": _selection_summary(top15, eligible_candidates, model_diagnostics, data_notes),
        "comp_fit_review": comp_fit_review,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "provenance": _provenance(cfg, [companies_by_ticker[t] for t in top15 if t in companies_by_ticker]),
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
    output_paths["small_sample_warning"] = small_sample_warning

    return output_paths
