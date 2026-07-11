"""Valuation math for the report.

This module owns multiple distributions, implied-EV ranges, size
screens/regression, dispersion diagnostics, and benchmark rows. Everything
here is pure computation over the selected comps — no file I/O, no LLM calls
— so sensitivities and benchmark metrics can change without touching
selection semantics or rendering. reporter.py re-exports the shared helpers
so existing callers and tests can keep addressing reporter.<name>.
"""
import math

import numpy as np
import pandas as pd
from scipy import stats

from src import feature_builder


def _ordinal(value: float | int | None) -> str:
    if value is None:
        return "N/A"

    n = int(round(value))
    suffix = "th"
    if not 10 <= n % 100 <= 20:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


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
    range for the target. target_ebitda_margin follows the same
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
