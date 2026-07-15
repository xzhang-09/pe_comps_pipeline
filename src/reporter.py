import csv
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from scipy import stats

from src import get_logger

logger = get_logger(__name__)

# Plain-language labels for the report's "Top features by SHAP importance"
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

OUTPUTS_DIR = Path("outputs")
CSV_PATH = OUTPUTS_DIR / "comps_report.csv"
HTML_PATH = OUTPUTS_DIR / "comps_report.html"
TEMPLATE_DIR = Path("src/templates")
FAILED_TICKERS_PATH = Path("outputs/failed_tickers.csv")

TOP_N = 15
BUSINESS_MODEL_PENALTY = 10
EXEMPT_BUSINESS_MODELS = (None, "other")
CUSTOMER_TYPE_PENALTY = 10
EXEMPT_CUSTOMER_TYPES = (None, "mixed")

# Soft, continuous penalty for revenue-scale mismatch rather than a hard
# size band: our universe doesn't contain companies within a tight size
# band of a typical mid-market target (the smallest real comp on hand is
# ~$500mm), so a hard cutoff would leave too few or zero candidates.
# Companies within 10x of the target's revenue (either direction) get no
# penalty; each further order of magnitude adds SIZE_PENALTY_PER_EXTRA_LOG10.
SIZE_PENALTY_FREE_LOG10_RANGE = 1.0
SIZE_PENALTY_PER_EXTRA_LOG10 = 5.0

DISCLAIMER = "Analysis based on public company data via FMP and SEC EDGAR. For reference purposes only."

# (display label, source field, format) — target estimate for ebitda_margin
# comes from config; the other three have no explicit target estimate, so
# they fall back to the same imputation_medians used to fill the target's
# feature vector in scorer.predict_target().
BENCHMARK_METRICS = (
    ("EBITDA Margin", "ebitda_margin", "percent"),
    ("Revenue Growth", "revenue_cagr_3yr", "percent"),
    ("Gross Margin", "gross_margin", "percent"),
    ("Net Debt/EBITDA", "net_debt_ebitda", "multiple"),
)

CSV_COLUMNS = (
    "rank", "ticker", "company_name", "ev_ebitda_actual", "ev_ebitda_predicted",
    "residual_abs", "ebitda_margin", "gross_margin", "revenue_ttm_usd_mm",
    "revenue_cagr_3yr", "net_debt_ebitda", "business_model", "customer_type",
    "capital_intensity", "sub_sector_description", "judge_score", "low_confidence_flag",
)


def _size_mismatch_penalty(candidate_revenue: float | None, target_revenue: float | None) -> float:
    if not candidate_revenue or not target_revenue or candidate_revenue <= 0 or target_revenue <= 0:
        return 0.0
    log_ratio = abs(math.log10(candidate_revenue / target_revenue))
    excess = max(0.0, log_ratio - SIZE_PENALTY_FREE_LOG10_RANGE)
    return excess * SIZE_PENALTY_PER_EXTRA_LOG10


def _select_top_15(
    company_scores: pd.DataFrame,
    llm_features: dict,
    companies_by_ticker: dict,
    target_business_model: str | None,
    target_customer_type: str | None,
    target_revenue: float | None,
    k: int = TOP_N,
) -> list[str]:
    """
    A hard filter on low_confidence_flag, then residual_abs ranking with
    soft penalties (not exclusions) for:
    - business_model mismatch (+10)
    - customer_type mismatch (+10) — catches cases business_model alone
      doesn't, e.g. a government-contractor "manufacturer" vs. a B2B
      commercial one, both tagged "manufacturing" by the LLM
    - revenue-scale mismatch (continuous, see _size_mismatch_penalty) —
      without this, residual fit alone can surface comps orders of
      magnitude larger/smaller than the target (verified: a $88B-revenue
      aerospace prime ranked into the Top 15 for a $150mm target before
      this penalty existed)
    """
    apply_business_model_penalty = target_business_model not in EXEMPT_BUSINESS_MODELS
    apply_customer_type_penalty = target_customer_type not in EXEMPT_CUSTOMER_TYPES

    candidates = [
        ticker for ticker in company_scores.index
        if llm_features.get(ticker) is not None
        and not llm_features[ticker].get("low_confidence_flag")
    ]

    if len(candidates) < k:
        logger.warning(f"Only {len(candidates)} companies available after low-confidence filter (wanted {k})")

    base_rank = {
        ticker: rank
        for rank, ticker in enumerate(
            sorted(candidates, key=lambda t: company_scores.loc[t, "residual_abs"]), start=1,
        )
    }

    def adjusted_rank(ticker: str) -> float:
        rank = float(base_rank[ticker])
        llm = llm_features[ticker]

        if apply_business_model_penalty and llm.get("business_model") != target_business_model:
            rank += BUSINESS_MODEL_PENALTY
        if apply_customer_type_penalty and llm.get("customer_type") != target_customer_type:
            rank += CUSTOMER_TYPE_PENALTY

        candidate_revenue = companies_by_ticker.get(ticker, {}).get("revenue_ttm_usd_mm")
        rank += _size_mismatch_penalty(candidate_revenue, target_revenue)

        return rank

    ordered = sorted(candidates, key=lambda t: (adjusted_rank(t), base_rank[t]))
    return ordered[:k]


def _distribution_stats(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "mean": float(np.mean(arr)),
    }


def _valuation_multiple_distribution(company_scores: pd.DataFrame, companies_by_ticker: dict, top15: list[str]) -> dict:
    ev_ebitda_values = [company_scores.loc[t, "ev_ebitda_actual"] for t in top15]
    ev_revenue_values = [
        companies_by_ticker[t]["ev_revenue"] for t in top15
        if companies_by_ticker.get(t, {}).get("ev_revenue") is not None
    ]

    return {
        "ev_ebitda": _distribution_stats(ev_ebitda_values),
        "ev_revenue": _distribution_stats(ev_revenue_values) if ev_revenue_values else _distribution_stats([0.0]),
    }


def _target_estimate(field: str, target_config: dict, imputation_medians: dict) -> float | None:
    if field == "ebitda_margin":
        value = target_config.get("ebitda_margin_estimate")
        if value is not None:
            return value
    return imputation_medians.get(field)


def _financial_benchmarks(companies_by_ticker: dict, top15: list[str], target_config: dict, imputation_medians: dict) -> list[dict]:
    rows = []
    for label, field, fmt in BENCHMARK_METRICS:
        values = [
            companies_by_ticker[t][field] for t in top15
            if companies_by_ticker.get(t, {}).get(field) is not None
        ]
        if not values:
            continue

        target_value = _target_estimate(field, target_config, imputation_medians)
        percentile = float(stats.percentileofscore(values, target_value)) if target_value is not None else None

        dist = _distribution_stats(values)
        rows.append({
            "metric": label,
            "format": fmt,
            "target_est": target_value,
            "p25": dist["p25"],
            "median": dist["median"],
            "p75": dist["p75"],
            "target_percentile": percentile,
        })
    return rows


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
            "ev_ebitda_predicted": float(scores["ev_ebitda_predicted"]),
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
        })
    return rows


def _model_diagnostics(scorer_results: dict, n_training_companies: int) -> dict:
    # Precision@15-vs-SEC-proxy-peer-groups used to be surfaced here, but it
    # measures overlap with executive-compensation peer selection (a
    # different objective than valuation comparability) and the raw number
    # is easily misread without the caveats already written in
    # eval/results.md. That nuanced writeup is the right place for it —
    # this client-facing report isn't.
    cv_mae_median = scorer_results.get("cv_mae_median", scorer_results.get("cv_rmse_multiple_space"))
    top_features = scorer_results["feature_importance"].head(5).to_dict(orient="records")
    for row in top_features:
        row["label"] = _humanize_feature_name(row["feature"])

    return {
        "cv_mae_median": cv_mae_median,
        "n_training_companies": n_training_companies,
        "top_features": top_features,
    }


def _data_notes(llm_features: dict) -> dict:
    low_confidence_count = sum(1 for v in llm_features.values() if v.get("low_confidence_flag"))

    failed_fetch_count = 0
    if FAILED_TICKERS_PATH.exists():
        with open(FAILED_TICKERS_PATH, "r", encoding="utf-8") as f:
            failed_fetch_count = max(sum(1 for _ in f) - 1, 0)  # minus header row

    return {
        "low_confidence_count": low_confidence_count,
        "failed_fetch_count": failed_fetch_count,
        "disclaimer": DISCLAIMER,
    }


def _write_csv(rows: list[dict]) -> str:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in CSV_COLUMNS})
    return str(CSV_PATH)


def _render_html(context: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("report.html")
    return template.render(**context)


def _write_html(context: dict) -> str:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    html_text = _render_html(context)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_text)
    return str(HTML_PATH)


def generate(
    scorer_results: dict,
    companies: list[dict],
    llm_features: dict[str, dict],
    target_llm_features: dict,
    imputation_medians: dict,
    config: dict,
) -> dict[str, str]:
    """
    Select Top 15 comps and generate reports.

    Note: target_llm_features and imputation_medians are required here even
    though the original spec signature didn't list them — Top 15 selection
    needs the target's business_model (from analyze_target) for the soft
    penalty, and the financial-benchmarks section needs imputation_medians
    to fill in target estimates config.yaml doesn't provide directly
    (only ebitda_margin_estimate is explicit there).

    Returns:
        {"csv": "outputs/comps_report.csv", "html": "outputs/comps_report.html"}
    """
    companies_by_ticker = {c["ticker"]: c for c in companies}
    company_scores = scorer_results["company_scores"]
    target_config = config["target_company"]

    target_business_model = target_llm_features.get("business_model")
    target_customer_type = target_llm_features.get("customer_type")
    target_revenue = target_config.get("revenue_usd_mm")
    top15 = _select_top_15(
        company_scores, llm_features, companies_by_ticker,
        target_business_model, target_customer_type, target_revenue,
    )

    top15_rows = _top15_table(companies_by_ticker, llm_features, company_scores, top15)

    context = {
        "target_name": target_config.get("name"),
        "target_description": target_config.get("description"),
        "target_prediction": scorer_results["target_prediction"],
        "valuation_multiples": _valuation_multiple_distribution(company_scores, companies_by_ticker, top15),
        "financial_benchmarks": _financial_benchmarks(companies_by_ticker, top15, target_config, imputation_medians),
        "top15": top15_rows,
        "model_diagnostics": _model_diagnostics(scorer_results, len(company_scores)),
        "data_notes": _data_notes(llm_features),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "n_comps": len(top15_rows),
    }

    csv_path = _write_csv(top15_rows)
    html_path = _write_html(context)

    return {"csv": csv_path, "html": html_path}
