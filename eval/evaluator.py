import math
import random
import statistics
from pathlib import Path

import openai

from src import get_logger
from src.llm_analyzer import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, _call_openai, _parse_json_response

logger = get_logger(__name__)

RESULTS_PATH = Path("eval/results.md")
MANUAL_REVIEW_PATH = Path("eval/manual_review_sample.txt")

TOP_K = 15
BUSINESS_MODEL_PENALTY = 10
EXEMPT_BUSINESS_MODELS = (None, "other")
CUSTOMER_TYPE_PENALTY = 10
EXEMPT_CUSTOMER_TYPES = (None, "mixed")

# Mirrors src/reporter.py's _size_mismatch_penalty — kept in sync since this
# function exists specifically to evaluate that selection logic.
SIZE_PENALTY_FREE_LOG10_RANGE = 1.0
SIZE_PENALTY_PER_EXTRA_LOG10 = 5.0


def _size_mismatch_penalty(candidate_revenue: float | None, target_revenue: float | None) -> float:
    if not candidate_revenue or not target_revenue or candidate_revenue <= 0 or target_revenue <= 0:
        return 0.0
    log_ratio = abs(math.log10(candidate_revenue / target_revenue))
    excess = max(0.0, log_ratio - SIZE_PENALTY_FREE_LOG10_RANGE)
    return excess * SIZE_PENALTY_PER_EXTRA_LOG10

CONSISTENCY_FIELDS = (
    "business_model", "revenue_recurrence", "customer_type",
    "capital_intensity", "primary_value_driver",
)
CONSISTENCY_SAMPLE_SIZE = 30
MANUAL_REVIEW_SAMPLE_SIZE = 15
MIN_DESCRIPTION_LENGTH = 100


def _select_top_k(
    target_ticker: str, company_scores, llm_features: dict, companies_by_ticker: dict, k: int = TOP_K,
) -> list[str]:
    """
    Mirrors src/reporter.py's _select_top_15: a hard filter on
    low_confidence_flag, then a residual_abs ranking with soft penalties
    (not exclusions) for business_model mismatch, customer_type mismatch,
    and revenue-scale mismatch (continuous, see _size_mismatch_penalty).
    Kept as a separate copy (not a shared import) since this evaluates
    that selection design rather than depending on it directly.
    """
    target_llm = llm_features.get(target_ticker, {})
    target_business_model = target_llm.get("business_model")
    target_customer_type = target_llm.get("customer_type")
    target_revenue = companies_by_ticker.get(target_ticker, {}).get("revenue_ttm_usd_mm")

    apply_business_model_penalty = target_business_model not in EXEMPT_BUSINESS_MODELS
    apply_customer_type_penalty = target_customer_type not in EXEMPT_CUSTOMER_TYPES

    candidates = [
        ticker for ticker in company_scores.index
        if ticker != target_ticker
        and llm_features.get(ticker) is not None
        and not llm_features[ticker].get("low_confidence_flag")
    ]

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


def _precision_at_k(
    target_ticker: str, ground_truth_peers: list[str], company_scores, llm_features: dict,
    companies_by_ticker: dict, k: int = TOP_K,
):
    peers = {p for p in ground_truth_peers if p != target_ticker}
    if not peers:
        return None

    top_k = set(_select_top_k(target_ticker, company_scores, llm_features, companies_by_ticker, k=k))
    hits = len(top_k & peers)
    precision = hits / len(peers)

    logger.info(
        f"{target_ticker} — Ground truth: {len(peers)} peers, "
        f"Pipeline hits: {hits}/{len(peers)}, Precision: {precision * 100:.1f}%"
    )
    return precision


def _evaluate_precision_at_k(
    ground_truth: dict[str, list[str]], company_scores, llm_features: dict, companies_by_ticker: dict,
) -> dict:
    per_company = {}
    for ticker, peers in ground_truth.items():
        if ticker not in company_scores.index and ticker not in llm_features:
            logger.warning(f"{ticker} — not present in dataset, skipping precision evaluation")
            continue
        precision = _precision_at_k(ticker, peers, company_scores, llm_features, companies_by_ticker)
        if precision is not None:
            per_company[ticker] = precision

    values = list(per_company.values())
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "per_company": {}}

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "per_company": per_company,
    }


def _run_single_extraction(client: openai.OpenAI, company_name: str, business_description: str, config: dict) -> dict | None:
    prompt = USER_PROMPT_TEMPLATE.format(company_name=company_name, business_description=business_description)
    try:
        text = _call_openai(
            client, config["llm"]["judge_model"], SYSTEM_PROMPT, prompt,
            config["llm"]["temperature"], config["llm"]["max_tokens"],
        )
    except Exception as e:
        logger.warning(f"{company_name} — consistency extraction call failed: {e}")
        return None
    return _parse_json_response(text, company_name)


def _evaluate_llm_consistency(companies: list[dict], config: dict, sample_size: int = CONSISTENCY_SAMPLE_SIZE, seed: int | None = None) -> tuple[dict, int]:
    candidates = [
        c for c in companies
        if c.get("business_description") and len(c["business_description"]) >= MIN_DESCRIPTION_LENGTH
    ]
    sample = random.Random(seed).sample(candidates, min(sample_size, len(candidates)))

    client = openai.OpenAI()
    agreement_counts = {field: 0 for field in CONSISTENCY_FIELDS}

    for company in sample:
        name = company.get("company_name", company["ticker"])
        description = company["business_description"]
        run1 = _run_single_extraction(client, name, description, config)
        run2 = _run_single_extraction(client, name, description, config)

        for field in CONSISTENCY_FIELDS:
            if run1 is not None and run2 is not None and run1.get(field) == run2.get(field):
                agreement_counts[field] += 1

    n = len(sample)
    agreement_rates = {
        f"{field}_agreement": (agreement_counts[field] / n if n else 0.0)
        for field in CONSISTENCY_FIELDS
    }
    return agreement_rates, n


def _write_manual_review_sample(companies: list[dict], llm_features: dict, sample_size: int = MANUAL_REVIEW_SAMPLE_SIZE, seed: int | None = None) -> str:
    candidates = [
        c for c in companies
        if llm_features.get(c["ticker"]) and not llm_features[c["ticker"]].get("extraction_failed")
    ]
    sample = random.Random(seed).sample(candidates, min(sample_size, len(candidates)))

    lines = []
    for company in sample:
        ticker = company["ticker"]
        llm = llm_features[ticker]
        lines.append(f"=== TICKER: {ticker} — {company.get('company_name', ticker)} ===")
        lines.append("Business Description (first 300 chars):")
        lines.append((company.get("business_description") or "")[:300])
        lines.append("")
        lines.append("LLM Extraction:")
        for field in CONSISTENCY_FIELDS:
            lines.append(f"  {field}: {llm.get(field)}")
        lines.append(f"  sub_sector_description: {llm.get('sub_sector_description')}")
        lines.append(f"  confidence: {llm.get('confidence')}")
        lines.append(f"  judge_score: {llm.get('judge_score')}")
        lines.append("")
        lines.append("Your assessment (fill in manually):")
        lines.append("  business_model correct? [Y/N]: ___")
        lines.append("  revenue_recurrence correct? [Y/N]: ___")
        lines.append("  customer_type correct? [Y/N]: ___")
        lines.append("")

    text = "\n".join(lines) + "\n"
    MANUAL_REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANUAL_REVIEW_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    return text


def run_evaluation(
    ground_truth: dict[str, list[str]],
    companies: list[dict],
    llm_features: dict[str, dict],
    scorer_results: dict,
    config: dict,
) -> dict:
    """
    Run all three evaluations.
    """
    companies_by_ticker = {c["ticker"]: c for c in companies}
    precision_at_k = _evaluate_precision_at_k(ground_truth, scorer_results["company_scores"], llm_features, companies_by_ticker)
    llm_consistency, n_consistency_samples = _evaluate_llm_consistency(companies, config)
    _write_manual_review_sample(companies, llm_features)

    return {
        "precision_at_k": precision_at_k,
        "llm_consistency": llm_consistency,
        "n_test_companies": len(precision_at_k["per_company"]),
        "n_consistency_samples": n_consistency_samples,
    }


def generate_eval_report(eval_results: dict) -> str:
    """
    Format evaluation results as a readable text report.
    Write to eval/results.md.
    """
    p = eval_results["precision_at_k"]
    c = eval_results["llm_consistency"]

    lines = [
        "# Evaluation Results",
        "",
        "## Precision@15 vs SEC Proxy Peer Groups",
        f"- Mean: {p['mean'] * 100:.1f}%",
        f"- Median: {p['median'] * 100:.1f}%",
        f"- Min: {p['min'] * 100:.1f}%",
        f"- Max: {p['max'] * 100:.1f}%",
        f"- Test companies: {eval_results['n_test_companies']}",
        "",
        "Interpretation: [fill in after reviewing the numbers above]",
        "",
        f"## LLM Extraction Consistency ({eval_results['n_consistency_samples']}-company sample)",
        f"- business_model agreement: {c['business_model_agreement'] * 100:.1f}%",
        f"- revenue_recurrence agreement: {c['revenue_recurrence_agreement'] * 100:.1f}%",
        f"- customer_type agreement: {c['customer_type_agreement'] * 100:.1f}%",
        f"- capital_intensity agreement: {c['capital_intensity_agreement'] * 100:.1f}%",
        f"- primary_value_driver agreement: {c['primary_value_driver_agreement'] * 100:.1f}%",
        "",
        "## Manual Review Results (15 companies)",
        "- business_model accuracy: TBD — fill in after reviewing eval/manual_review_sample.txt",
        "- revenue_recurrence accuracy: TBD",
        "- customer_type accuracy: TBD",
        "",
        "## Key Findings",
        "- [What worked well]",
        "- [What did not work well]",
        "- [Any unexpected patterns]",
    ]
    text = "\n".join(lines) + "\n"

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    return text
