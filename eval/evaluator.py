import random
import statistics
from pathlib import Path

import numpy as np
import openai

from src import get_logger
from src.llm_analyzer import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, _call_openai, _parse_json_response, embed_texts
from src.reporter import EXEMPT_BUSINESS_MODELS, EXEMPT_CUSTOMER_TYPES, _size_mismatch_penalty

logger = get_logger(__name__)

RESULTS_PATH = Path("eval/results.md")

TOP_K = 15

# Penalty magnitudes (business_model/customer_type/size/sub-sector) and the
# embedding model come from config.yaml's scorer.ranking_penalties /
# llm.embedding_model — passed through run_evaluation()'s `config` argument
# rather than hardcoded here, so this evaluation can't drift out of sync
# with reporter.py's actual production selection logic the way two
# independently-maintained copies of the same constants could.
# EXEMPT_BUSINESS_MODELS/EXEMPT_CUSTOMER_TYPES and _size_mismatch_penalty
# are imported directly from reporter.py for the same reason.


def _subsector_similarities(
    target_description: str | None, llm_features: dict, candidates: list[str], embedding_model: str,
) -> dict[str, float]:
    """Mirrors src/reporter.py's _subsector_similarities — see there for the
    full rationale. Returns {} (no penalty applied to anyone) if the target
    has no sub_sector_description, no candidate has one either, or the
    embedding call fails."""
    if not target_description:
        return {}

    tickers_with_text = [t for t in candidates if llm_features.get(t, {}).get("sub_sector_description")]
    if not tickers_with_text:
        return {}

    texts = [target_description] + [llm_features[t]["sub_sector_description"] for t in tickers_with_text]
    vectors = embed_texts(texts, model=embedding_model)
    if vectors is None:
        logger.warning("Sub-sector embedding lookup failed; skipping sub-sector mismatch penalty for this evaluation")
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

CONSISTENCY_FIELDS = (
    "business_model", "revenue_recurrence", "customer_type",
    "capital_intensity", "primary_value_driver",
)
CONSISTENCY_SAMPLE_SIZE = 30
MIN_DESCRIPTION_LENGTH = 100


def _select_top_k(
    target_ticker: str, company_scores, llm_features: dict, companies_by_ticker: dict,
    penalties: dict, embedding_model: str, k: int = TOP_K,
) -> list[str]:
    """
    Mirrors src/reporter.py's _select_top_15: a hard filter on
    low_confidence_flag, then a residual_abs ranking with soft penalties
    (not exclusions, magnitudes from `penalties` — see config.yaml's
    scorer.ranking_penalties) for business_model mismatch, customer_type
    mismatch, revenue-scale mismatch (continuous, see
    reporter._size_mismatch_penalty), and sub-sector mismatch (continuous,
    via embeddings — see _subsector_similarities). Kept as a separate
    ranking-loop copy (not a shared import of _select_top_15 itself) since
    this evaluates that selection design rather than depending on it
    directly — but the penalty magnitudes and exemption lists are imported/
    threaded from the same source as reporter.py, not re-hardcoded.
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

    subsector_similarities = _subsector_similarities(
        target_llm.get("sub_sector_description"), llm_features, candidates, embedding_model,
    )
    subsector_threshold = penalties["subsector_similarity_threshold"]

    def adjusted_score(ticker: str) -> float:
        # Penalties are added to the continuous financial distance (residual_abs),
        # not the ordinal base_rank — must stay in lockstep with
        # reporter._penalty_breakdown so the eval harness ranks the same way the
        # report does. See that function's comment for why distance-unit penalties.
        score = float(company_scores.loc[ticker, "residual_abs"])
        llm = llm_features[ticker]

        if apply_business_model_penalty and llm.get("business_model") != target_business_model:
            score += penalties["business_model_penalty"]
        if apply_customer_type_penalty and llm.get("customer_type") != target_customer_type:
            score += penalties["customer_type_penalty"]

        candidate_revenue = companies_by_ticker.get(ticker, {}).get("revenue_ttm_usd_mm")
        score += _size_mismatch_penalty(candidate_revenue, target_revenue, penalties)

        subsector_similarity = subsector_similarities.get(ticker)
        if subsector_similarity is not None and subsector_similarity < subsector_threshold:
            score += penalties["subsector_mismatch_penalty"]

        return score

    ordered = sorted(candidates, key=lambda t: (adjusted_score(t), base_rank[t]))
    return ordered[:k]


def _precision_at_k(
    target_ticker: str, ground_truth_peers: list[str], company_scores, llm_features: dict,
    companies_by_ticker: dict, penalties: dict, embedding_model: str, k: int = TOP_K,
):
    peers = {p for p in ground_truth_peers if p != target_ticker}
    if not peers:
        return None

    top_k = set(_select_top_k(target_ticker, company_scores, llm_features, companies_by_ticker, penalties, embedding_model, k=k))
    hits = len(top_k & peers)
    precision = hits / len(peers)

    logger.info(
        f"{target_ticker} — Ground truth: {len(peers)} peers, "
        f"Pipeline hits: {hits}/{len(peers)}, Precision: {precision * 100:.1f}%"
    )
    return precision


def _evaluate_precision_at_k(
    ground_truth: dict[str, list[str]], company_scores, llm_features: dict, companies_by_ticker: dict,
    penalties: dict, embedding_model: str,
) -> dict:
    per_company = {}
    for ticker, peers in ground_truth.items():
        if ticker not in company_scores.index and ticker not in llm_features:
            logger.warning(f"{ticker} — not present in dataset, skipping precision evaluation")
            continue
        precision = _precision_at_k(ticker, peers, company_scores, llm_features, companies_by_ticker, penalties, embedding_model)
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
    penalties = config["scorer"]["ranking_penalties"]
    embedding_model = config["llm"]["embedding_model"]
    precision_at_k = _evaluate_precision_at_k(
        ground_truth, scorer_results["company_scores"], llm_features, companies_by_ticker, penalties, embedding_model,
    )
    llm_consistency, n_consistency_samples = _evaluate_llm_consistency(companies, config)

    return {
        "precision_at_k": precision_at_k,
        "llm_consistency": llm_consistency,
        "n_test_companies": len(precision_at_k["per_company"]),
        "n_consistency_samples": n_consistency_samples,
    }


def generate_eval_report(eval_results: dict) -> str:
    """
    Format evaluation results as a readable text report.
    Write current evaluation output to eval/results.md.
    """
    p = eval_results["precision_at_k"]
    c = eval_results["llm_consistency"]

    lines = [
        "# Evaluation Results",
        "",
        "> These results are generated by the current evaluation harness. Publish them",
        "> only after the ground-truth source has been validated for the current",
        "> universe and target set.",
        "",
        "## Precision@15",
        f"- Mean: {p['mean'] * 100:.1f}%",
        f"- Median: {p['median'] * 100:.1f}%",
        f"- Min: {p['min'] * 100:.1f}%",
        f"- Max: {p['max'] * 100:.1f}%",
        f"- Test companies: {eval_results['n_test_companies']}",
        "",
        "Interpretation: Add only after validating the ground-truth source and reviewing the numbers above.",
        "",
        f"## LLM Extraction Consistency ({eval_results['n_consistency_samples']}-company sample)",
        f"- business_model agreement: {c['business_model_agreement'] * 100:.1f}%",
        f"- revenue_recurrence agreement: {c['revenue_recurrence_agreement'] * 100:.1f}%",
        f"- customer_type agreement: {c['customer_type_agreement'] * 100:.1f}%",
        f"- capital_intensity agreement: {c['capital_intensity_agreement'] * 100:.1f}%",
        f"- primary_value_driver agreement: {c['primary_value_driver_agreement'] * 100:.1f}%",
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
