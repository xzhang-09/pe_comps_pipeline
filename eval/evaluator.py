import json
import random
import statistics
from pathlib import Path

import openai

from src import get_logger
from src.llm_analyzer import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, _call_openai, _parse_json_response
from src.report_selection import _eligible_candidates, _ranked_candidates, _select_ranked_tickers, _subsector_similarities

logger = get_logger(__name__)

RESULTS_PATH = Path("eval/results.md")
MANUAL_DEALS_PATH = Path("eval/ground_truth/manual_deals.json")

TOP_K = 15

# Penalty magnitudes and the embedding model come from config.yaml's
# scorer.ranking_penalties / llm.embedding_model. The ranking math itself is
# shared with src.report_selection so eval cannot drift from production.

CONSISTENCY_FIELDS = (
    "business_model", "revenue_recurrence", "customer_type",
    "capital_intensity", "primary_value_driver",
)
CONSISTENCY_SAMPLE_SIZE = 30
MIN_DESCRIPTION_LENGTH = 100


def _select_top_k(
    target_ticker: str, company_scores, llm_features: dict, companies_by_ticker: dict,
    penalties: dict, embedding_model: str, llm_rerank: dict | None = None, k: int = TOP_K,
) -> list[str]:
    """
    Use the same ranking core as production Top-N selection, with the eval
    harness's only eligibility difference: exclude the target ticker itself
    instead of reporter.py's training-bucket exclusion.
    """
    target_llm = llm_features.get(target_ticker, {})
    target_business_model = target_llm.get("business_model")
    target_customer_type = target_llm.get("customer_type")
    target_revenue = companies_by_ticker.get(target_ticker, {}).get("revenue_ttm_usd_mm")

    candidates = _eligible_candidates(
        company_scores, llm_features, companies_by_ticker,
        excluded_tickers={target_ticker}, exclude_training=False,
    )
    subsector_similarities = _subsector_similarities(
        target_llm.get("sub_sector_description"), llm_features, candidates, embedding_model,
    )
    ranked = _ranked_candidates(
        company_scores, llm_features, companies_by_ticker,
        target_business_model, target_customer_type, target_revenue,
        subsector_similarities, penalties,
        excluded_tickers={target_ticker}, exclude_training=False,
    )
    return _select_ranked_tickers(
        ranked,
        k,
        llm_rerank=llm_rerank,
        rerank_context={
            "target_profile": {
                "ticker": target_ticker,
                **target_llm,
                "revenue_usd_mm": target_revenue,
            },
            "llm_features": llm_features,
            "companies_by_ticker": companies_by_ticker,
        },
    )


def _precision_at_k(
    target_ticker: str, ground_truth_peers: list[str], company_scores, llm_features: dict,
    companies_by_ticker: dict, penalties: dict, embedding_model: str, llm_rerank: dict | None = None, k: int = TOP_K,
):
    peers = {p for p in ground_truth_peers if p != target_ticker}
    if not peers:
        return None

    top_k = set(_select_top_k(target_ticker, company_scores, llm_features, companies_by_ticker, penalties, embedding_model, llm_rerank, k=k))
    hits = len(top_k & peers)
    precision = hits / len(peers)

    logger.info(
        f"{target_ticker} — Ground truth: {len(peers)} peers, "
        f"Pipeline hits: {hits}/{len(peers)}, Precision: {precision * 100:.1f}%"
    )
    return precision


def _evaluate_precision_at_k(
    ground_truth: dict[str, list[str]], company_scores, llm_features: dict, companies_by_ticker: dict,
    penalties: dict, embedding_model: str, llm_rerank: dict | None = None,
) -> dict:
    per_company = {}
    for ticker, peers in ground_truth.items():
        if ticker not in company_scores.index and ticker not in llm_features:
            logger.warning(f"{ticker} — not present in dataset, skipping precision evaluation")
            continue
        precision = _precision_at_k(ticker, peers, company_scores, llm_features, companies_by_ticker, penalties, embedding_model, llm_rerank)
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


def _ticker(value: str | None) -> str:
    return str(value or "").strip().upper()


def load_manual_deals(path: str | Path = MANUAL_DEALS_PATH) -> list[dict]:
    """Load hand-built fairness-opinion comp sets."""
    with open(path, encoding="utf-8") as f:
        deals = json.load(f)
    if not isinstance(deals, list):
        raise ValueError("Manual ground truth must be a list of deals")
    required = {"deal_id", "target_ticker", "target_name", "filing_url", "selected_companies"}
    for deal in deals:
        missing = required - set(deal)
        if missing:
            raise ValueError(f"{deal.get('deal_id', '<unknown>')} missing required fields: {sorted(missing)}")
        if not isinstance(deal["selected_companies"], list) or not deal["selected_companies"]:
            raise ValueError(f"{deal['deal_id']} must include selected_companies")
        deal["target_ticker"] = _ticker(deal["target_ticker"])
        for comp in deal["selected_companies"]:
            comp["ticker"] = _ticker(comp.get("ticker"))
    return deals


def validate_manual_deals_benchmark(deals: list[dict], min_deals: int = 8, max_deals: int = 10) -> dict:
    """Validate that manual deals are ready to serve as an audited benchmark."""
    if not min_deals <= len(deals) <= max_deals:
        raise ValueError(f"Manual benchmark must contain {min_deals}-{max_deals} deals; found {len(deals)}")

    required = {
        "deal_id", "target_ticker", "target_name", "target_cik", "target_sic",
        "business_description", "target_financials", "filing_url", "filing_date",
        "advisor", "selected_companies", "review_status",
    }
    reviewed = 0
    eligible_public_comps = 0
    excluded_delisted_comps = 0
    sic_codes = set()

    for deal in deals:
        deal_id = deal.get("deal_id", "<unknown>")
        missing = required - set(deal)
        if missing:
            raise ValueError(f"{deal_id} missing benchmark fields: {sorted(missing)}")
        if deal["review_status"] != "reviewed":
            raise ValueError(f"{deal_id} review_status must be reviewed")
        reviewed += 1

        financials = deal.get("target_financials") or {}
        if financials.get("revenue_usd_mm") is None or not financials.get("source"):
            raise ValueError(f"{deal_id} must include target revenue and financial source")
        if not deal.get("business_description"):
            raise ValueError(f"{deal_id} must include business_description")

        selected_companies = deal.get("selected_companies")
        if not isinstance(selected_companies, list) or not selected_companies:
            raise ValueError(f"{deal_id} must include selected_companies")
        for comp in selected_companies:
            if not comp.get("ticker") or not comp.get("company_name"):
                raise ValueError(f"{deal_id} selected companies must include ticker and company_name")
            if "still_public" not in comp:
                raise ValueError(f"{deal_id} selected companies must include still_public")
            if comp["still_public"]:
                eligible_public_comps += 1
            else:
                excluded_delisted_comps += 1

        sic_codes.add(str(deal["target_sic"]))

    return {
        "n_deals": len(deals),
        "reviewed_deals": reviewed,
        "target_sic_codes": sorted(sic_codes),
        "eligible_public_comps": eligible_public_comps,
        "excluded_delisted_comps": excluded_delisted_comps,
    }


def _manual_ground_truth_tickers(deal: dict) -> tuple[list[str], list[str], list[str]]:
    """Split a deal's banker comps into (eligible, delisted, non_us_filer).
    Only still-public US 10-K filers enter the precision denominator: a
    delisted comp or a foreign-listed one (us_filer=False, e.g. an LSE-only
    name) cannot be discovered by the EDGAR/SIC pipeline even in principle,
    so counting it as a miss would measure the documented data contract, not
    selection quality. Both exclusions stay in the audit trail. A missing
    us_filer key counts as eligible for backward compatibility with entries
    audited before the flag existed."""
    eligible = []
    delisted = []
    non_us = []
    for comp in deal["selected_companies"]:
        ticker = _ticker(comp.get("ticker"))
        if not ticker:
            continue
        if not comp.get("still_public", True):
            delisted.append(ticker)
        elif comp.get("us_filer", True) is False:
            non_us.append(ticker)
        else:
            eligible.append(ticker)
    return list(dict.fromkeys(eligible)), list(dict.fromkeys(delisted)), list(dict.fromkeys(non_us))


def _evaluate_manual_deal(deal: dict, company_scores, llm_features: dict, companies_by_ticker: dict, config: dict, k: int) -> dict:
    target = _ticker(deal["target_ticker"])
    eligible, delisted, non_us = _manual_ground_truth_tickers(deal)
    penalties = config["scorer"]["ranking_penalties"]
    embedding_model = config["llm"]["embedding_model"]
    llm_rerank = config.get("scorer", {}).get("llm_rerank")
    selected = _select_top_k(target, company_scores, llm_features, companies_by_ticker, penalties, embedding_model, llm_rerank, k=k)
    selected_set = set(selected)
    score_index = set(company_scores.index)
    hits = [ticker for ticker in eligible if ticker in selected_set]
    missed = [ticker for ticker in eligible if ticker not in selected_set]
    not_in_universe = [ticker for ticker in missed if ticker not in score_index]
    not_selected = [ticker for ticker in missed if ticker in score_index]
    precision = len(hits) / len(eligible) if eligible else None
    return {
        "deal_id": deal["deal_id"],
        "target_ticker": target,
        "target_name": deal["target_name"],
        "filing_url": deal.get("filing_url"),
        "advisor": deal.get("advisor"),
        "filing_date": deal.get("filing_date"),
        "selected_tickers": selected,
        "eligible_ground_truth_tickers": eligible,
        "excluded_delisted_tickers": delisted,
        "excluded_non_us_filer_tickers": non_us,
        "hits": hits,
        "missed_not_in_universe": not_in_universe,
        "missed_not_selected": not_selected,
        "precision": precision,
    }


def run_manual_ground_truth_evaluation(
    deals: list[dict], company_scores, llm_features: dict, companies_by_ticker: dict, config: dict, k: int = TOP_K,
) -> dict:
    per_deal = [
        _evaluate_manual_deal(deal, company_scores, llm_features, companies_by_ticker, config, k)
        for deal in deals
    ]
    values = [row["precision"] for row in per_deal if row["precision"] is not None]
    return {
        "mean_precision": statistics.mean(values) if values else 0.0,
        "median_precision": statistics.median(values) if values else 0.0,
        "n_deals": len(per_deal),
        "k": k,
        "per_deal": per_deal,
    }


def generate_manual_ground_truth_report(results: dict) -> str:
    lines = [
        "# Manual Ground Truth Evaluation",
        "",
        "## Methodology",
        "",
        "- Ground truth: fairness-opinion Selected Companies Analysis lists, audited against the filings",
        "  (verbatim comp names, advisor, projected financials) — see eval/ground_truth/manual_deals.json",
        "  per-deal notes for provenance.",
        "- Denominator: only banker comps that are still public **and** US 10-K filers (us_filer=true)",
        "  count toward Precision. Delisted comps and foreign-listed non-filers are excluded from the",
        "  denominator but retained in the audit trail — the pipeline's documented data contract is US",
        "  EDGAR 10-K filers, so those names are out of reach by design, not selection misses.",
        "- Target financials: the projection year is the filing-date fiscal year when at least ~6 months",
        "  of it remained, otherwise the next fiscal year; both years' figures are recorded in each",
        "  deal's `source` field.",
        "- Discovery vs. ranking: every eligible comp is attributed to a waterfall stage (below).",
        "  `Reachable precision` scores the ranking layer alone — hits over comps that actually reached",
        "  the scored pool — since a comp lost at discovery says nothing about ranking quality.",
        "",
        f"## Precision@{results['k']}",
        f"- Mean: {results['mean_precision'] * 100:.1f}%",
        f"- Median: {results['median_precision'] * 100:.1f}%",
        f"- Deals: {results['n_deals']}",
    ]
    if results.get("discovery_mode"):
        lines.append(
            f"- Discovery mode: {results['discovery_mode']} "
            "(ladder: single-sic baseline -> suggest-sic expansion; compare runs by mode, "
            "the delta is the measured value of each discovery upgrade)"
        )
    if results.get("n_failed_deals"):
        lines.append(f"- Failed deal runs (excluded from aggregates): {results['n_failed_deals']}")
    if results.get("mean_reachable_precision") is not None:
        lines.append(
            f"- Mean reachable (ranking-layer) precision: {results['mean_reachable_precision'] * 100:.1f}%"
        )

    waterfall = results.get("coverage_waterfall") or {}
    if waterfall:
        total = sum(waterfall.values())
        lines += [
            "",
            "## Coverage Waterfall (all eligible banker comps)",
            "",
            "| Stage | Count |",
            "| --- | ---: |",
        ]
        lines += [f"| {stage} | {count} |" for stage, count in waterfall.items()]
        reached = sum(waterfall.get(s, 0) for s in ("hit", "ranked_but_not_top_k"))
        lines += [
            "",
            f"{reached} of {total} eligible comps reached the scored pool; stages above the pair "
            "hit/ranked_but_not_top_k are ranking outcomes, everything below is a coverage loss "
            "(discovery, filters, or data gaps). `missing_market_cap` usually means an FMP "
            "quota/coverage miss — re-run after the quota resets before reading it as a data gap.",
        ]

    lines += ["", "## Deal Detail"]
    for row in results["per_deal"]:
        lines += [
            f"### {row['deal_id']} — {row['target_ticker']} ({row.get('target_name')})",
            f"- Source: {row.get('advisor') or 'unknown advisor'}, {row.get('filing_date') or 'unknown date'} — {row.get('filing_url') or 'no URL'}",
        ]
        if row.get("error"):
            lines += [f"- RUN FAILED: {row['error']}", ""]
            continue
        precision = "N/A" if row["precision"] is None else f"{row['precision'] * 100:.1f}%"
        reachable = row.get("reachable_precision")
        lines += [
            f"- Precision@{results['k']}: {precision}"
            + (f" (reachable: {reachable * 100:.1f}%)" if reachable is not None else " (no comp reached ranking)"),
            f"- Hits: {', '.join(row['hits']) or 'none'}",
            f"- Missed before universe/scoring: {', '.join(row['missed_not_in_universe']) or 'none'}",
            f"- Missed after ranking: {', '.join(row['missed_not_selected']) or 'none'}",
            f"- Excluded delisted banker comps: {', '.join(row['excluded_delisted_tickers']) or 'none'}",
        ]
        if row.get("excluded_non_us_filer_tickers"):
            lines.append(
                f"- Excluded non-US-filer banker comps: {', '.join(row['excluded_non_us_filer_tickers'])}"
            )
        if row.get("n_scored") is not None:
            trivial = " — selection trivial (pool <= K, precision measures coverage only)" if row.get("selection_trivial") else ""
            lines.append(f"- Scored pool: {row['n_scored']} companies ({row.get('n_selectable')} selectable){trivial}")
        codes = row.get("discovery_sic_codes") or {}
        if codes:
            adjacent = ", ".join(codes.get("adjacent") or []) or "none"
            lines.append(f"- Discovery SIC codes: primary {', '.join(codes.get('primary') or [])}; adjacent {adjacent}")
        misses = {t: s for t, s in (row.get("coverage") or {}).items() if s != "hit"}
        if misses:
            lines.append("- Loss stages: " + ", ".join(f"{t}={s}" for t, s in sorted(misses.items())))
        lines.append("")
    text = "\n".join(lines)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(text + "\n", encoding="utf-8")
    return text + "\n"


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
    llm_rerank = config.get("scorer", {}).get("llm_rerank")
    precision_at_k = _evaluate_precision_at_k(
        ground_truth, scorer_results["company_scores"], llm_features, companies_by_ticker, penalties, embedding_model, llm_rerank,
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
