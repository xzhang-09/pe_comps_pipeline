import csv
from pathlib import Path

import pandas as pd
import pytest

import src.reporter as reporter
from src.config_schema import as_config


@pytest.fixture(autouse=True)
def no_real_embedding_calls(mocker):
    """
    reporter._subsector_similarities() calls llm_analyzer.embed_texts(),
    which hits the real OpenAI API. Block that by default for every test in
    this module — tests that exercise the sub-sector penalty itself
    override this with their own `mocker.patch(...)` of the same target.
    Scoped to this module only (not tests/conftest.py) so it doesn't shadow
    test_llm_analyzer.py's tests of the real embed_texts implementation.
    """
    mocker.patch("src.reporter.llm_analyzer.embed_texts", return_value=None)
    mocker.patch("src.reporter.comp_fit_reviewer.review_comp_fit", return_value={
        "status": "unavailable",
        "reason": "mocked in reporter tests",
    })


# Penalties are in residual-distance units (added to residual_abs, not an
# ordinal rank) — see reporter._penalty_breakdown. Fixtures below pair these
# with residual gaps small enough that a penalty can override a modest
# financial edge, the behavior these tests assert.
DEFAULT_PENALTIES = {
    "business_model_penalty": 0.6,
    "customer_type_penalty": 0.5,
    "subsector_similarity_threshold": 0.5,
    "subsector_mismatch_penalty": 0.4,
    "size_penalty_free_log10_range": 1.0,
    "size_penalty_per_extra_log10": 1.0,
}


def _company(ticker, **overrides):
    base = {
        "ticker": ticker,
        "company_name": f"{ticker} Inc.",
        "market_cap_usd_mm": 500.0,
        "revenue_ttm_usd_mm": 200.0,
        "ebitda_margin": 0.20,
        "gross_margin": 0.35,
        "revenue_cagr_3yr": 0.05,
        "net_debt_ebitda": 1.5,
        "capex_revenue": 0.04,
        "ev_ebitda": 12.0,
        "ev_revenue": 2.4,
        "gics_sector": "20",
        "business_description": "test",
        "description_source": "edgar",
        "fetch_timestamp": "2026-06-16T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def _llm(business_model="manufacturing", low_confidence_flag=False, **overrides):
    base = {
        "business_model": business_model,
        "revenue_recurrence": "medium",
        "customer_type": "B2B",
        "capital_intensity": "moderate",
        "primary_value_driver": "scale",
        "sub_sector_description": "test sub-sector",
        "confidence": 4,
        "judge_score": 4,
        "judge_reason": "ok",
        "low_confidence_flag": low_confidence_flag,
        "extraction_failed": False,
    }
    base.update(overrides)
    return base


def _build_sample(n=30, n_matching=15):
    companies = []
    llm_features = {}
    scores_rows = {}
    for i in range(n):
        ticker = f"T{i:03d}"
        matching = i < n_matching
        companies.append(_company(ticker, ev_ebitda=10.0 + i * 0.1))
        llm_features[ticker] = _llm(business_model="manufacturing" if matching else "services")
        scores_rows[ticker] = {
            "ev_ebitda_actual": 10.0 + i * 0.1,
            "residual_abs": 0.5 + i * 0.01,
        }
    company_scores = pd.DataFrame(scores_rows).T
    return companies, llm_features, company_scores


def _scorer_results(company_scores):
    sq_diff = pd.DataFrame({
        "revenue_ttm_log": [0.10] * len(company_scores),
        "ebitda_margin": [0.09] * len(company_scores),
        "gross_margin": [0.08] * len(company_scores),
        "revenue_cagr_3yr": [0.07] * len(company_scores),
        "net_debt_ebitda": [0.06] * len(company_scores),
        "capex_revenue": [0.05] * len(company_scores),
    }, index=company_scores.index)
    return {
        "company_scores": company_scores,
        "feature_distance_sq_diff": sq_diff,
    }


def _sample_config():
    return {
        "target_company": {
            "name": "Example Manufacturing Co.",
            "description": "A test target company.",
            "primary_sic_codes": ["3714"],
            "revenue_usd_mm": 150,
            "ebitda_margin_estimate": 0.18,
        },
        "universe": {"max_candidates": 10},
        "llm": {
            "extraction_model": "gpt-4.1",
            "judge_model": "gpt-4.1-mini",
            "temperature": 0,
            "max_tokens": 500,
            "batch_size": 20,
            "judge_threshold": 3,
            "embedding_model": "text-embedding-3-small",
        },
        "output": {"top_n_comps": 15, "report_formats": ["csv", "html"]},
        "scorer": {"ranking_penalties": dict(DEFAULT_PENALTIES)},
    }


def _imputation_medians():
    global_medians = {
        "revenue_ttm_log": 5.0,
        "ebitda_margin": 0.2,
        "gross_margin": 0.35,
        "revenue_cagr_3yr": 0.05,
        "net_debt_ebitda": 1.5,
        "capex_revenue": 0.04,
    }
    return {"by_group": {"manufacturing": global_medians}, "global": global_medians}


def test_top15_selected_correctly():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm(business_model="manufacturing")

    result = reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), _sample_config(),
    )

    with open(result["csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 15


def test_config_hash_deterministic_and_sensitive(make_config):
    # Same config -> same fingerprint; any change -> different fingerprint.
    assert reporter._config_hash(make_config()) == reporter._config_hash(make_config())
    assert reporter._config_hash(make_config(universe={"max_candidates": 99})) != reporter._config_hash(make_config())


def test_oldest_timestamp_picks_earliest_ignoring_missing():
    records = [
        {"fetch_timestamp": "2026-06-20T00:00:00+00:00"},
        {"fetch_timestamp": "2026-06-10T00:00:00+00:00"},
        {},  # no timestamp — ignored
    ]
    assert reporter._oldest_timestamp(records, "fetch_timestamp") == "2026-06-10T00:00:00+00:00"
    assert reporter._oldest_timestamp([], "fetch_timestamp") is None


def test_provenance_reports_market_data_range_and_warning(make_config):
    records = [
        {"market_data_timestamp": "2026-06-10T00:00:00+00:00", "fetch_timestamp": "2026-06-09T00:00:00+00:00"},
        {"market_data_timestamp": "2026-06-12T00:00:00+00:00", "fetch_timestamp": "2026-06-08T00:00:00+00:00"},
    ]

    result = reporter._provenance(make_config(), records)

    assert result["market_data_as_of"] == "2026-06-10T00:00:00+00:00"
    assert result["market_data_through"] == "2026-06-12T00:00:00+00:00"
    assert "span more than 1 day" in result["market_data_warning"]


def test_report_html_embeds_provenance():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)
    config = _sample_config()

    result = reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing"),
        _imputation_medians(), config,
    )

    html = Path(result["html"]).read_text(encoding="utf-8")
    assert reporter._config_hash(as_config(config)) in html  # report self-attests its config
    assert "Provenance" in html


def test_top_n_comps_config_controls_csv_row_count():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm(business_model="manufacturing")
    config = _sample_config()
    config["output"]["top_n_comps"] = 7

    result = reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), config,
    )

    with open(result["csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 7


def test_report_formats_config_can_generate_html_only():
    companies, llm_features, company_scores = _build_sample()
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm()
    config = _sample_config()
    config["output"]["report_formats"] = ["html"]

    result = reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), config,
    )

    assert "html" in result
    assert "csv" not in result
    assert reporter.HTML_PATH.exists()
    assert not reporter.CSV_PATH.exists()


def test_csv_file_created():
    companies, llm_features, company_scores = _build_sample()
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm()

    reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), _sample_config(),
    )

    assert reporter.CSV_PATH.exists()


def test_report_marks_analyst_specified_top_comp_in_csv_and_html():
    companies, llm_features, company_scores = _build_sample(n=2, n_matching=2)
    companies[0]["candidate_source"] = "analyst_specified"
    companies[0]["universe_metadata"] = {"candidate_source": "analyst_specified"}
    scorer_results = _scorer_results(company_scores)
    config = _sample_config()
    config["output"]["top_n_comps"] = 2
    config["universe"]["must_include_tickers"] = [companies[0]["ticker"]]

    result = reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing"), _imputation_medians(), config,
    )

    with open(result["csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    html = Path(result["html"]).read_text(encoding="utf-8")
    assert rows[0]["candidate_source"] == "analyst_specified"
    assert "Analyst-specified" in html


def test_report_notes_low_confidence_must_include_exclusion():
    companies = [
        _company("KEEP"),
        _company("MANUAL", candidate_source="analyst_specified", universe_metadata={"candidate_source": "analyst_specified"}),
    ]
    llm_features = {
        "KEEP": _llm(business_model="manufacturing", low_confidence_flag=False),
        "MANUAL": _llm(business_model="manufacturing", low_confidence_flag=True),
    }
    company_scores = pd.DataFrame({
        "ev_ebitda_actual": {"KEEP": 10.0, "MANUAL": 11.0},
        "residual_abs": {"KEEP": 0.5, "MANUAL": 0.01},
    })
    scorer_results = _scorer_results(company_scores)
    config = _sample_config()
    config["output"]["top_n_comps"] = 1
    config["universe"]["must_include_tickers"] = ["MANUAL"]

    result = reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing"), _imputation_medians(), config,
    )

    html = Path(result["html"]).read_text(encoding="utf-8")
    assert "MANUAL was analyst-specified but excluded" in html
    assert "low-confidence" in html


def test_size_and_customer_type_mismatch_excluded():
    # Target: $150mm revenue, B2B. GIANT has the best residual fit by far
    # but is a $90B-revenue, B2G company — both a scale and customer-type
    # mismatch. CLOSE is a worse fit but matches the target's profile.
    companies = [
        _company("CLOSE", revenue_ttm_usd_mm=180.0),
        _company("GIANT", revenue_ttm_usd_mm=90_000.0),
    ]
    llm_features = {
        "CLOSE": _llm(business_model="manufacturing", customer_type="B2B"),
        "GIANT": _llm(business_model="manufacturing", customer_type="B2G"),
    }
    company_scores = pd.DataFrame({
        "residual_abs": {"CLOSE": 0.5, "GIANT": 0.05},
    })

    top1 = reporter._select_top_15(
        company_scores, llm_features, {c["ticker"]: c for c in companies},
        target_business_model="manufacturing", target_customer_type="B2B", target_revenue=150.0,
        subsector_similarities=None, penalties=DEFAULT_PENALTIES, k=1,
    )

    assert top1 == ["CLOSE"]


def test_training_only_candidates_are_excluded_from_top_comps():
    companies = [
        _company("COMP", source_bucket="primary"),
        _company("TRAIN", source_bucket="training"),
    ]
    llm_features = {
        "COMP": _llm(business_model="manufacturing", customer_type="B2B"),
        "TRAIN": _llm(business_model="manufacturing", customer_type="B2B"),
    }
    company_scores = pd.DataFrame({
        "residual_abs": {"COMP": 1.0, "TRAIN": 0.01},
    })

    top1 = reporter._select_top_15(
        company_scores, llm_features, {c["ticker"]: c for c in companies},
        target_business_model="manufacturing", target_customer_type="B2B", target_revenue=150.0,
        subsector_similarities=None, penalties=DEFAULT_PENALTIES, k=1,
    )

    assert top1 == ["COMP"]


def test_subsector_mismatch_excludes_low_similarity_candidate(mocker):
    # FAR has the best residual fit but its sub_sector_description is
    # orthogonal (cosine similarity 0) to the target's; CLOSE is a slightly
    # worse financial fit but matches sub-sector closely (similarity ~0.98).
    # Gap (0.20 vs 0.05) is within one subsector penalty (0.4) so the mismatch
    # flips the order — see DEFAULT_PENALTIES note on distance-unit penalties.
    companies = [_company("CLOSE"), _company("FAR")]
    llm_features = {
        "CLOSE": _llm(sub_sector_description="automotive OEM parts manufacturer"),
        "FAR": _llm(sub_sector_description="semiconductor process equipment maker"),
    }
    company_scores = pd.DataFrame({"residual_abs": {"CLOSE": 0.20, "FAR": 0.05}})
    # target vector [1, 0]; CLOSE [0.99, 0.14] (~similarity 0.99); FAR [0, 1] (similarity 0)
    mocker.patch(
        "src.reporter.llm_analyzer.embed_texts",
        return_value=[[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]],
    )

    subsector_similarities = reporter._subsector_similarities(
        "automotive OEM parts target description", llm_features, ["CLOSE", "FAR"], "text-embedding-3-small",
    )
    top1 = reporter._select_top_15(
        company_scores, llm_features, {c["ticker"]: c for c in companies},
        target_business_model=None, target_customer_type=None, target_revenue=None,
        subsector_similarities=subsector_similarities, penalties=DEFAULT_PENALTIES, k=1,
    )

    assert subsector_similarities["FAR"] < DEFAULT_PENALTIES["subsector_similarity_threshold"]
    assert top1 == ["CLOSE"]


def test_subsector_similarities_empty_when_embedding_call_fails(mocker):
    mocker.patch("src.reporter.llm_analyzer.embed_texts", return_value=None)
    llm_features = {"A": _llm(sub_sector_description="some sub-sector")}

    result = reporter._subsector_similarities("target description", llm_features, ["A"], "text-embedding-3-small")

    assert result == {}


def test_subsector_similarities_empty_when_target_has_no_description(mocker):
    mock_embed = mocker.patch("src.reporter.llm_analyzer.embed_texts")
    llm_features = {"A": _llm(sub_sector_description="some sub-sector")}

    result = reporter._subsector_similarities(None, llm_features, ["A"], "text-embedding-3-small")

    assert result == {}
    mock_embed.assert_not_called()


def test_html_file_created():
    companies, llm_features, company_scores = _build_sample()
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm()

    reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), _sample_config(),
    )

    assert reporter.HTML_PATH.exists()
    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "Example Manufacturing Co." in html_text


def test_audit_trail_shows_near_miss_candidates_with_reasons():
    # CLOSE makes the Top-1; GIANT is the next-best by financial distance
    # alone but is excluded by the customer-type penalty — the audit trail
    # for k=1 should surface GIANT with that reason.
    companies = [
        _company("CLOSE", revenue_ttm_usd_mm=180.0),
        _company("GIANT", revenue_ttm_usd_mm=190.0),
    ]
    llm_features = {
        "CLOSE": _llm(business_model="manufacturing", customer_type="B2B"),
        "GIANT": _llm(business_model="manufacturing", customer_type="B2G"),
    }
    company_scores = pd.DataFrame({
        "residual_abs": {"CLOSE": 0.5, "GIANT": 0.05},
    })

    audit = reporter._audit_trail(
        company_scores, llm_features, {c["ticker"]: c for c in companies},
        target_business_model="manufacturing", target_customer_type="B2B", target_revenue=150.0,
        top_n=1, subsector_similarities=None, penalties=DEFAULT_PENALTIES,
    )

    assert len(audit) == 1
    assert audit[0]["ticker"] == "GIANT"
    assert any("customer type mismatch" in reason for reason in audit[0]["reasons"])


def test_audit_trail_limited_to_audit_size(monkeypatch):
    companies = [_company(f"T{i}") for i in range(10)]
    llm_features = {c["ticker"]: _llm() for c in companies}
    company_scores = pd.DataFrame({
        "residual_abs": {c["ticker"]: float(i) for i, c in enumerate(companies)},
    })

    audit = reporter._audit_trail(
        company_scores, llm_features, {c["ticker"]: c for c in companies},
        target_business_model="manufacturing", target_customer_type="B2B", target_revenue=150.0,
        top_n=1, subsector_similarities=None, penalties=DEFAULT_PENALTIES,
    )

    assert len(audit) == reporter.AUDIT_SIZE


def test_audit_trail_display_sorted_by_financial_rank():
    companies = [
        _company("TOP", revenue_ttm_usd_mm=150.0),
        _company("EARLY", revenue_ttm_usd_mm=150.0),
        _company("MID", revenue_ttm_usd_mm=150.0),
        _company("LATE", revenue_ttm_usd_mm=150.0),
    ]
    llm_features = {
        "TOP": _llm(customer_type="B2B"),
        "EARLY": _llm(customer_type="B2G"),
        "MID": _llm(customer_type="B2B"),
        "LATE": _llm(customer_type="B2B"),
    }
    company_scores = pd.DataFrame({
        "residual_abs": {"TOP": 0.01, "EARLY": 0.02, "MID": 0.03, "LATE": 0.04},
    })

    audit = reporter._audit_trail(
        company_scores, llm_features, {c["ticker"]: c for c in companies},
        target_business_model="manufacturing", target_customer_type="B2B", target_revenue=150.0,
        top_n=1, subsector_similarities=None, penalties=DEFAULT_PENALTIES, audit_size=3,
    )

    assert [row["base_rank"] for row in audit] == [2, 3, 4]


def test_html_report_includes_near_miss_section():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm(business_model="manufacturing")

    reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "Near-Miss Candidates" in html_text


def test_html_report_separates_quality_snapshot_from_extraction_reliability():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm(business_model="manufacturing")

    reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "Selection Snapshot" in html_text
    assert "companies excluded for weak source support" in html_text
    assert "Extraction Judge Score" not in html_text
    assert "Comparable Fit Review" in html_text
    assert "AI assistance note" in html_text


def test_ordinal_suffixes_handle_common_edge_cases():
    assert reporter._ordinal(1) == "1st"
    assert reporter._ordinal(2) == "2nd"
    assert reporter._ordinal(3) == "3rd"
    assert reporter._ordinal(11) == "11th"
    assert reporter._ordinal(12) == "12th"
    assert reporter._ordinal(13) == "13th"
    assert reporter._ordinal(21) == "21st"
    assert reporter._ordinal(73) == "73rd"


def test_html_report_explains_target_percentile_basis():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm(business_model="manufacturing")

    reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "calculated directly from company-level data" in html_text


def test_html_report_includes_comp_fit_review_when_available(mocker):
    mocker.patch("src.reporter.comp_fit_reviewer.review_comp_fit", return_value={
        "status": "available",
        "overall_score": 84,
        "review_confidence": "medium",
        "summary": "The Top 15 are a solid directional comp set.",
        "strengths": ["Most selected companies are manufacturing/B2B."],
        "weaknesses": ["A few candidates have adjacent end markets."],
        "top_fits": [{"ticker": "T000", "score": 90, "reason": "Best business-model fit."}],
        "questionable_fits": [{"ticker": "T001", "score": 58, "reason": "Different end market."}],
        "near_miss_upgrades": [{"ticker": "T015", "score": 76, "reason": "Could replace a weaker fit."}],
    })
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm(business_model="manufacturing")

    reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "Comparable Fit Review" in html_text
    assert "84 / 100" in html_text
    assert "Best business-model fit." in html_text
    assert "Different end market." in html_text


def test_html_report_downgrades_good_label_when_fit_evidence_has_material_caveats(mocker):
    mocker.patch("src.reporter.comp_fit_reviewer.review_comp_fit", return_value={
        "status": "available",
        "overall_score": 65,
        "review_confidence": "high",
        "summary": "Directional set with material caveats.",
        "strengths": ["Most companies share a broad manufacturing label."],
        "weaknesses": [
            "Revenue scale mismatch is significant, with several comps above 5x target revenue.",
            "End-market fit is weak for several selected comps.",
        ],
        "top_fits": [{"ticker": "T000", "score": 80, "reason": "Best selected fit."}],
        "questionable_fits": [{"ticker": "T001", "score": 50, "reason": "Different end market."}],
        "near_miss_upgrades": [],
    })
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm(business_model="manufacturing", customer_type="B2B")

    reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "65 / 100 (Mixed / directionally useful with material caveats)" in html_text
    assert "65 / 100 (Good / directionally supportive)" not in html_text


def test_fit_label_downgrades_when_many_selected_comps_are_review_exclude():
    rows = [
        {"ticker": "A", "tier": "core", "revenue_ttm_usd_mm": 100.0},
        {"ticker": "B", "tier": "secondary", "revenue_ttm_usd_mm": 120.0},
        {"ticker": "C", "tier": "review_exclude", "revenue_ttm_usd_mm": 700.0},
        {"ticker": "D", "tier": "review_exclude", "revenue_ttm_usd_mm": 800.0},
    ]

    diagnostics = reporter._fit_quality_diagnostics(rows, target_revenue=100.0, model_diagnostics={})

    assert diagnostics["review_exclude_share"] == pytest.approx(0.5)
    assert diagnostics["max_revenue_ratio"] == pytest.approx(8.0)
    assert reporter._fit_label(68, [], diagnostics) == "Mixed / directionally useful with material caveats"


def test_selection_quality_summary_counts_tiers_and_revenue_ratios():
    rows = [
        {"ticker": "A", "tier": "core", "revenue_ttm_usd_mm": 100.0},
        {"ticker": "B", "tier": "secondary", "revenue_ttm_usd_mm": 300.0},
        {"ticker": "C", "tier": "review_exclude", "revenue_ttm_usd_mm": 600.0},
    ]

    summary = reporter._selection_quality_summary(rows, target_revenue=150.0)

    assert summary["n_total"] == 3
    assert summary["n_core"] == 1
    assert summary["n_secondary"] == 1
    assert summary["n_review_exclude"] == 1
    assert summary["usable_count"] == 2
    assert summary["max_revenue_ratio"] == pytest.approx(4.0)
    assert summary["median_revenue_ratio"] == pytest.approx(2.0)


def test_html_report_counts_business_model_alignment_when_top_set_has_exception(mocker):
    mocker.patch("src.reporter.comp_fit_reviewer.review_comp_fit", return_value={
        "status": "available",
        "overall_score": 70,
        "review_confidence": "medium",
        "summary": "Directional set.",
        "strengths": [],
        "weaknesses": [],
        "top_fits": [],
        "questionable_fits": [],
        "near_miss_upgrades": [],
    })
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    llm_features["T014"] = _llm(business_model="services", customer_type="B2B")
    llm_features["T013"] = _llm(business_model="manufacturing", customer_type="B2C")
    scorer_results = _scorer_results(company_scores)

    reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing", customer_type="B2B"),
        _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "14/15 comps match the target's business model (manufacturing)" in html_text
    assert "Exceptions: T014 (services)" in html_text
    assert "14/15 comps match the target's customer base (B2B)" in html_text
    assert "Exceptions: T013 (B2C)" in html_text
    assert "All Top 15 comps share" not in html_text


def test_html_report_notes_invalid_ticker_references_in_review_narrative(mocker):
    mocker.patch("src.reporter.comp_fit_reviewer.review_comp_fit", return_value={
        "status": "available",
        "overall_score": 65,
        "review_confidence": "high",
        "summary": "Directional set.",
        "strengths": [],
        "weaknesses": ["One selected comp (HLLO) serves B2C customers."],
        "top_fits": [],
        "questionable_fits": [],
        "near_miss_upgrades": [],
    })
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    llm_features["T001"] = _llm(customer_type="B2C")
    scorer_results = _scorer_results(company_scores)

    reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing", customer_type="B2B"),
        _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "Review narrative referenced unavailable ticker HLLO" in html_text


def test_html_report_filters_comp_fit_review_tickers_to_their_scope(mocker):
    mocker.patch("src.reporter.comp_fit_reviewer.review_comp_fit", return_value={
        "status": "available",
        "overall_score": 70,
        "review_confidence": "medium",
        "summary": "Directional set.",
        "strengths": [],
        "weaknesses": [],
        "top_fits": [{"ticker": "T000", "score": 90, "reason": "Selected comp."}],
        "questionable_fits": [{"ticker": "T020", "score": 55, "reason": "Near miss incorrectly listed as selected."}],
        "near_miss_upgrades": [{"ticker": "T001", "score": 80, "reason": "Selected comp incorrectly listed as near miss."}],
    })
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)

    reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing"),
        _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "Near miss incorrectly listed as selected." not in html_text
    assert "Selected comp incorrectly listed as near miss." not in html_text
    assert "Review scope check removed 1 selected-comp callout" in html_text


def test_top_table_moves_business_customer_and_fit_reason_to_note(mocker):
    mocker.patch("src.reporter.comp_fit_reviewer.review_comp_fit", return_value={
        "status": "available",
        "overall_score": 70,
        "review_confidence": "medium",
        "summary": "Directional set.",
        "strengths": [],
        "weaknesses": [],
        "top_fits": [],
        "questionable_fits": [{"ticker": "T001", "score": 55, "reason": "Different end market."}],
        "near_miss_upgrades": [],
    })
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    llm_features["T001"] = _llm(business_model="services", customer_type="B2C")
    company_scores.loc["T001", "residual_abs"] = 0.01
    scorer_results = _scorer_results(company_scores)

    reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing", customer_type="B2B"),
        _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "<th>Business</th>" not in html_text
    assert "<th>Customer</th>" not in html_text
    assert "<th>Fit Notes</th>" not in html_text
    assert "business model mismatch" in html_text
    assert "customer type mismatch" in html_text
    assert "Different end market." in html_text


def test_relative_dispersion_below_one_when_selection_narrows_spread():
    # Pool spans a wide EV/EBITDA range; the selected Top-N is a tight
    # cluster near the middle — selection should show as narrowing spread.
    pool_tickers = [f"P{i}" for i in range(20)]
    company_scores = pd.DataFrame({
        "ev_ebitda_actual": {t: 5.0 + i * 2.0 for i, t in enumerate(pool_tickers)},
    })
    top_n_tickers = pool_tickers[9:12]  # a narrow band in the middle of the pool

    result = reporter._relative_dispersion(company_scores, pool_tickers, top_n_tickers)

    assert result["ratio"] is not None
    assert result["ratio"] < 1.0
    assert result["n_pool"] == 20
    assert result["n_selected"] == 3


def test_relative_dispersion_none_when_pool_too_small():
    company_scores = pd.DataFrame({"ev_ebitda_actual": {"A": 10.0}})

    result = reporter._relative_dispersion(company_scores, ["A"], ["A"])

    assert result["ratio"] is None
    assert result["pool_iqr"] is None


def test_iqr_returns_none_below_min_values():
    assert reporter._iqr([10.0]) is None
    assert reporter._iqr([]) is None


def test_iqr_computed_for_sufficient_values():
    assert reporter._iqr([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.5)


def test_model_diagnostics_includes_relative_dispersion():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)
    eligible = reporter._eligible_candidates(company_scores, llm_features, {c["ticker"]: c for c in companies})
    top15 = list(company_scores.index[:15])

    diagnostics = reporter._model_diagnostics(scorer_results, len(company_scores), top15, eligible)

    assert "relative_dispersion" in diagnostics
    assert diagnostics["relative_dispersion"]["n_pool"] == len(eligible)


def test_outlier_absolute_guard_catches_extreme_within_tukey_fence():
    import numpy as np

    # A dispersed set whose Tukey upper fence is wide enough to wave through
    # a 41x multiple sitting next to a ~14x median — the absolute guard
    # (ABSOLUTE_OUTLIER_MEDIAN_MULTIPLE × median) should still flag it.
    multiples = [4.9, 6.4, 8.7, 10.2, 11.7, 11.8, 13.0, 13.9, 15.1, 15.6, 23.5, 27.5, 28.4, 41.3, 49.7]
    tickers = [f"T{i:02d}" for i in range(len(multiples))]
    company_scores = pd.DataFrame({"ev_ebitda_actual": dict(zip(tickers, multiples))})

    # Precondition: 41.3 is inside the Tukey fence, so Tukey alone misses it.
    p25, p75 = np.percentile(multiples, 25), np.percentile(multiples, 75)
    tukey_upper = p75 + reporter.TUKEY_FENCE_MULTIPLIER * (p75 - p25)
    assert 41.3 < tukey_upper

    flagged = reporter._ev_ebitda_outlier_tickers(company_scores, tickers)

    assert tickers[13] in flagged  # 41.3 — only the absolute guard catches this
    assert tickers[14] in flagged  # 49.7 — Tukey catches this too


def test_discounted_valuation_applies_haircut_to_both_bases():
    implied = {
        "by_ebitda": {"p25": 100.0, "median": 200.0, "p75": 300.0},
        "by_revenue": {"p25": 80.0, "median": 160.0, "p75": 240.0},
    }

    result = reporter._discounted_valuation(implied, 0.25)

    assert result["discount"] == 0.25
    assert result["factor"] == pytest.approx(0.75)
    assert result["by_ebitda"]["median"] == pytest.approx(150.0)
    assert result["by_revenue"]["p75"] == pytest.approx(180.0)


def test_discounted_valuation_none_when_no_discount():
    implied = {"by_ebitda": {"p25": 100.0, "median": 200.0, "p75": 300.0}, "by_revenue": None}

    assert reporter._discounted_valuation(implied, 0.0) is None
    assert reporter._discounted_valuation(implied, None) is None


def test_discounted_valuation_none_when_no_ranges():
    assert reporter._discounted_valuation({"by_ebitda": None, "by_revenue": None}, 0.25) is None


def test_football_field_rows_order_and_contents():
    implied = {
        "by_ebitda": {"p25": 200.0, "median": 260.0, "p75": 320.0},
        "by_revenue": {"p25": 180.0, "median": 240.0, "p75": 300.0},
    }
    discounted = {"discount": 0.25, "by_ebitda": {"p25": 150.0, "median": 195.0, "p75": 240.0}, "by_revenue": None}
    size_anchor = {"n": 4, "median_ev_ebitda": 9.0, "implied_ev": 180.0}
    size_adjusted = {"implied_ev": 210.0, "is_significant": True}

    rows = reporter._football_field_rows(implied, None, discounted, size_adjusted, size_anchor)
    labels = [r["label"] for r in rows]

    # Size anchor and the post-discount range lead; raw Top-N ranges and the
    # regression point follow.
    assert labels[0].startswith("Size anchor")
    assert labels[1].startswith("After 25% discount")
    assert "EV/EBITDA (Top-N)" in labels
    assert "EV/Revenue (Top-N)" in labels
    assert labels[-1].startswith("Size-adjusted")
    # The discounted row uses the EV/EBITDA basis (by_revenue was None).
    disc_row = next(r for r in rows if r["label"].startswith("After"))
    assert disc_row["mid"] == pytest.approx(195.0)


def test_football_field_excludes_non_significant_size_adjusted_regression():
    implied = {
        "by_ebitda": {"p25": 200.0, "median": 260.0, "p75": 320.0},
        "by_revenue": None,
    }
    size_adjusted = {"implied_ev": 210.0, "is_significant": False}

    rows = reporter._football_field_rows(implied, None, None, size_adjusted, None)

    assert "Size-adjusted (regr.)" not in [r["label"] for r in rows]


def test_football_field_svg_renders_when_a_range_exists():
    implied = {"by_ebitda": {"p25": 200.0, "median": 260.0, "p75": 320.0}, "by_revenue": None}

    svg = reporter._football_field_svg(implied, None, None, None, None)

    assert svg is not None
    assert svg.startswith("<svg")
    assert "EV/EBITDA (Top-N)" in svg
    assert "Implied Enterprise Value" in svg


def test_football_field_svg_none_without_any_range():
    # Only point estimates, no comp range -> nothing to anchor the axis on.
    implied = {"by_ebitda": None, "by_revenue": None}
    size_anchor = {"n": 3, "median_ev_ebitda": 9.0, "implied_ev": 180.0}

    assert reporter._football_field_svg(implied, None, None, None, size_anchor) is None


def test_valuation_multiple_distribution_includes_added_multiples():
    top = ["A", "B", "C"]
    companies_by_ticker = {
        t: _company(t, ev_revenue=2.0 + i, ev_ebit=14.0 + i, ev_gross_profit=6.0 + i,
                    pe_ratio=20.0 + i, fcf_yield=0.05 + 0.01 * i)
        for i, t in enumerate(top)
    }
    company_scores = pd.DataFrame({"ev_ebitda_actual": [10.0, 11.0, 12.0]}, index=top)

    dist = reporter._valuation_multiple_distribution(company_scores, companies_by_ticker, top)

    for key in ("ev_ebitda", "ev_revenue", "ev_ebit", "ev_gross_profit", "pe_ratio", "fcf_yield"):
        assert key in dist
        assert "median" in dist[key]
    assert dist["ev_ebit"]["median"] == pytest.approx(15.0)
    assert dist["pe_ratio"]["median"] == pytest.approx(21.0)


def test_valuation_multiple_distribution_placeholder_when_field_absent():
    # No comp carries ev_ebit -> a zero placeholder so the template's fixed
    # rows still have stats to format (same convention as ev_revenue).
    top = ["A", "B"]
    companies_by_ticker = {t: _company(t, ev_ebit=None) for t in top}
    company_scores = pd.DataFrame({"ev_ebitda_actual": [10.0, 11.0]}, index=top)

    dist = reporter._valuation_multiple_distribution(company_scores, companies_by_ticker, top)

    assert dist["ev_ebit"]["median"] == pytest.approx(0.0)


def test_financial_benchmarks_includes_leverage_and_fcf_rows():
    top = ["A", "B", "C"]
    companies_by_ticker = {
        t: _company(t, fcf_conversion=0.60 + 0.05 * i, interest_coverage=5.0 + i, debt_to_equity=0.5 + 0.1 * i)
        for i, t in enumerate(top)
    }
    imputation_medians = {
        "by_group": {},
        "global": {"ebitda_margin": 0.18, "revenue_cagr_3yr": 0.05, "gross_margin": 0.35, "capex_revenue": 0.04, "net_debt_ebitda": 1.5},
    }
    target_config = {"ebitda_margin_estimate": 0.18}

    rows = reporter._financial_benchmarks(companies_by_ticker, top, target_config, imputation_medians, "manufacturing")
    by_metric = {r["metric"]: r for r in rows}

    assert {"FCF Conversion", "Interest Coverage", "Debt/Equity"} <= set(by_metric)
    # Interest Coverage isn't in imputation_medians, so the target has no
    # estimate — the row still shows the comp distribution.
    assert by_metric["Interest Coverage"]["target_est"] is None
    assert by_metric["Interest Coverage"]["median"] == pytest.approx(6.0)
    # capex_revenue does have an imputation median, so its target_est is filled.
    assert by_metric["Capex/Revenue"]["target_est"] == pytest.approx(0.04)


def test_size_anchor_uses_strict_screen_median_times_ebitda():
    strict = {"n": 4, "tickers": ["A", "B", "C", "D"], "median_ev_ebitda": 11.0}

    anchor = reporter._size_anchor(strict, target_ebitda=27.0)

    assert anchor["n"] == 4
    assert anchor["median_ev_ebitda"] == 11.0
    assert anchor["implied_ev"] == pytest.approx(297.0)


def test_size_anchor_none_without_screen_or_ebitda():
    assert reporter._size_anchor(None, 27.0) is None
    assert reporter._size_anchor({"median_ev_ebitda": 11.0}, None) is None


def test_html_report_includes_private_company_adjusted_range():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm(business_model="manufacturing")
    config = _sample_config()
    config["valuation"] = {"size_marketability_discount": 0.25, "discount_note": "test note"}

    reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), config,
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "Private-company-adjusted" in html_text
    assert "25% net size/marketability discount" in html_text


def test_html_report_downgrades_non_significant_size_adjusted_regression(mocker):
    mocker.patch("src.reporter._size_adjusted_valuation", return_value={
        "n": 30,
        "slope": 0.12,
        "r_squared": 0.03,
        "p_value": 0.42,
        "is_significant": False,
        "predicted_multiple": 9.5,
        "implied_ev": 256.0,
    })
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm(business_model="manufacturing")

    reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "Size-adjusted regression tested but not used" in html_text
    assert "Implied EV (directional)" not in html_text
    assert "Size-adjusted (regr.)" not in html_text


def test_html_report_flags_low_margin_comp():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    companies[0]["ebitda_margin"] = 0.05  # T000 is a matching comp -> lands in Top-N
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm(business_model="manufacturing")

    reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "soft data point" in html_text  # the low-margin caution note
    assert "5.0% EBITDA margin" in html_text


def test_html_report_includes_table_of_contents():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm(business_model="manufacturing")

    reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert 'class="toc"' in html_text
    assert 'href="#sec4"' in html_text


def test_html_report_omits_adjusted_range_when_discount_zero():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm(business_model="manufacturing")

    # _sample_config() has no "valuation" block at all — the no-op default.
    reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "Private-company-adjusted implied EV" not in html_text


def test_assign_tier_requires_no_mismatch_for_core():
    assert reporter._assign_tier("strong", False, has_mismatch=False) == "core"
    assert reporter._assign_tier("strong", False, has_mismatch=True) == "secondary"
    assert reporter._assign_tier("strong", True, has_mismatch=False) == "review_exclude"
    assert reporter._assign_tier("weak", False, has_mismatch=False) == "review_exclude"
    assert reporter._assign_tier(None, False, has_mismatch=False) == "core"


def test_mismatched_strong_fit_capped_at_secondary_in_csv(mocker):
    # The LLM review calls both T000 and T005 strongest fits, but T005's
    # business model mismatches the target — the deterministic mismatch
    # check must cap it at secondary so the tier never contradicts the
    # fit_notes printed next to it.
    mocker.patch("src.reporter.comp_fit_reviewer.review_comp_fit", return_value={
        "status": "available",
        "overall_score": 80,
        "review_confidence": "medium",
        "summary": "Directional set.",
        "strengths": [],
        "weaknesses": [],
        "top_fits": [
            {"ticker": "T000", "score": 90, "reason": "Best fit."},
            {"ticker": "T005", "score": 85, "reason": "Strong fit."},
        ],
        "questionable_fits": [],
        "near_miss_upgrades": [],
    })
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    llm_features["T005"] = _llm(business_model="services")  # mismatch vs manufacturing target
    scorer_results = _scorer_results(company_scores)

    result = reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing"),
        _imputation_medians(), _sample_config(),
    )

    with open(result["csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tiers = {row["ticker"]: row["tier"] for row in rows}
    assert tiers["T000"] == "core"
    assert tiers["T005"] == "secondary"

    html_text = Path(result["html"]).read_text(encoding="utf-8")
    assert "fit-strong" not in html_text
    assert "tier-row-core" in html_text
    assert "tier-row-secondary" in html_text
    assert "Strongest fit" not in html_text
    assert "Secondary" in html_text


def test_any_subsector_shortfall_blocks_core_even_with_strong_fit(mocker):
    # Similarity just below the threshold (0.47 vs 0.5). The graded penalty is
    # tiny (0.4 * 0.03 / 0.5 ≈ 0.02) so the ranking impact is negligible, but
    # Core requires clearing the similarity bar outright — the old
    # near-threshold + LLM-strong exception existed only to soften the cliff
    # penalty, which the graded ramp removed.
    mocker.patch("src.reporter.comp_fit_reviewer.review_comp_fit", return_value={
        "status": "available",
        "overall_score": 80,
        "review_confidence": "medium",
        "summary": "Directional set.",
        "strengths": [],
        "weaknesses": [],
        "top_fits": [{"ticker": "T000", "score": 90, "reason": "Best fit."}],
        "questionable_fits": [],
        "near_miss_upgrades": [],
    })
    mocker.patch(
        "src.reporter.llm_analyzer.embed_texts",
        return_value=[[1.0, 0.0], *([[0.47, 0.8832]] * 30)],
    )
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=30)
    target_llm = _llm(business_model="manufacturing", sub_sector_description="target automotive OEM parts")
    scorer_results = _scorer_results(company_scores)

    result = reporter.generate(
        scorer_results, companies, llm_features, target_llm,
        _imputation_medians(), _sample_config(),
    )

    with open(result["csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = next(row for row in rows if row["ticker"] == "T000")
    assert row["tier"] == "secondary"
    assert "end-market similarity below threshold" in row["fit_notes"]
    # Graded penalty: shortfall of ~0.03 against max 0.4 → ~0.02, not the cliff 0.4.
    assert "+0.02 rank penalty" in row["fit_notes"]


def test_severe_subsector_penalty_blocks_core_even_with_strong_fit(mocker):
    mocker.patch("src.reporter.comp_fit_reviewer.review_comp_fit", return_value={
        "status": "available",
        "overall_score": 80,
        "review_confidence": "medium",
        "summary": "Directional set.",
        "strengths": [],
        "weaknesses": [],
        "top_fits": [{"ticker": "T000", "score": 90, "reason": "Best fit."}],
        "questionable_fits": [],
        "near_miss_upgrades": [],
    })
    mocker.patch(
        "src.reporter.llm_analyzer.embed_texts",
        return_value=[[1.0, 0.0], *([[0.0, 1.0]] * 30)],
    )
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=30)
    target_llm = _llm(business_model="manufacturing", sub_sector_description="target automotive OEM parts")
    scorer_results = _scorer_results(company_scores)

    result = reporter.generate(
        scorer_results, companies, llm_features, target_llm,
        _imputation_medians(), _sample_config(),
    )

    with open(result["csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = next(row for row in rows if row["ticker"] == "T000")
    assert row["tier"] == "secondary"
    assert "end-market similarity below threshold" in row["fit_notes"]


def test_subsector_penalty_blocks_core_without_strong_fit(mocker):
    mocker.patch(
        "src.reporter.llm_analyzer.embed_texts",
        return_value=[[1.0, 0.0], *([[0.0, 1.0]] * 30)],
    )
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=30)
    target_llm = _llm(business_model="manufacturing", sub_sector_description="target automotive OEM parts")
    scorer_results = _scorer_results(company_scores)

    result = reporter.generate(
        scorer_results, companies, llm_features, target_llm,
        _imputation_medians(), _sample_config(),
    )

    with open(result["csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = next(row for row in rows if row["ticker"] == "T000")
    assert row["tier"] == "secondary"
    assert "end-market similarity below threshold" in row["fit_notes"]


def test_revenue_gap_beyond_10x_blocks_core():
    # T001 is a clean categorical match but 13x the target's revenue
    # (2000 vs 150, log10 gap ≈ 1.13 > CORE_MAX_LOG10_REVENUE_GAP) — it stays
    # usable but is capped at secondary. T000 at 200 (1.3x) keeps core.
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    for company in companies:
        if company["ticker"] == "T001":
            company["revenue_ttm_usd_mm"] = 2000.0
    scorer_results = _scorer_results(company_scores)

    result = reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing"),
        _imputation_medians(), _sample_config(),
    )

    with open(result["csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tiers = {row["ticker"]: row["tier"] for row in rows}
    assert tiers["T000"] == "core"
    assert tiers["T001"] == "secondary"
    row = next(row for row in rows if row["ticker"] == "T001")
    assert "revenue scale mismatch" in row["fit_notes"]


def test_subsector_penalty_is_graded_not_cliff():
    penalties = dict(DEFAULT_PENALTIES)  # threshold 0.5, max penalty 0.4
    from src.report_selection import _subsector_mismatch_penalty

    assert _subsector_mismatch_penalty(None, penalties) == 0.0
    assert _subsector_mismatch_penalty(0.55, penalties) == 0.0
    assert _subsector_mismatch_penalty(0.5, penalties) == 0.0
    # Linear ramp: halfway to zero similarity costs half the max.
    assert _subsector_mismatch_penalty(0.25, penalties) == pytest.approx(0.2)
    assert _subsector_mismatch_penalty(0.0, penalties) == pytest.approx(0.4)
    # A 0.02 similarity difference moves the penalty by 0.4 * 0.02 / 0.5 = 0.016,
    # not by the full 0.4 the old cliff applied.
    near = _subsector_mismatch_penalty(0.48, penalties)
    assert near == pytest.approx(0.016)


def test_report_rows_ordered_tier_first_and_reranked():
    # T000 has the best financial fit but an extreme EV/EBITDA multiple —
    # flagged review_exclude, so it must sink below the unflagged comps
    # instead of leading the table, and ranks must be renumbered to match.
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    company_scores.loc["T000", "ev_ebitda_actual"] = 100.0
    scorer_results = _scorer_results(company_scores)

    result = reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing"),
        _imputation_medians(), _sample_config(),
    )

    with open(result["csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["ticker"] == "T000"
    assert rows[-1]["tier"] == "review_exclude"
    tier_order = [reporter.TIER_ORDER[row["tier"]] for row in rows]
    assert tier_order == sorted(tier_order)
    assert [int(row["rank"]) for row in rows] == list(range(1, len(rows) + 1))


def test_review_exclude_rows_do_not_consume_usable_comp_slots():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=30)
    company_scores.loc["T000", "ev_ebitda_actual"] = 100.0
    scorer_results = _scorer_results(company_scores)

    result = reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing"),
        _imputation_medians(), _sample_config(),
    )

    with open(result["csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    usable = [row for row in rows if row["tier"] != "review_exclude"]

    assert len(usable) == 15
    assert rows[-1]["ticker"] == "T000"
    assert rows[-1]["tier"] == "review_exclude"
    assert "T015" in {row["ticker"] for row in rows}


def test_fill_usable_comp_slots_rejects_review_exclude_replacement():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=30)
    company_scores.loc["T000", "ev_ebitda_actual"] = 100.0
    company_scores.loc["T015", "ev_ebitda_actual"] = 95.0

    rows, selected, flagged_tickers, substitution_audit = reporter._fill_usable_comp_slots(
        [f"T{i:03d}" for i in range(15)],
        15,
        company_scores,
        llm_features,
        {company["ticker"]: company for company in companies},
        "manufacturing",
        "B2B",
        150.0,
        {},
        DEFAULT_PENALTIES,
        set(),
        set(),
        {},
    )

    assert "T015" not in selected
    assert "T016" in selected
    assert "T000" in flagged_tickers
    assert next(row for row in rows if row["ticker"] == "T000")["tier"] == "review_exclude"
    assert any(item["ticker"] == "T015" and item["decision"] == "rejected" for item in substitution_audit)
    assert any(item["ticker"] == "T016" and item["decision"] == "accepted" for item in substitution_audit)


def test_fill_usable_comp_slots_accepts_secondary_replacement_when_better_than_review_slot():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=30)
    company_scores.loc["T000", "ev_ebitda_actual"] = 100.0
    companies[15]["revenue_ttm_usd_mm"] = 3000.0
    company_scores.loc["T015", "residual_abs"] = 0.01

    rows, selected, flagged_tickers, substitution_audit = reporter._fill_usable_comp_slots(
        [f"T{i:03d}" for i in range(15)],
        15,
        company_scores,
        llm_features,
        {company["ticker"]: company for company in companies},
        "manufacturing",
        "B2B",
        150.0,
        {},
        DEFAULT_PENALTIES,
        set(),
        set(),
        {},
    )

    assert "T015" in selected
    assert "T000" in flagged_tickers
    assert any(
        item["ticker"] == "T015"
        and item["decision"] == "accepted"
        and item["tier"] == "secondary"
        and "better Secondary replacement" in item["reason"]
        for item in substitution_audit
    )


def test_html_report_shows_selection_quality_and_substitution_audit():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=30)
    company_scores.loc["T000", "ev_ebitda_actual"] = 100.0
    scorer_results = _scorer_results(company_scores)

    reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing"),
        _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "Selection quality check" in html_text
    assert "usable comps" in html_text
    assert "Replacement decisions" in html_text
    assert "<th>Decision</th><th>Ticker</th><th>Tier</th><th>Reason</th>" in html_text


def test_substitution_audit_summary_limits_rejected_rows():
    audit = [
        {"ticker": "A", "decision": "accepted", "tier": "core", "reason": "ok"},
        *[
            {"ticker": f"R{i}", "decision": "rejected", "tier": "secondary", "reason": "no"}
            for i in range(7)
        ],
    ]

    summary = reporter._substitution_audit_summary(audit, rejected_limit=5)

    assert summary["n_accepted"] == 1
    assert summary["n_rejected"] == 7
    assert [item["ticker"] for item in summary["displayed"]] == ["A", "R0", "R1", "R2", "R3", "R4"]
    assert summary["hidden_rejected_count"] == 2


def test_html_report_formats_fit_flags_as_table():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    llm_features["T000"] = _llm(business_model="services")
    scorer_results = _scorer_results(company_scores)

    reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing"),
        _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "<summary><strong>Fit flags</strong>" in html_text
    assert "<th>Ticker</th><th>Tier</th><th>Primary flags</th>" in html_text


def test_html_report_shows_valuation_role_summary():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=30)
    companies[1]["revenue_ttm_usd_mm"] = 3000.0
    company_scores.loc["T001", "residual_abs"] = 0.01
    scorer_results = _scorer_results(company_scores)

    reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing"),
        _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "Valuation anchor comps" in html_text
    assert "Broad reference comps" in html_text
    assert "Review / sensitivity only" in html_text


def test_html_report_notes_analyst_approved_comps_without_overriding_tier():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=30)
    companies[1]["revenue_ttm_usd_mm"] = 3000.0
    company_scores.loc["T001", "residual_abs"] = 0.01
    scorer_results = _scorer_results(company_scores)
    config = _sample_config()
    config["universe"]["analyst_approved_tickers"] = ["T001"]

    reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing"),
        _imputation_medians(), config,
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "Analyst-approved comps" in html_text
    assert "T001" in html_text
    assert "model tier/caveats still shown" in html_text
    assert "T001</td>" in html_text
    assert "Secondary Comps" in html_text


def test_small_pool_stamps_warning_on_report_and_return_value():
    companies, llm_features, company_scores = _build_sample(n=9, n_matching=9)
    scorer_results = _scorer_results(company_scores)

    result = reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing"),
        _imputation_medians(), _sample_config(),
    )

    assert "only 9 eligible companies" in result["small_sample_warning"]
    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "Sample-size warning" in html_text
    assert "only 9 eligible companies" in html_text


def test_no_small_sample_warning_at_or_above_threshold():
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)

    result = reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing"),
        _imputation_medians(), _sample_config(),
    )

    assert result["small_sample_warning"] is None
    assert "Sample-size warning" not in reporter.HTML_PATH.read_text(encoding="utf-8")
