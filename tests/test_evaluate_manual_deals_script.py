from pathlib import Path

import pandas as pd
import yaml

from scripts import evaluate_manual_deals
from src.pipeline import EnrichedUniverse


def _write_config(tmp_path: Path) -> Path:
    config = {
        "target_company": {
            "name": "Base Target",
            "description": "Base description",
            "revenue_usd_mm": 100.0,
            "ebitda_margin_estimate": 0.1,
            "primary_sic_codes": ["3714"],
            "adjacent_sic_codes": ["3569"],
        },
        "universe": {"max_candidates": 10, "primary_allocation_pct": 1.0},
        "llm": {
            "extraction_model": "gpt-4.1",
            "judge_model": "gpt-4.1-mini",
            "temperature": 0,
            "max_tokens": 500,
            "batch_size": 10,
            "judge_threshold": 3,
            "embedding_model": "text-embedding-3-small",
        },
        "output": {"top_n_comps": 15, "report_formats": ["csv"]},
        "scorer": {
            "ranking_penalties": {
                "business_model_penalty": 0.6,
                "customer_type_penalty": 0.5,
                "subsector_similarity_threshold": 0.5,
                "subsector_mismatch_penalty": 0.4,
                "size_penalty_free_log10_range": 1.0,
                "size_penalty_per_extra_log10": 1.0,
            },
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _write_manual_deals(tmp_path: Path) -> Path:
    path = tmp_path / "manual_deals.json"
    path.write_text(
        """
        [
          {
            "deal_id": "demo-2026",
            "target_ticker": "TGT",
            "target_name": "Target Co",
            "target_cik": "0000000000",
            "target_sic": "3559",
            "business_description": "Makes engineered industrial equipment.",
            "target_financials": {
              "revenue_usd_mm": 250.0,
              "ebitda_margin_estimate": 0.18,
              "source": "manual test"
            },
            "filing_url": "https://www.sec.gov/example",
            "filing_date": "2026-01-15",
            "advisor": "Example Bank",
            "selected_companies": [
              {"ticker": "HIT", "company_name": "Hit Co", "still_public": true},
              {"ticker": "MISS", "company_name": "Miss Co", "still_public": true},
              {"ticker": "OLD", "company_name": "Old Co", "still_public": false}
            ],
            "review_status": "reviewed"
          }
        ]
        """,
        encoding="utf-8",
    )
    return path


def test_evaluate_manual_deals_runs_each_deal_and_writes_outputs(mocker, tmp_path):
    config_path = _write_config(tmp_path)
    manual_deals_path = _write_manual_deals(tmp_path)
    output_json = tmp_path / "outputs" / "manual_results.json"
    results_md = tmp_path / "eval" / "results.md"

    universe = EnrichedUniverse(
        companies=[
            {"ticker": "HIT", "revenue_ttm_usd_mm": 240.0},
            {"ticker": "DISTRACTOR", "revenue_ttm_usd_mm": 260.0},
        ],
        llm_features={
            "HIT": {"business_model": "manufacturing", "customer_type": "B2B", "low_confidence_flag": False, "sub_sector_description": None},
            "DISTRACTOR": {"business_model": "manufacturing", "customer_type": "B2B", "low_confidence_flag": False, "sub_sector_description": None},
        },
        feature_matrix=pd.DataFrame(index=["HIT", "DISTRACTOR"]),
        ev_ebitda_raw=pd.Series(dtype=float),
        imputation_medians={"revenue_ttm_log": 1.0, "ebitda_margin": 0.2},
    )
    enrich_universe = mocker.patch("scripts.evaluate_manual_deals.pipeline.enrich_universe", return_value=universe)
    analyze_target = mocker.patch(
        "scripts.evaluate_manual_deals.llm_analyzer.analyze_target",
        return_value={"business_model": "manufacturing", "customer_type": "B2B", "low_confidence_flag": False, "sub_sector_description": None},
    )
    scorer_run = mocker.patch(
        "scripts.evaluate_manual_deals.scorer.run",
        return_value={
            "company_scores": pd.DataFrame(
                {"residual_abs": {"HIT": 0.1, "DISTRACTOR": 0.2}},
            ),
        },
    )
    mocker.patch(
        "scripts.evaluate_manual_deals.coverage.deal_discovery_sets",
        return_value=({"MISS"}, set()),
    )

    results = evaluate_manual_deals.evaluate_manual_deals(
        config_path=config_path,
        manual_deals_path=manual_deals_path,
        output_json_path=output_json,
        results_md_path=results_md,
        k=1,
        validate_size=False,
    )

    assert results["mean_precision"] == 0.5
    row = results["per_deal"][0]
    assert row["hits"] == ["HIT"]
    assert row["missed_not_in_universe"] == ["MISS"]
    assert row["excluded_delisted_tickers"] == ["OLD"]
    # MISS is in the SIC filer set but not among candidates -> quota truncation
    assert row["coverage"] == {"HIT": "hit", "MISS": "truncated_by_max_candidates"}
    assert row["reachable_precision"] == 1.0
    assert row["n_scored"] == 2
    assert row["n_selectable"] == 2
    assert row["selection_trivial"] is False
    assert results["coverage_waterfall"] == {"hit": 1, "truncated_by_max_candidates": 1}
    assert results["mean_reachable_precision"] == 1.0
    assert results["n_failed_deals"] == 0
    assert output_json.exists()
    report = results_md.read_text(encoding="utf-8")
    assert "Precision@1" in report
    assert "demo-2026" in report
    assert "Coverage Waterfall" in report
    assert "truncated_by_max_candidates" in report

    cfg = enrich_universe.call_args.args[0]
    assert cfg.target_company.name == "Target Co"
    assert cfg.target_company.description == "Makes engineered industrial equipment."
    assert cfg.target_company.revenue_usd_mm == 250.0
    assert cfg.target_company.ebitda_margin_estimate == 0.18
    assert cfg.target_company.primary_sic_codes == ["3559"]
    assert cfg.target_company.adjacent_sic_codes == []
    assert cfg.universe.allow_broad_sic_codes is True
    # Full quota to the primary bucket (empty adjacent bucket doesn't refund
    # its share) and a revenue band scaled to the deal target, not the demo
    # target's absolute band.
    assert cfg.universe.primary_allocation_pct == 1.0
    assert cfg.universe.min_revenue_usd_mm == 25.0
    assert cfg.universe.max_revenue_usd_mm == 2500.0
    analyze_target.assert_called_once()
    scorer_run.assert_called_once()


def test_failed_deal_is_recorded_and_does_not_stop_the_run(mocker, tmp_path):
    config_path = _write_config(tmp_path)
    manual_deals_path = _write_manual_deals(tmp_path)

    mocker.patch(
        "scripts.evaluate_manual_deals.pipeline.enrich_universe",
        side_effect=RuntimeError("SEC preflight aborted"),
    )

    results = evaluate_manual_deals.evaluate_manual_deals(
        config_path=config_path,
        manual_deals_path=manual_deals_path,
        output_json_path=tmp_path / "results.json",
        results_md_path=tmp_path / "results.md",
        k=1,
        validate_size=False,
    )

    row = results["per_deal"][0]
    assert results["n_failed_deals"] == 1
    assert row["error"] == "RuntimeError: SEC preflight aborted"
    assert row["precision"] is None
    assert results["mean_precision"] == 0.0
    assert "RUN FAILED: RuntimeError" in (tmp_path / "results.md").read_text(encoding="utf-8")


def test_only_deal_ids_filters_and_rejects_unknown(mocker, tmp_path):
    import pytest

    config_path = _write_config(tmp_path)
    manual_deals_path = _write_manual_deals(tmp_path)
    enrich = mocker.patch(
        "scripts.evaluate_manual_deals.pipeline.enrich_universe",
        side_effect=RuntimeError("should not matter"),
    )

    with pytest.raises(ValueError, match="Unknown deal ids"):
        evaluate_manual_deals.evaluate_manual_deals(
            config_path=config_path,
            manual_deals_path=manual_deals_path,
            output_json_path=tmp_path / "results.json",
            results_md_path=tmp_path / "results.md",
            validate_size=False,
            only_deal_ids=["nope-2020"],
        )
    enrich.assert_not_called()

    results = evaluate_manual_deals.evaluate_manual_deals(
        config_path=config_path,
        manual_deals_path=manual_deals_path,
        output_json_path=tmp_path / "results.json",
        results_md_path=tmp_path / "results.md",
        validate_size=False,
        only_deal_ids=["demo-2026"],
    )
    assert [r["deal_id"] for r in results["per_deal"]] == ["demo-2026"]


def test_suggested_discovery_codes_dedupes_and_drops_zero_yield(mocker, tmp_path):
    from src.config_schema import PipelineConfig

    config_path = _write_config(tmp_path)
    base_config = PipelineConfig.model_validate(yaml.safe_load(config_path.read_text()))
    deal = {
        "deal_id": "demo-2026",
        "target_ticker": "TGT",
        "target_name": "Target Co",
        "target_sic": "3559",
        "business_description": "Makes engineered industrial equipment.",
        "target_financials": {"revenue_usd_mm": 250.0, "ebitda_margin_estimate": 0.18},
    }
    suggest = mocker.patch(
        "scripts.evaluate_manual_deals.llm_analyzer.suggest_sic_codes",
        return_value=[
            {"sic_code": "3559", "bucket": "primary"},   # target's own code, dropped
            {"sic_code": "3714", "bucket": "primary"},
            {"sic_code": "3714", "bucket": "adjacent"},  # already primary, dropped
            {"sic_code": "3569", "bucket": "adjacent"},
            {"sic_code": "9999", "bucket": "adjacent"},  # zero-yield, dropped
        ],
    )
    mocker.patch(
        "scripts.evaluate_manual_deals.sic_universe_builder.preflight_sic_codes",
        return_value={"3714": 40, "3569": 12, "9999": 0},
    )

    codes = evaluate_manual_deals._suggested_discovery_codes(deal, base_config)

    assert codes == {"primary": ["3714"], "adjacent": ["3569"]}
    # The suggestion call must see a raised output-token budget.
    suggestion_config = suggest.call_args.args[0]
    assert suggestion_config.llm.max_tokens >= evaluate_manual_deals.SUGGESTION_MIN_MAX_TOKENS


def test_suggest_sic_mode_expands_discovery_and_labels_results(mocker, tmp_path):
    config_path = _write_config(tmp_path)
    manual_deals_path = _write_manual_deals(tmp_path)

    universe = EnrichedUniverse(
        companies=[{"ticker": "HIT", "revenue_ttm_usd_mm": 240.0}],
        llm_features={"HIT": {"business_model": "manufacturing", "customer_type": "B2B", "low_confidence_flag": False, "sub_sector_description": None}},
        feature_matrix=pd.DataFrame(index=["HIT"]),
        ev_ebitda_raw=pd.Series(dtype=float),
        imputation_medians={},
    )
    enrich_universe = mocker.patch("scripts.evaluate_manual_deals.pipeline.enrich_universe", return_value=universe)
    mocker.patch(
        "scripts.evaluate_manual_deals.llm_analyzer.analyze_target",
        return_value={"business_model": "manufacturing", "customer_type": "B2B", "low_confidence_flag": False, "sub_sector_description": None},
    )
    mocker.patch(
        "scripts.evaluate_manual_deals.scorer.run",
        return_value={"company_scores": pd.DataFrame({"residual_abs": {"HIT": 0.1}})},
    )
    mocker.patch("scripts.evaluate_manual_deals.coverage.deal_discovery_sets", return_value=(set(), set()))
    mocker.patch(
        "scripts.evaluate_manual_deals._suggested_discovery_codes",
        return_value={"primary": ["3714"], "adjacent": ["3569"]},
    )

    results = evaluate_manual_deals.evaluate_manual_deals(
        config_path=config_path,
        manual_deals_path=manual_deals_path,
        output_json_path=tmp_path / "results.json",
        results_md_path=tmp_path / "results.md",
        k=1,
        validate_size=False,
        discovery="suggest-sic",
    )

    cfg = enrich_universe.call_args.args[0]
    assert cfg.target_company.primary_sic_codes == ["3559", "3714"]
    assert cfg.target_company.adjacent_sic_codes == ["3569"]
    assert cfg.universe.primary_allocation_pct == 0.7
    assert results["discovery_mode"] == "suggest-sic"
    row = results["per_deal"][0]
    assert row["discovery_mode"] == "suggest-sic"
    assert row["discovery_sic_codes"] == {"primary": ["3559", "3714"], "adjacent": ["3569"]}
    report = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "Discovery mode: suggest-sic" in report
    assert "primary 3559, 3714; adjacent 3569" in report


def test_embedding_discovery_mode_enables_semantic_universe_without_suggestions(mocker, tmp_path):
    config_path = _write_config(tmp_path)
    manual_deals_path = _write_manual_deals(tmp_path)

    universe = EnrichedUniverse(
        companies=[{"ticker": "HIT", "revenue_ttm_usd_mm": 240.0}],
        llm_features={"HIT": {"business_model": "manufacturing", "customer_type": "B2B", "low_confidence_flag": False, "sub_sector_description": None}},
        feature_matrix=pd.DataFrame(index=["HIT"]),
        ev_ebitda_raw=pd.Series(dtype=float),
        imputation_medians={},
    )
    enrich_universe = mocker.patch("scripts.evaluate_manual_deals.pipeline.enrich_universe", return_value=universe)
    suggest = mocker.patch("scripts.evaluate_manual_deals._suggested_discovery_codes")
    mocker.patch(
        "scripts.evaluate_manual_deals.llm_analyzer.analyze_target",
        return_value={"business_model": "manufacturing", "customer_type": "B2B", "low_confidence_flag": False, "sub_sector_description": None},
    )
    mocker.patch(
        "scripts.evaluate_manual_deals.scorer.run",
        return_value={"company_scores": pd.DataFrame({"residual_abs": {"HIT": 0.1}})},
    )
    mocker.patch("scripts.evaluate_manual_deals.coverage.deal_discovery_sets", return_value=(set(), set()))

    results = evaluate_manual_deals.evaluate_manual_deals(
        config_path=config_path,
        manual_deals_path=manual_deals_path,
        output_json_path=tmp_path / "results.json",
        results_md_path=tmp_path / "results.md",
        k=1,
        validate_size=False,
        discovery="sic+embedding",
    )

    cfg = enrich_universe.call_args.args[0]
    assert cfg.universe.discovery_mode == "sic+embedding"
    assert cfg.target_company.primary_sic_codes == ["3559"]
    assert cfg.target_company.adjacent_sic_codes == []
    assert results["discovery_mode"] == "sic+embedding"
    suggest.assert_not_called()


def test_suggest_sic_embedding_mode_is_registered_and_delegates_to_production(mocker, tmp_path):
    config_path = _write_config(tmp_path)
    manual_deals_path = _write_manual_deals(tmp_path)
    universe = EnrichedUniverse(
        companies=[{"ticker": "HIT", "revenue_ttm_usd_mm": 240.0}],
        llm_features={"HIT": {"low_confidence_flag": False}},
        feature_matrix=pd.DataFrame(index=["HIT"]),
        ev_ebitda_raw=pd.Series(dtype=float),
        imputation_medians={},
    )
    enrich = mocker.patch("scripts.evaluate_manual_deals.pipeline.enrich_universe", return_value=universe)
    eval_suggest = mocker.patch("scripts.evaluate_manual_deals._suggested_discovery_codes")
    mocker.patch("scripts.evaluate_manual_deals.llm_analyzer.analyze_target", return_value={"low_confidence_flag": False})
    mocker.patch(
        "scripts.evaluate_manual_deals.scorer.run",
        return_value={"company_scores": pd.DataFrame({"residual_abs": {"HIT": 0.1}})},
    )
    mocker.patch("scripts.evaluate_manual_deals.coverage.deal_discovery_sets", return_value=(set(), set()))
    # Expansion happens on build()'s local config copy, so the eval must report
    # the post-expansion codes from the snapshot, not from deal_config.
    mocker.patch(
        "scripts.evaluate_manual_deals.universe_builder.last_discovery_snapshot",
        return_value={
            "discovery_mode": "suggest-sic+embedding",
            "primary_sic_codes": ["3559", "3561"],
            "adjacent_sic_codes": ["3569"],
        },
    )

    results = evaluate_manual_deals.evaluate_manual_deals(
        config_path=config_path,
        manual_deals_path=manual_deals_path,
        output_json_path=tmp_path / "results.json",
        results_md_path=tmp_path / "results.md",
        k=1,
        validate_size=False,
        discovery="suggest-sic+embedding",
    )

    assert enrich.call_args.args[0].universe.discovery_mode == "suggest-sic+embedding"
    assert enrich.call_args.args[0].universe.primary_allocation_pct == 0.7
    assert results["discovery_mode"] == "suggest-sic+embedding"
    eval_suggest.assert_not_called()
    row = results["per_deal"][0]
    assert row["discovery_sic_codes"] == {"primary": ["3559", "3561"], "adjacent": ["3569"]}


def test_unknown_discovery_mode_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="Unknown discovery mode"):
        evaluate_manual_deals.evaluate_manual_deals(
            config_path=_write_config(tmp_path),
            manual_deals_path=_write_manual_deals(tmp_path),
            validate_size=False,
            discovery="seed-tickers",
        )


def test_mode_suffixed_paths_only_rewrite_defaults():
    default = Path("outputs/eval/manual_deals/results.json")
    assert (
        evaluate_manual_deals._mode_suffixed(str(default), default, "suggest-sic")
        == "outputs/eval/manual_deals/results.suggest-sic.json"
    )
    assert (
        evaluate_manual_deals._mode_suffixed(str(default), default, "sic+embedding")
        == "outputs/eval/manual_deals/results.embedding.json"
    )
    assert evaluate_manual_deals._mode_suffixed(str(default), default, "single-sic") == str(default)
    assert evaluate_manual_deals._mode_suffixed("custom.json", default, "suggest-sic") == "custom.json"
