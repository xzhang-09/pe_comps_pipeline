import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import src.pipeline as pipeline


def _write_config(tmp_path: Path) -> Path:
    config = {
        "target_company": {
            "name": "Example Manufacturing Co.",
            "description": "A test target company.",
            "gics_sector": "20",
            "revenue_usd_mm": 150,
            "ebitda_margin_estimate": 0.18,
            "primary_sic_codes": ["3714"],
            "adjacent_sic_codes": ["3612"],
        },
        "universe": {"max_candidates": 10, "primary_allocation_pct": 0.8},
        "llm": {
            "extraction_model": "gpt-4.1", "judge_model": "gpt-4.1-mini",
            "temperature": 0, "max_tokens": 500, "batch_size": 20, "judge_threshold": 3,
        },
        "output": {"top_n_comps": 15, "report_formats": ["csv", "html"]},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return path


def _mock_all_steps(mocker, tmp_path):
    companies = [{"ticker": "AAA", "ev_ebitda": 12.0}, {"ticker": "BBB", "ev_ebitda": None}]
    llm_features = {
        "AAA": {"business_model": "manufacturing", "extraction_failed": False, "low_confidence_flag": False},
        "BBB": {"business_model": "manufacturing", "extraction_failed": True, "low_confidence_flag": False},
    }
    target_llm_features = {"business_model": "manufacturing"}
    feature_matrix = pd.DataFrame({"ebitda_margin": [0.2], "ev_ebitda_log": [2.5]}, index=["AAA"])
    imputation_medians = {"ebitda_margin": 0.2}
    company_scores = pd.DataFrame({"residual_abs": [0.5]}, index=["AAA"])
    scorer_results = {
        "company_scores": company_scores,
        "feature_distance_sq_diff": pd.DataFrame({"ebitda_margin": [0.1]}, index=["AAA"]),
    }

    def _fake_generate(*args, **kwargs):
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        csv_path = outputs_dir / "comps_report.csv"
        html_path = outputs_dir / "comps_report.html"
        csv_path.write_text("rank,ticker\n1,AAA\n", encoding="utf-8")
        html_path.write_text("<html>Example Manufacturing Co.</html>", encoding="utf-8")
        return {"csv": str(csv_path), "html": str(html_path)}

    mocks = {
        "universe_builder.build": mocker.patch("src.pipeline.universe_builder.build", return_value=["AAA", "BBB"]),
        "fetcher.fetch_batch": mocker.patch("src.pipeline.fetcher.fetch_batch", return_value=companies),
        "llm_analyzer.analyze_batch": mocker.patch("src.pipeline.llm_analyzer.analyze_batch", return_value=llm_features),
        "llm_analyzer.analyze_target": mocker.patch("src.pipeline.llm_analyzer.analyze_target", return_value=target_llm_features),
        "feature_builder.build": mocker.patch(
            "src.pipeline.feature_builder.build",
            return_value=(feature_matrix, pd.Series([12.0], index=["AAA"]), imputation_medians),
        ),
        "scorer.run": mocker.patch("src.pipeline.scorer.run", return_value=scorer_results),
        "reporter.generate": mocker.patch("src.pipeline.reporter.generate", side_effect=_fake_generate),
    }
    return mocks


def test_pipeline_calls_all_six_steps(mocker, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path)
    mocks = _mock_all_steps(mocker, tmp_path)

    pipeline.run_pipeline(str(config_path))

    for name, mock in mocks.items():
        assert mock.call_count == 1, f"{name} was called {mock.call_count} times, expected 1"


def test_pipeline_logs_step_headers(mocker, tmp_path):
    config_path = _write_config(tmp_path)
    _mock_all_steps(mocker, tmp_path)

    log_path = Path("logs/pipeline.log")
    log_path.parent.mkdir(exist_ok=True)
    before_size = log_path.stat().st_size if log_path.exists() else 0

    pipeline.run_pipeline(str(config_path))

    with open(log_path, encoding="utf-8") as f:
        f.seek(before_size)
        new_content = f.read()

    for i in range(1, 7):
        assert f"STEP {i}/6" in new_content


def test_pipeline_creates_required_directories(mocker, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path)
    _mock_all_steps(mocker, tmp_path)

    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "logs").exists()

    pipeline.run_pipeline(str(config_path))

    assert (tmp_path / "outputs").exists()
    assert (tmp_path / "logs").exists()


def test_main_suggest_sic_codes_prints_suggestions_without_running_pipeline(mocker, tmp_path, monkeypatch, capsys):
    config_path = _write_config(tmp_path)
    mock_suggest = mocker.patch(
        "src.pipeline.llm_analyzer.suggest_sic_codes",
        return_value=[
            {"sic_code": "3714", "title": "Motor Vehicle Parts", "bucket": "primary", "reason": "matches", "confidence": "high"},
        ],
    )
    mock_run = mocker.patch("src.pipeline.run_pipeline")

    monkeypatch.setattr(sys, "argv", ["pe-comps", "--config", str(config_path), "--suggest-sic-codes"])

    pipeline.main()

    output = capsys.readouterr().out
    assert "SIC 3714" in output
    assert "Motor Vehicle Parts" in output
    mock_suggest.assert_called_once()
    mock_run.assert_not_called()


def test_main_suggest_sic_codes_handles_empty_suggestions(mocker, tmp_path, monkeypatch, capsys):
    config_path = _write_config(tmp_path)
    mocker.patch("src.pipeline.llm_analyzer.suggest_sic_codes", return_value=[])
    mock_run = mocker.patch("src.pipeline.run_pipeline")

    monkeypatch.setattr(sys, "argv", ["pe-comps", "--config", str(config_path), "--suggest-sic-codes"])

    pipeline.main()

    output = capsys.readouterr().out
    assert "No SIC code suggestions" in output
    mock_run.assert_not_called()


def test_score_and_report_reuses_universe_without_re_enriching(mocker, tmp_path, monkeypatch):
    # The payoff of the enrichment/scoring split: an already-enriched universe can
    # be scored repeatedly (e.g. a tuning loop varying config) without re-running
    # any of the expensive discover/fetch/extract work.
    monkeypatch.chdir(tmp_path)
    config = pipeline._load_config(str(_write_config(tmp_path)))

    universe = pipeline.EnrichedUniverse(
        companies=[{"ticker": "AAA", "ev_ebitda": 12.0}],
        llm_features={"AAA": {"business_model": "manufacturing", "extraction_failed": False, "low_confidence_flag": False}},
        feature_matrix=pd.DataFrame({"ebitda_margin": [0.2], "ev_ebitda_log": [2.5]}, index=["AAA"]),
        ev_ebitda_raw=pd.Series([12.0], index=["AAA"]),
        imputation_medians={"ebitda_margin": 0.2},
    )

    enrich_mocks = {
        "universe_builder.build": mocker.patch("src.pipeline.universe_builder.build"),
        "fetcher.fetch_batch": mocker.patch("src.pipeline.fetcher.fetch_batch"),
        "llm_analyzer.analyze_batch": mocker.patch("src.pipeline.llm_analyzer.analyze_batch"),
        "feature_builder.build": mocker.patch("src.pipeline.feature_builder.build"),
    }
    mocker.patch("src.pipeline.llm_analyzer.analyze_target", return_value={"business_model": "manufacturing"})
    scorer_run = mocker.patch(
        "src.pipeline.scorer.run",
        return_value={
            "company_scores": pd.DataFrame({"residual_abs": [0.5]}, index=["AAA"]),
            "feature_distance_sq_diff": pd.DataFrame({"ebitda_margin": [0.1]}, index=["AAA"]),
        },
    )
    generate = mocker.patch("src.pipeline.reporter.generate", return_value={"csv": "x.csv", "n_comps": 1})

    pipeline.score_and_report(universe, config)
    pipeline.score_and_report(universe, config)

    for name, m in enrich_mocks.items():
        assert not m.called, f"{name} was called during scoring — universe was re-enriched"
    assert scorer_run.call_count == 2     # scored twice off the one enriched universe
    assert generate.call_count == 2


def test_universe_artifact_round_trips(tmp_path):
    feature_matrix = pd.DataFrame(
        {"ebitda_margin": [0.2, 0.3], "ev_ebitda_log": [np.log1p(12.0), np.log1p(8.0)]},
        index=["AAA", "BBB"],
    )
    universe = pipeline.EnrichedUniverse(
        companies=[{"ticker": "AAA", "ev_ebitda": 12.0}, {"ticker": "BBB", "ev_ebitda": 8.0}],
        llm_features={"AAA": {"business_model": "manufacturing"}, "BBB": {"business_model": "services"}},
        feature_matrix=feature_matrix,
        ev_ebitda_raw=pd.Series([12.0, 8.0], index=["AAA", "BBB"], name="ev_ebitda"),
        imputation_medians={"global": {"ebitda_margin": 0.25}},
    )

    pipeline.save_universe(universe, tmp_path / "u")
    loaded = pipeline.load_universe(tmp_path / "u")

    assert loaded.companies == universe.companies
    assert loaded.llm_features == universe.llm_features
    assert loaded.imputation_medians == universe.imputation_medians
    pd.testing.assert_frame_equal(loaded.feature_matrix, universe.feature_matrix)
    # ev_ebitda_raw isn't persisted; it's recomputed from the matrix label column.
    pd.testing.assert_series_equal(loaded.ev_ebitda_raw, universe.ev_ebitda_raw, check_names=False)


def test_load_universe_rejects_incompatible_schema(tmp_path):
    universe = pipeline.EnrichedUniverse(
        companies=[{"ticker": "AAA", "ev_ebitda": 12.0}],
        llm_features={"AAA": {"business_model": "manufacturing"}},
        feature_matrix=pd.DataFrame({"ev_ebitda_log": [np.log1p(12.0)]}, index=["AAA"]),
        ev_ebitda_raw=pd.Series([12.0], index=["AAA"]),
        imputation_medians={},
    )
    pipeline.save_universe(universe, tmp_path / "u")
    # Corrupt the manifest's schema version — a stale artifact must be refused, not scored.
    manifest_path = tmp_path / "u" / "manifest.json"
    manifest_path.write_text('{"schema_version": 999}', encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        pipeline.load_universe(tmp_path / "u")


def test_run_pipeline_reuse_universe_skips_enrichment(mocker, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path)

    feature_matrix = pd.DataFrame({"ebitda_margin": [0.2], "ev_ebitda_log": [np.log1p(12.0)]}, index=["AAA"])
    universe = pipeline.EnrichedUniverse(
        companies=[{"ticker": "AAA", "ev_ebitda": 12.0}],
        llm_features={"AAA": {"business_model": "manufacturing", "extraction_failed": False, "low_confidence_flag": False}},
        feature_matrix=feature_matrix,
        ev_ebitda_raw=pd.Series([12.0], index=["AAA"]),
        imputation_medians={"ebitda_margin": 0.2},
    )
    pipeline.save_universe(universe, tmp_path / "u")

    enrich_build = mocker.patch("src.pipeline.universe_builder.build")
    mocker.patch("src.pipeline.fetcher.fetch_batch")
    mocker.patch("src.pipeline.llm_analyzer.analyze_batch")
    mocker.patch("src.pipeline.feature_builder.build")
    mocker.patch("src.pipeline.llm_analyzer.analyze_target", return_value={"business_model": "manufacturing"})
    mocker.patch("src.pipeline.scorer.run", return_value={
        "company_scores": pd.DataFrame({"residual_abs": [0.5]}, index=["AAA"]),
        "feature_distance_sq_diff": pd.DataFrame({"ebitda_margin": [0.1]}, index=["AAA"]),
    })
    generate = mocker.patch("src.pipeline.reporter.generate", return_value={"csv": "x.csv", "n_comps": 1})

    pipeline.run_pipeline(str(config_path), reuse_universe=str(tmp_path / "u"))

    enrich_build.assert_not_called()   # steps 1-4 skipped entirely
    generate.assert_called_once()
