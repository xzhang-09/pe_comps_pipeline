from pathlib import Path

import pandas as pd
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
        "model": None,
        "feature_columns": ["ebitda_margin"],
        "cv_rmse_log": 0.4,
        "cv_mae_median": 5.0,
        "cv_rmse_multiple_space": 5.0,
        "feature_importance": pd.DataFrame({"feature": ["ebitda_margin"], "mean_abs_shap": [0.1]}),
        "company_scores": company_scores,
        "target_prediction": {"predicted_ev_ebitda": 15.0, "range_low": 10.0, "range_high": 22.0, "cv_rmse_log": 0.4, "cv_mae_median": 5.0},
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

    pipeline.run_pipeline(str(config_path), force_retrain=False)

    for name, mock in mocks.items():
        assert mock.call_count == 1, f"{name} was called {mock.call_count} times, expected 1"


def test_pipeline_logs_step_headers(mocker, tmp_path):
    config_path = _write_config(tmp_path)
    _mock_all_steps(mocker, tmp_path)

    log_path = Path("logs/pipeline.log")
    log_path.parent.mkdir(exist_ok=True)
    before_size = log_path.stat().st_size if log_path.exists() else 0

    pipeline.run_pipeline(str(config_path), force_retrain=False)

    with open(log_path, "r", encoding="utf-8") as f:
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

    pipeline.run_pipeline(str(config_path), force_retrain=False)

    assert (tmp_path / "outputs").exists()
    assert (tmp_path / "logs").exists()
