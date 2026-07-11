import sys
from pathlib import Path

import src.ui as ui
from src.config_schema import PipelineConfig
from src.ui import build_config_dict, parse_sic_codes


def test_parse_sic_codes_accepts_commas_spaces_and_newlines():
    assert parse_sic_codes("3714, 3569\n3490 3562") == ["3714", "3569", "3490", "3562"]


def test_build_config_dict_validates_form_values_without_api_keys():
    base_config = {
        "target_company": {
            "name": "Base Co",
            "description": "Base description",
            "primary_sic_codes": ["1111", "2222"],
            "adjacent_sic_codes": ["3333"],
        },
        "universe": {
            "max_candidates": 300,
            "primary_allocation_pct": 0.5,
            "sic_clusters": {"1111": "base_cluster"},
        },
        "llm": {
            "extraction_model": "gpt-4.1",
            "judge_model": "gpt-4.1-mini",
            "embedding_model": "text-embedding-3-small",
            "temperature": 0,
            "max_tokens": 500,
            "batch_size": 20,
            "judge_threshold": 3,
        },
        "scorer": {
            "feature_weights": {"manufacturing": {"revenue_ttm_log": 1.5}},
            "ranking_penalties": {"subsector_similarity_threshold": 0.6},
        },
    }
    config = build_config_dict(
        base_config=base_config,
        target_name="Acme Parts",
        target_description="Manufacturer of engineered parts for automotive OEMs.",
        revenue_usd_mm=150,
        ebitda_margin_pct=18,
        gross_margin_pct="",
        revenue_cagr_3yr_pct="",
        net_debt_ebitda="",
        capex_revenue_pct="",
        primary_sic_codes="3714,3569",
        adjacent_sic_codes="3490\n3562",
        seed_tickers="aaa bbb",
        must_include_tickers="aaa, bbb",
        exclude_tickers="ccc\nddd",
        max_candidates=50,
        primary_allocation_pct=60,
        top_n_comps=10,
        size_marketability_discount_pct=25,
        prepared_by="Deal Team",
        confidential=True,
    )

    validated = PipelineConfig.model_validate(config)

    assert validated.target_company.name == "Acme Parts"
    assert validated.target_company.ebitda_margin_estimate == 0.18
    assert validated.target_company.gross_margin_estimate is None
    assert validated.target_company.revenue_cagr_3yr_estimate is None
    assert validated.target_company.net_debt_ebitda_estimate is None
    assert validated.target_company.capex_revenue_estimate is None
    assert validated.target_company.primary_sic_codes == ["3714", "3569"]
    assert validated.target_company.adjacent_sic_codes == ["3490", "3562"]
    assert validated.universe.seed_tickers == ["AAA", "BBB"]
    assert validated.universe.must_include_tickers == ["AAA", "BBB"]
    assert validated.universe.exclude_tickers == ["CCC", "DDD"]
    assert validated.universe.sic_clusters == {"1111": "base_cluster"}
    assert validated.universe.primary_allocation_pct == 0.6
    assert validated.output.top_n_comps == 10
    assert validated.valuation.size_marketability_discount == 0.25
    assert validated.scorer.feature_weights == {"manufacturing": {"revenue_ttm_log": 1.5}}
    assert validated.scorer.ranking_penalties.subsector_similarity_threshold == 0.6
    assert "OPENAI_API_KEY" not in str(config)
    assert "FMP_API_KEY" not in str(config)


def test_run_from_form_copies_outputs_and_returns_report_actions(tmp_path, mocker, monkeypatch):
    monkeypatch.setattr(ui, "RUNS_DIR", tmp_path / "ui_runs")
    source_dir = tmp_path / "source_outputs"
    source_dir.mkdir()
    html_report = source_dir / "comps_report.html"
    csv_report = source_dir / "comps_report.csv"
    html_report.write_text("<html><body>Report Body</body></html>", encoding="utf-8")
    csv_report.write_text("rank,ticker\n1,AAA\n", encoding="utf-8")
    mocker.patch("src.ui.run_pipeline", return_value={"html": str(html_report), "csv": str(csv_report)})

    status, report_actions, html_file, csv_file = ui.run_from_form(
        "Acme Parts",
        "Manufacturer of engineered parts for automotive OEMs.",
        150,
        18,
        None,
        None,
        None,
        None,
        "3714",
        "3569",
        "",
        "",
        "",
        50,
        60,
        10,
        25,
        "Deal Team",
        True,
    )

    assert "Run complete" in status
    assert "Report Body" not in report_actions
    assert "HTML report ready" in report_actions
    assert "CSV report ready" in report_actions
    assert html_file.endswith("comps_report.html")
    assert csv_file.endswith("comps_report.csv")
    assert (tmp_path / "ui_runs").exists()


def test_run_from_form_surfaces_small_sample_warning_in_status(tmp_path, mocker, monkeypatch):
    monkeypatch.setattr(ui, "RUNS_DIR", tmp_path / "ui_runs")
    source_dir = tmp_path / "source_outputs"
    source_dir.mkdir()
    html_report = source_dir / "comps_report.html"
    html_report.write_text("<html><body>Report Body</body></html>", encoding="utf-8")
    mocker.patch("src.ui.run_pipeline", return_value={
        "html": str(html_report),
        "small_sample_warning": "Small comp pool: only 9 eligible companies survived filtering",
    })

    status, _, _, _ = ui.run_from_form(
        "Acme Parts",
        "Manufacturer of engineered parts for automotive OEMs.",
        150,
        18,
        None,
        None,
        None,
        None,
        "3714",
        "3569",
        "",
        "",
        "",
        50,
        60,
        10,
        25,
        "Deal Team",
        True,
    )

    assert "Run complete" in status
    assert "WARNING: Small comp pool: only 9 eligible companies" in status


def test_run_from_form_failure_writes_error_log_and_reports_reason(tmp_path, mocker, monkeypatch):
    monkeypatch.setattr(ui, "RUNS_DIR", tmp_path / "ui_runs")
    mocker.patch("src.ui.run_pipeline", side_effect=RuntimeError("SEC discovery returned no tickers"))

    status, html_preview, html_file, csv_file = ui.run_from_form(
        "Acme Parts",
        "Manufacturer of engineered parts for automotive OEMs.",
        150,
        18,
        None,
        None,
        None,
        None,
        "3714",
        "3569",
        "",
        "",
        "",
        50,
        60,
        10,
        25,
        "Deal Team",
        True,
    )

    assert "Run failed" in status
    assert "SEC discovery returned no tickers" in status
    assert html_preview == ""
    assert html_file is None
    assert csv_file is None

    run_dirs = list((tmp_path / "ui_runs").iterdir())
    assert len(run_dirs) == 1
    error_log = run_dirs[0] / "error.log"
    assert error_log.exists()
    assert "RuntimeError: SEC discovery returned no tickers" in error_log.read_text(encoding="utf-8")
    assert (run_dirs[0] / "config.yaml").exists()


def test_run_from_form_validates_required_target_description_and_sic_codes(mocker):
    mock_run = mocker.patch("src.ui.run_pipeline")

    status, html_preview, html_file, csv_file = ui.run_from_form(
        "",
        "",
        None,
        None,
        None,
        None,
        None,
        None,
        "",
        "",
        "",
        "",
        "",
        50,
        60,
        10,
        25,
        "",
        False,
    )

    assert "Missing required inputs" in status
    assert "Target company" in status
    assert "Business description" in status
    assert "Primary or Adjacent SIC codes" in status
    assert html_preview == ""
    assert html_file is None
    assert csv_file is None
    mock_run.assert_not_called()


def test_build_app_keeps_demo_values_as_reference_not_input_defaults():
    app = ui.build_app()
    components = app.config["components"]

    def props_for_label(label):
        return next(component["props"] for component in components if component["props"].get("label") == label)

    assert props_for_label("Target company").get("value") == ""
    assert props_for_label("Prepared by").get("value") == ""
    assert props_for_label("Business description").get("value") == ""
    assert props_for_label("Revenue ($mm)").get("value") == ""
    assert props_for_label("EBITDA margin (%)").get("value") == ""
    assert props_for_label("Primary SIC codes").get("value") == ""
    assert props_for_label("Adjacent SIC codes").get("value") == ""

    config_text = str(app.config)
    assert "Reference example" in config_text
    assert "Precision Motion Components Co." in config_text
    assert "PE Comps Pipeline Demo" in config_text


def test_suggest_sic_codes_from_description_returns_rows_and_advisory_status(mocker):
    mock_suggest = mocker.patch(
        "src.ui.llm_analyzer.suggest_sic_codes",
        return_value=[
            {
                "bucket": "primary",
                "sic_code": "3714",
                "title": "Motor Vehicle Parts and Accessories",
                "confidence": "high",
                "reason": "Matches the target's parts manufacturing focus.",
            },
            {
                "bucket": "adjacent",
                "sic_code": "3569",
                "title": "General Industrial Machinery",
                "confidence": "medium",
                "reason": "Can broaden the candidate pool.",
            },
        ],
    )

    status, rows, primary_codes, adjacent_codes = ui.suggest_sic_codes_from_description(
        "Engineered automotive parts manufacturer."
    )

    assert "Advisory only" in status
    assert "populated Primary and Adjacent SIC codes" in status
    assert "validated against the SEC SIC list" in status
    assert rows == [
        ["primary", "3714", "Motor Vehicle Parts and Accessories", "high", "Matches the target's parts manufacturing focus."],
        ["adjacent", "3569", "General Industrial Machinery", "medium", "Can broaden the candidate pool."],
    ]
    assert primary_codes == "3714"
    assert adjacent_codes == "3569"
    config = mock_suggest.call_args.args[0]
    assert config["target_company"]["description"] == "Engineered automotive parts manufacturer."
    assert config["target_company"]["primary_sic_codes"] == []
    assert config["target_company"]["adjacent_sic_codes"] == []


def test_suggest_sic_codes_from_description_handles_empty_results(mocker):
    mocker.patch("src.ui.llm_analyzer.suggest_sic_codes", return_value=[])

    status, rows, primary_codes, adjacent_codes = ui.suggest_sic_codes_from_description(
        "Engineered automotive parts manufacturer."
    )

    assert "No SIC code suggestions" in status
    assert rows == []
    assert primary_codes == ""
    assert adjacent_codes == ""


def test_main_launches_with_host_and_port_args(mocker, monkeypatch):
    class FakeApp:
        def queue(self, default_concurrency_limit):
            self.default_concurrency_limit = default_concurrency_limit
            return self

        # Signature mirrors gradio's Blocks.launch(), where CSS is passed.
        def launch(self, server_name, server_port, css):
            self.server_name = server_name
            self.server_port = server_port
            self.css = css

    fake_app = FakeApp()
    mocker.patch("src.ui.build_app", return_value=fake_app)
    monkeypatch.setattr(sys, "argv", ["pe-comps-ui", "--host", "0.0.0.0", "--port", "8015"])

    ui.main()

    assert fake_app.default_concurrency_limit == 1
    assert fake_app.server_name == "0.0.0.0"
    assert fake_app.server_port == 8015
    assert fake_app.css == ui.APP_CSS


def test_build_app_applies_full_width_css():
    app = ui.build_app()

    assert "max-width: none" in (app.css or "")


def test_disable_gradio_analytics_sets_opt_out_env(monkeypatch):
    monkeypatch.delenv("GRADIO_ANALYTICS_ENABLED", raising=False)

    ui.disable_gradio_analytics()

    assert ui.os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"


def test_build_app_uses_full_width_layout():
    app = ui.build_app()

    assert app.fill_width is True
    assert "max-width: none" in ui.APP_CSS


def test_build_app_links_reports_without_full_html_preview():
    app = ui.build_app()
    config_text = str(app.config)

    assert "Run outputs" in config_text
    assert "HTML report preview" not in config_text
    assert "report-preview" not in config_text


def test_ui_source_has_no_chinese_copy():
    source = Path("src/ui.py").read_text(encoding="utf-8")

    assert not any("\u4e00" <= char <= "\u9fff" for char in source)
