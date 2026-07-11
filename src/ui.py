from __future__ import annotations

import argparse
import copy
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src import llm_analyzer
from src.defaults import DEFAULT_LLM_CONFIG
from src.paths import project_path
from src.pipeline import run_pipeline

RUNS_DIR = project_path("outputs", "ui_runs")
DEFAULT_CONFIG_PATH = project_path("config.yaml")
SIC_SUGGESTION_HEADERS = ["Bucket", "SIC code", "Industry title", "Confidence", "Reason"]
RUN_FORM_FIELDS = [
    "target_name", "target_description", "revenue_usd_mm", "ebitda_margin_pct",
    "gross_margin_pct", "revenue_cagr_3yr_pct", "net_debt_ebitda", "capex_revenue_pct",
    "primary_sic_codes", "adjacent_sic_codes", "seed_tickers", "must_include_tickers", "exclude_tickers",
    "max_candidates", "primary_allocation_pct", "top_n_comps", "size_marketability_discount_pct",
    "prepared_by", "confidential",
]

APP_CSS = """
.gradio-container,
.contain,
.wrap {
    max-width: none !important;
}

.gradio-container {
    width: 100% !important;
}

#pe-comps-shell {
    max-width: 1440px;
    margin: 0 auto;
    padding: 0 24px;
}

#run-panel {
    position: sticky;
    top: 16px;
}
"""


def disable_gradio_analytics() -> None:
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


def parse_sic_codes(raw_value: str | None) -> list[str]:
    """Parse comma, whitespace, or newline separated SIC codes."""
    if not raw_value:
        return []
    normalized = raw_value.replace(",", " ")
    return [part.strip() for part in normalized.split() if part.strip()]


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _pct_to_ratio(value: float | int | str | None) -> float | None:
    if _is_blank(value):
        return None
    return float(value) / 100


def _optional_float(value: float | int | str | None) -> float | None:
    if _is_blank(value):
        return None
    return float(value)


def _ratio_to_pct(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value) * 100


def _ratio_to_pct_text(value: float | int | None) -> str:
    return "" if value is None else f"{float(value) * 100:g}"


def _optional_float_text(value: float | int | None) -> str:
    return "" if value is None else f"{float(value):g}"


def _reference_text(target_defaults: dict[str, Any], output_defaults: dict[str, Any]) -> str:
    target_name = target_defaults.get("name") or "Example Manufacturing Co."
    prepared_by = output_defaults.get("prepared_by") or "Deal Team"
    description = target_defaults.get(
        "description",
        "Mid-market industrial parts manufacturer serving automotive OEMs in North America. "
        "Primarily B2B with long-term supply contracts.",
    )
    return (
        "### Reference example\n"
        f"**Target company:** {target_name}\n\n"
        f"**Prepared by:** {prepared_by}\n\n"
        f"**Business description:** {description}"
    )


def load_base_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_config_dict(
    *,
    base_config: dict[str, Any] | None = None,
    target_name: str,
    target_description: str,
    revenue_usd_mm: float | None,
    ebitda_margin_pct: float | str | None,
    gross_margin_pct: float | str | None,
    revenue_cagr_3yr_pct: float | str | None,
    net_debt_ebitda: float | str | None,
    capex_revenue_pct: float | str | None,
    primary_sic_codes: str,
    adjacent_sic_codes: str,
    seed_tickers: str,
    must_include_tickers: str,
    exclude_tickers: str,
    max_candidates: int,
    primary_allocation_pct: float,
    top_n_comps: int,
    size_marketability_discount_pct: float,
    prepared_by: str | None,
    confidential: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config) if base_config else {}
    config.setdefault("target_company", {})
    config.setdefault("universe", {})
    config.setdefault(
        "llm",
        copy.deepcopy(DEFAULT_LLM_CONFIG),
    )
    config.setdefault("output", {})
    config.setdefault("valuation", {})
    config.setdefault("scorer", {})

    config["target_company"].update(
        {
            "name": target_name.strip(),
            "description": target_description.strip(),
            "revenue_usd_mm": _optional_float(revenue_usd_mm),
            "ebitda_margin_estimate": _pct_to_ratio(ebitda_margin_pct),
            "gross_margin_estimate": _pct_to_ratio(gross_margin_pct),
            "revenue_cagr_3yr_estimate": _pct_to_ratio(revenue_cagr_3yr_pct),
            "net_debt_ebitda_estimate": _optional_float(net_debt_ebitda),
            "capex_revenue_estimate": _pct_to_ratio(capex_revenue_pct),
            "primary_sic_codes": parse_sic_codes(primary_sic_codes),
            "adjacent_sic_codes": parse_sic_codes(adjacent_sic_codes),
        }
    )
    config["universe"].update(
        {
            "max_candidates": int(max_candidates),
            "primary_allocation_pct": float(primary_allocation_pct) / 100,
            "seed_tickers": parse_sic_codes(seed_tickers),
            "must_include_tickers": parse_sic_codes(must_include_tickers),
            "exclude_tickers": parse_sic_codes(exclude_tickers),
        }
    )
    config["output"].update(
        {
            "top_n_comps": int(top_n_comps),
            "report_formats": ["csv", "html"],
            "prepared_by": (prepared_by or "").strip() or None,
            "confidential": bool(confidential),
        }
    )
    config["valuation"].update(
        {
            "size_marketability_discount": float(size_marketability_discount_pct) / 100,
            "include_operating_leases_in_ev": False,
        }
    )
    return _drop_none_values(config)


def _drop_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _drop_none_values(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none_values(v) for v in value]
    return value


def write_run_config(config: dict[str, Any], run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _validate_required_form_values(target_name: Any, target_description: Any, primary_sic_codes: Any, adjacent_sic_codes: Any) -> str | None:
    missing = []
    if _is_blank(target_name):
        missing.append("Target company")
    if _is_blank(target_description):
        missing.append("Business description")
    if not parse_sic_codes(primary_sic_codes) and not parse_sic_codes(adjacent_sic_codes):
        missing.append("Primary or Adjacent SIC codes")
    if not missing:
        return None
    return "Missing required inputs: " + ", ".join(missing)


def _sic_suggestion_config(description: str) -> dict[str, Any]:
    config = copy.deepcopy(load_base_config())
    config.setdefault(
        "target_company",
        {
            "name": "Target company",
        },
    )
    config["target_company"]["name"] = config["target_company"].get("name") or "Target company"
    config["target_company"]["description"] = description.strip()
    config["target_company"]["primary_sic_codes"] = []
    config["target_company"]["adjacent_sic_codes"] = []
    config.setdefault("universe", {})
    config["universe"].setdefault("max_candidates", 300)
    config.setdefault(
        "llm",
        copy.deepcopy(DEFAULT_LLM_CONFIG),
    )
    config["llm"]["max_tokens"] = max(int(config["llm"].get("max_tokens", 500)), 1200)
    return _drop_none_values(config)


def _suggested_code_text(suggestions: list[dict], bucket: str) -> str:
    codes = [
        str(s.get("sic_code")).strip()
        for s in suggestions
        if str(s.get("bucket") or "").strip().lower() == bucket and s.get("sic_code")
    ]
    return ", ".join(dict.fromkeys(codes))


def suggest_sic_codes_from_description(description: str | None) -> tuple[str, list[list[str]], str, str]:
    if _is_blank(description):
        return "Enter a business description before requesting SIC suggestions.", [], "", ""

    suggestions = llm_analyzer.suggest_sic_codes(_sic_suggestion_config(str(description)))
    if not suggestions:
        return "No SIC code suggestions returned. Check API configuration or refine the business description.", [], "", ""

    rows = [
        [
            str(s.get("bucket") or ""),
            str(s.get("sic_code") or ""),
            str(s.get("title") or ""),
            str(s.get("confidence") or ""),
            str(s.get("reason") or ""),
        ]
        for s in suggestions
    ]
    primary_codes = _suggested_code_text(suggestions, "primary")
    adjacent_codes = _suggested_code_text(suggestions, "adjacent")
    status = (
        "Advisory only: populated Primary and Adjacent SIC codes from LLM-suggested candidates "
        "validated against the SEC SIC list. Review the primary/adjacent classification before running."
    )
    return status, rows, primary_codes, adjacent_codes


def run_from_form(*form_values: Any) -> tuple[str, str | None, str | None, str | None]:
    form_data = dict(zip(RUN_FORM_FIELDS, form_values, strict=True))
    validation_error = _validate_required_form_values(
        form_data["target_name"],
        form_data["target_description"],
        form_data["primary_sic_codes"],
        form_data["adjacent_sic_codes"],
    )
    if validation_error:
        return validation_error, "", None, None

    config = build_config_dict(base_config=load_base_config(), **form_data)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / run_id
    config_path = write_run_config(config, run_dir)

    try:
        output_paths = run_pipeline(str(config_path))
    except Exception as exc:
        # The run directory already holds the config that produced the
        # failure; add the traceback next to it so a config.yaml-only
        # directory is never ambiguous between "crashed" and "in progress",
        # and the failure can be reproduced without re-running from the UI.
        error_log = run_dir / "error.log"
        error_log.write_text(traceback.format_exc(), encoding="utf-8")
        status = f"Run failed: {run_id}\n{type(exc).__name__}: {exc}\nDetails: {error_log}"
        return status, "", None, None

    html_path = _copy_output(output_paths.get("html"), run_dir)
    csv_path = _copy_output(output_paths.get("csv"), run_dir)
    status = f"Run complete: {run_id}\nHTML: {html_path or 'not generated'}\nCSV: {csv_path or 'not generated'}"
    if output_paths.get("small_sample_warning"):
        status += f"\nWARNING: {output_paths['small_sample_warning']}"
    report_actions = _report_actions_text(html_path, csv_path)
    return status, report_actions, html_path, csv_path


def _report_actions_text(html_path: str | None, csv_path: str | None) -> str:
    if not html_path and not csv_path:
        return "No report files were generated for this run."
    lines = ["Run outputs"]
    if html_path:
        lines.append("HTML report ready. Open or download the file below for the full report.")
    if csv_path:
        lines.append("CSV report ready. Download the file below for spreadsheet analysis.")
    return "\n\n".join(lines)


def _copy_output(source_path: str | None, run_dir: Path) -> str | None:
    if not source_path:
        return None
    source = Path(source_path)
    if not source.exists():
        return None
    destination = run_dir / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return str(destination)


def build_app():
    disable_gradio_analytics()
    import gradio as gr

    base_config = load_base_config()
    target_defaults = base_config.get("target_company", {})
    universe_defaults = base_config.get("universe", {})
    output_defaults = base_config.get("output", {})
    valuation_defaults = base_config.get("valuation", {})

    with gr.Blocks(title="PE Comps Pipeline", fill_width=True) as app:
        with gr.Column(elem_id="pe-comps-shell"):
            gr.Markdown("# PE Comps Pipeline")
            gr.Markdown(
                "Enter target-company inputs, run the public-company comparable screening pipeline, "
                "and preview or download the HTML/CSV reports. API keys are read from server environment variables or `.env`."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        target_name = gr.Textbox(label="Target company", value="", placeholder="Enter target company name")
                        prepared_by = gr.Textbox(label="Prepared by", value="", placeholder="Analyst or deal team name")
                        confidential = gr.Checkbox(label="Confidential", value=bool(output_defaults.get("confidential", False)))

                    target_description = gr.Textbox(
                        label="Business description",
                        lines=5,
                        value="",
                        placeholder="Enter the target business description",
                    )
                    with gr.Accordion("Reference example", open=False):
                        gr.Markdown(_reference_text(target_defaults, output_defaults))

                    with gr.Row():
                        revenue_usd_mm = gr.Textbox(label="Revenue ($mm)", value="")
                        ebitda_margin_pct = gr.Textbox(label="EBITDA margin (%)", value="")
                        gross_margin_pct = gr.Textbox(label="Gross margin (%)", value=_ratio_to_pct_text(target_defaults.get("gross_margin_estimate")))

                    with gr.Row():
                        revenue_cagr_3yr_pct = gr.Textbox(
                            label="3-year revenue CAGR (%)",
                            value=_ratio_to_pct_text(target_defaults.get("revenue_cagr_3yr_estimate")),
                        )
                        net_debt_ebitda = gr.Textbox(label="Net debt / EBITDA", value=_optional_float_text(target_defaults.get("net_debt_ebitda_estimate")))
                        capex_revenue_pct = gr.Textbox(label="Capex / revenue (%)", value=_ratio_to_pct_text(target_defaults.get("capex_revenue_estimate")))

                    with gr.Row():
                        primary_sic_codes = gr.Textbox(
                            label="Primary SIC codes",
                            value="",
                            lines=2,
                        )
                        adjacent_sic_codes = gr.Textbox(
                            label="Adjacent SIC codes",
                            value="",
                            lines=2,
                        )
                    with gr.Row():
                        seed_tickers = gr.Textbox(label="Seed tickers", value="", lines=1)
                        must_include_tickers = gr.Textbox(label="Must-include tickers", value="", lines=1)
                        exclude_tickers = gr.Textbox(label="Exclude tickers", value="", lines=1)

                    with gr.Row():
                        suggest_sic_button = gr.Button("Suggest SIC codes", variant="secondary")
                        sic_suggestion_status = gr.Textbox(label="SIC suggestion status", interactive=False)
                    sic_suggestions = gr.Dataframe(
                        headers=SIC_SUGGESTION_HEADERS,
                        datatype=["str", "str", "str", "str", "str"],
                        label="LLM-suggested SIC candidates",
                        interactive=False,
                        wrap=True,
                    )

                    with gr.Accordion("Advanced settings", open=False):
                        with gr.Row():
                            max_candidates = gr.Slider(25, 500, value=universe_defaults.get("max_candidates", 300), step=25, label="Max candidates")
                            primary_allocation_pct = gr.Slider(
                                10,
                                100,
                                value=_ratio_to_pct(universe_defaults.get("primary_allocation_pct", 0.5)),
                                step=5,
                                label="Primary allocation (%)",
                            )
                            top_n_comps = gr.Slider(5, 30, value=output_defaults.get("top_n_comps", 15), step=1, label="Top comps")
                        size_marketability_discount_pct = gr.Slider(
                            0,
                            60,
                            value=_ratio_to_pct(valuation_defaults.get("size_marketability_discount", 0.25)),
                            step=1,
                            label="Private-company discount (%)",
                        )

                with gr.Column(scale=1, elem_id="run-panel"):
                    run_button = gr.Button("Run analysis", variant="primary")
                    status = gr.Textbox(label="Status", lines=6, interactive=False)
                    report_actions = gr.Markdown("Run outputs will appear here after analysis.")
                    html_file = gr.File(label="HTML report")
                    csv_file = gr.File(label="CSV report")

            inputs = [
                target_name,
                target_description,
                revenue_usd_mm,
                ebitda_margin_pct,
                gross_margin_pct,
                revenue_cagr_3yr_pct,
                net_debt_ebitda,
                capex_revenue_pct,
                primary_sic_codes,
                adjacent_sic_codes,
                seed_tickers,
                must_include_tickers,
                exclude_tickers,
                max_candidates,
                primary_allocation_pct,
                top_n_comps,
                size_marketability_discount_pct,
                prepared_by,
                confidential,
            ]
            run_button.click(
                fn=run_from_form,
                inputs=inputs,
                outputs=[status, report_actions, html_file, csv_file],
                show_progress="full",
            )
            suggest_sic_button.click(
                fn=suggest_sic_codes_from_description,
                inputs=target_description,
                outputs=[sic_suggestion_status, sic_suggestions, primary_sic_codes, adjacent_sic_codes],
                show_progress="full",
            )

    app.css = APP_CSS
    return app


def main() -> None:
    disable_gradio_analytics()
    parser = argparse.ArgumentParser(description="PE Comps Pipeline web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind (default: 7860)")
    args = parser.parse_args()

    app = build_app()
    app.queue(default_concurrency_limit=1).launch(server_name=args.host, server_port=args.port, css=APP_CSS)


if __name__ == "__main__":
    main()
