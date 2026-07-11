import pytest

from src.config_schema import PipelineConfig


@pytest.fixture
def make_config():
    """Factory for a *valid* PipelineConfig in tests. Pass nested overrides as
    dicts (shallow-merged into the defaults), e.g.
    make_config(universe={"max_candidates": 5}). Validating through the real
    schema means tests exercise the production contract instead of an ad-hoc
    partial dict that could never pass pipeline._load_config — the migration to
    typed config (config_schema.as_config) relies on every consumer receiving a
    config that actually validates."""
    def _make(**overrides) -> PipelineConfig:
        base = {
            "target_company": {
                "name": "Test Co.",
                "description": "Test company that manufactures industrial parts for OEMs.",
                "primary_sic_codes": ["1111"],
                "adjacent_sic_codes": ["2222"],
            },
            "universe": {"max_candidates": 10, "primary_allocation_pct": 0.8},
            "llm": {
                "extraction_model": "gpt-4.1",
                "judge_model": "gpt-4.1-mini",
                "temperature": 0,
                "max_tokens": 500,
                "batch_size": 20,
                "judge_threshold": 3,
            },
        }
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = {**base[key], **value}
            else:
                base[key] = value
        return PipelineConfig.model_validate(base)

    return _make


@pytest.fixture
def sample_company():
    return {
        "ticker": "TEST",
        "company_name": "Test Industrial Co.",
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
        "business_description": "Test Industrial Co. manufactures industrial parts for automotive OEMs.",
        "description_source": "edgar",
        "fetch_timestamp": "2026-06-16T00:00:00+00:00",
    }


@pytest.fixture
def sample_config():
    return {
        "target_company": {
            "name": "Example Manufacturing Co.",
            "description": "Industrial parts manufacturer serving automotive OEMs.",
            "revenue_usd_mm": 150,
            "ebitda_margin_estimate": 0.18,
            "geography": "north_america",
            "primary_sic_codes": ["3714", "3559", "3569", "3490"],
            "adjacent_sic_codes": ["3612", "3613", "3621", "3640"],
        },
        "universe": {
            "max_candidates": 10,
            "primary_allocation_pct": 0.8,
            "min_revenue_usd_mm": 30,
            "max_revenue_usd_mm": 800,
            "min_ebitda_margin": 0.05,
        },
        "llm": {
            "extraction_model": "gpt-4.1",
            "judge_model": "gpt-4.1-mini",
            "temperature": 0,
            "max_tokens": 500,
            "batch_size": 20,
            "judge_threshold": 3,
        },
        "output": {
            "top_n_comps": 15,
            "report_formats": ["csv", "html"],
        },
    }


@pytest.fixture(autouse=True)
def isolated_data_dirs(tmp_path, monkeypatch):
    """Redirect fetcher's cache/output paths into tmp_path so tests never
    touch the real data/cache or outputs directories."""
    import eval.evaluator as evaluator
    import eval.ground_truth_builder as ground_truth_builder
    import scripts.data_quality as data_quality
    import src.fetcher as fetcher
    import src.llm_analyzer as llm_analyzer
    import src.reporter as reporter
    import src.sic_universe_builder as sic_universe_builder

    cache_dir = tmp_path / "cache"
    outputs_dir = tmp_path / "outputs"
    checkpoints_dir = tmp_path / "checkpoints"
    eval_dir = tmp_path / "eval"
    monkeypatch.setattr(fetcher, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(fetcher, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(fetcher, "FAILED_TICKERS_CSV", outputs_dir / "failed_tickers.csv")
    monkeypatch.setattr(data_quality, "OUTPUT_PATH", outputs_dir / "data_quality_report.txt")
    monkeypatch.setattr(llm_analyzer, "CHECKPOINT_PATH", checkpoints_dir / "llm_checkpoint.json")
    monkeypatch.setattr(ground_truth_builder, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(evaluator, "RESULTS_PATH", eval_dir / "results.md")
    monkeypatch.setattr(reporter, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(reporter, "CSV_PATH", outputs_dir / "comps_report.csv")
    monkeypatch.setattr(reporter, "HTML_PATH", outputs_dir / "comps_report.html")
    monkeypatch.setattr(reporter, "FAILED_TICKERS_PATH", outputs_dir / "failed_tickers.csv")
    monkeypatch.setattr(sic_universe_builder, "CACHE_DIR", cache_dir)
