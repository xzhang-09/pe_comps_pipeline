from pathlib import Path

import yaml

from src.config_schema import PipelineConfig


def test_sic_clusters_only_document_configured_sic_codes():
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    target = config["target_company"]
    universe = config["universe"]

    configured_sics = set(target["primary_sic_codes"]) | set(target.get("adjacent_sic_codes", []))
    clustered_sics = set(universe.get("sic_clusters", {}))

    assert clustered_sics <= configured_sics


def test_target_company_config_does_not_include_unused_gics_fields():
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

    assert "gics_sector" not in config["target_company"]
    assert "gics_industry" not in config["target_company"]


def test_demo_config_uses_business_fit_first_end_market_penalty():
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    penalties = config["scorer"]["ranking_penalties"]

    assert penalties["subsector_similarity_threshold"] <= 0.48
    assert penalties["subsector_mismatch_penalty"] >= 2.0


def test_suggest_sic_embedding_mode_validates_without_changing_default():
    raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    assert PipelineConfig.model_validate(raw).universe.discovery_mode == "sic"

    raw["universe"]["discovery_mode"] = "suggest-sic+embedding"
    assert PipelineConfig.model_validate(raw).universe.discovery_mode == "suggest-sic+embedding"
