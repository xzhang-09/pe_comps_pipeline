from pathlib import Path

import yaml


def test_sic_clusters_only_document_configured_sic_codes():
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    target = config["target_company"]
    universe = config["universe"]

    configured_sics = set(target["primary_sic_codes"]) | set(target.get("adjacent_sic_codes", []))
    clustered_sics = set(universe.get("sic_clusters", {}))

    assert clustered_sics <= configured_sics
