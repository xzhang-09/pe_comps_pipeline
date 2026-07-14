import json
from pathlib import Path

from scripts import check_eval_regression as gate


def _results(**overrides) -> dict:
    base = {
        "discovery_mode": "single-sic",
        "k": 15,
        "n_deals": 2,
        "n_failed_deals": 0,
        "mean_precision": 0.10,
        "median_precision": 0.10,
        "mean_reachable_precision": 1.0,
        "coverage_waterfall": {"hit": 3, "not_in_sic_universe": 5},
        "per_deal": [
            {"deal_id": "dev-deal", "precision": 0.10, "hits": ["AAA"],
             "eligible_ground_truth_tickers": ["AAA", "BBB"]},
            {"deal_id": "holdout-deal", "precision": 0.10, "hits": ["CCC"],
             "eligible_ground_truth_tickers": ["CCC", "DDD"]},
        ],
    }
    base.update(overrides)
    return base


def _baseline(**overrides) -> dict:
    baseline = gate.build_baseline(_results(), holdout_ids=("holdout-deal",))
    baseline.update(overrides)
    return baseline


def test_identical_run_passes():
    failures, notes = gate.compare(_results(), _baseline(), tolerance=0.01)
    assert failures == []
    assert any("dev mean" in n and "holdout mean" in n for n in notes)


def test_mean_precision_drop_beyond_tolerance_fails():
    results = _results(mean_precision=0.05)
    failures, _ = gate.compare(results, _baseline(), tolerance=0.01)
    assert any("mean Precision@15 dropped" in f for f in failures)


def test_mean_precision_drop_within_tolerance_passes():
    results = _results(mean_precision=0.095)
    failures, _ = gate.compare(results, _baseline(), tolerance=0.01)
    assert failures == []


def test_low_confidence_filtered_increase_fails():
    results = _results(coverage_waterfall={"hit": 3, "low_confidence_filtered": 2})
    failures, _ = gate.compare(results, _baseline(), tolerance=0.01)
    assert any("low_confidence_filtered increased" in f for f in failures)


def test_waterfall_hit_drop_fails():
    results = _results(coverage_waterfall={"hit": 1, "not_in_sic_universe": 7})
    failures, _ = gate.compare(results, _baseline(), tolerance=0.01)
    assert any("waterfall hits dropped" in f for f in failures)


def test_deal_losing_all_hits_fails():
    results = _results()
    results["per_deal"][0] = {
        "deal_id": "dev-deal", "precision": 0.0, "hits": [],
        "eligible_ground_truth_tickers": ["AAA", "BBB"],
    }
    failures, _ = gate.compare(results, _baseline(), tolerance=0.01)
    assert any("dev-deal: lost all hits" in f for f in failures)


def test_missing_baseline_deal_fails():
    results = _results()
    results["per_deal"] = results["per_deal"][:1]
    failures, _ = gate.compare(results, _baseline(), tolerance=0.01)
    assert any("holdout-deal: present in baseline but missing" in f for f in failures)


def test_failed_deals_fail():
    results = _results(n_failed_deals=1)
    failures, _ = gate.compare(results, _baseline(), tolerance=0.01)
    assert any("failed to evaluate" in f for f in failures)


def test_mode_mismatch_fails_without_metric_comparison():
    results = _results(discovery_mode="suggest-sic")
    failures, notes = gate.compare(results, _baseline(), tolerance=0.01)
    assert len(failures) == 1
    assert "compare like against like" in failures[0]


def test_published_baseline_is_in_sync_with_gate_schema():
    """The committed baseline must load and self-compare cleanly so a schema
    change in the gate cannot silently orphan it."""
    path = Path("eval/baselines/manual_deals.single-sic.baseline.json")
    baseline = json.loads(path.read_text(encoding="utf-8"))
    for key in ("discovery_mode", "k", "metrics", "coverage_waterfall", "splits", "per_deal"):
        assert key in baseline, f"baseline missing {key}"
    assert set(baseline["splits"]["holdout"]) == set(gate.DEFAULT_HOLDOUT)
    assert baseline["n_deals"] == len(baseline["per_deal"])
