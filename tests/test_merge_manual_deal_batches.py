import json

import pytest

from scripts.merge_manual_deal_batches import merge_batches


def _row(deal_id, precision, reachable=1.0, coverage=None):
    return {
        "deal_id": deal_id,
        "precision": precision,
        "reachable_precision": reachable,
        "coverage": coverage or {"AAA": "hit", "BBB": "not_in_sic_universe"},
    }


def _write_batch(tmp_path, name, discovery_mode, k, rows, n_failed=0):
    path = tmp_path / name
    path.write_text(json.dumps({
        "discovery_mode": discovery_mode, "k": k, "n_deals": len(rows),
        "n_failed_deals": n_failed, "per_deal": rows,
    }))
    return path


def test_merge_combines_disjoint_batches_and_recomputes_aggregates(tmp_path):
    batch1 = _write_batch(tmp_path, "b1.json", "sic+embedding", 15, [
        _row("deal-a", 0.5), _row("deal-b", 0.0),
    ])
    batch2 = _write_batch(tmp_path, "b2.json", "sic+embedding", 15, [
        _row("deal-c", 1.0),
    ])

    merged = merge_batches([batch1, batch2])

    assert merged["n_deals"] == 3
    assert {row["deal_id"] for row in merged["per_deal"]} == {"deal-a", "deal-b", "deal-c"}
    assert merged["mean_precision"] == pytest.approx(0.5)  # (0.5 + 0.0 + 1.0) / 3
    assert merged["coverage_waterfall"] == {"hit": 3, "not_in_sic_universe": 3}


def test_merge_rejects_duplicate_deal_id_across_batches(tmp_path):
    batch1 = _write_batch(tmp_path, "b1.json", "single-sic", 15, [_row("deal-a", 0.5)])
    batch2 = _write_batch(tmp_path, "b2.json", "single-sic", 15, [_row("deal-a", 1.0)])

    with pytest.raises(ValueError, match="deal-a.*more than one batch"):
        merge_batches([batch1, batch2])


def test_merge_rejects_mismatched_discovery_mode(tmp_path):
    batch1 = _write_batch(tmp_path, "b1.json", "single-sic", 15, [_row("deal-a", 0.5)])
    batch2 = _write_batch(tmp_path, "b2.json", "suggest-sic", 15, [_row("deal-b", 1.0)])

    with pytest.raises(ValueError, match="different discovery modes"):
        merge_batches([batch1, batch2])


def test_merge_rejects_mismatched_k(tmp_path):
    batch1 = _write_batch(tmp_path, "b1.json", "single-sic", 15, [_row("deal-a", 0.5)])
    batch2 = _write_batch(tmp_path, "b2.json", "single-sic", 10, [_row("deal-b", 1.0)])

    with pytest.raises(ValueError, match="different K values"):
        merge_batches([batch1, batch2])


def test_merge_excludes_none_precision_from_mean(tmp_path):
    """A deal with an empty ground-truth denominator (e.g. Duckhorn — all
    banker comps foreign-listed) contributes no precision value, matching
    evaluate_manual_deals' own aggregation."""
    batch = _write_batch(tmp_path, "b1.json", "single-sic", 15, [
        _row("deal-a", 0.5), _row("deal-empty-denom", None),
    ])

    merged = merge_batches([batch])

    assert merged["mean_precision"] == pytest.approx(0.5)
    assert merged["n_deals"] == 2


def test_merge_counts_failed_deals(tmp_path):
    batch = _write_batch(tmp_path, "b1.json", "single-sic", 15, [_row("deal-a", 0.5)], n_failed=0)
    batch_row_with_error = tmp_path / "b2.json"
    batch_row_with_error.write_text(json.dumps({
        "discovery_mode": "single-sic", "k": 15, "n_deals": 1, "n_failed_deals": 1,
        "per_deal": [{"deal_id": "deal-b", "precision": None, "error": "boom"}],
    }))

    merged = merge_batches([batch, batch_row_with_error])

    assert merged["n_failed_deals"] == 1
    assert merged["n_deals"] == 2


def test_merge_requires_at_least_one_batch():
    with pytest.raises(ValueError, match="No batch files"):
        merge_batches([])
