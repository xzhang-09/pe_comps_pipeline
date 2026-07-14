"""Merge partial manual-deals eval results from batched --deals runs.

evaluate_manual_deals's output write is an unconditional overwrite of the
whole file, with no per-deal incremental save — a batch split across FMP
quota windows via --deals (each batch its own --output-json path) needs an
explicit merge step afterward, or later batches silently destroy earlier
ones if pointed at the same path. This combines N same-discovery-mode,
disjoint-deal-id batch files into one results.json + results.md, using the
exact aggregation eval.coverage.waterfall_counts already uses — so a merged
16-deal file is indistinguishable from one produced by a single full run.

Usage:
    python -m scripts.evaluate_manual_deals --discovery sic+embedding \\
        --deals squarespace-inc-2024,smartsheet-inc-2024 \\
        --output-json outputs/eval/manual_deals/batch1.json
    python -m scripts.evaluate_manual_deals --discovery sic+embedding \\
        --deals stericycle-inc-2024,duckhorn-portfolio-inc-2024 \\
        --output-json outputs/eval/manual_deals/batch2.json
    python -m scripts.merge_manual_deal_batches \\
        outputs/eval/manual_deals/batch1.json outputs/eval/manual_deals/batch2.json \\
        --output-json outputs/eval/manual_deals/results.sic+embedding.json \\
        --results-md eval/results.sic+embedding.md
"""
import argparse
import json
import statistics
from pathlib import Path

from eval import coverage, evaluator
from scripts.evaluate_manual_deals import _write_results_json, _write_results_md
from src import get_logger

logger = get_logger(__name__)


def merge_batches(batch_paths: list[Path]) -> dict:
    batches = [json.loads(p.read_text(encoding="utf-8")) for p in batch_paths]
    if not batches:
        raise ValueError("No batch files given")

    modes = {b["discovery_mode"] for b in batches}
    if len(modes) > 1:
        raise ValueError(f"Batches use different discovery modes, cannot merge: {sorted(modes)}")
    ks = {b["k"] for b in batches}
    if len(ks) > 1:
        raise ValueError(f"Batches use different K values, cannot merge: {sorted(ks)}")

    per_deal = []
    seen_ids: set[str] = set()
    for batch, path in zip(batches, batch_paths):
        for row in batch["per_deal"]:
            if row["deal_id"] in seen_ids:
                raise ValueError(
                    f"{row['deal_id']} appears in more than one batch (last seen in {path}) — "
                    "batches must cover disjoint deal_ids, re-running the same deal twice silently "
                    "picks whichever batch file happens to be listed last."
                )
            seen_ids.add(row["deal_id"])
            per_deal.append(row)

    precision_values = [row["precision"] for row in per_deal if row["precision"] is not None]
    reachable_values = [
        row["reachable_precision"] for row in per_deal if row.get("reachable_precision") is not None
    ]
    return {
        "discovery_mode": modes.pop(),
        "mean_precision": sum(precision_values) / len(precision_values) if precision_values else 0.0,
        "median_precision": statistics.median(precision_values) if precision_values else 0.0,
        "mean_reachable_precision": (
            sum(reachable_values) / len(reachable_values) if reachable_values else None
        ),
        "n_deals": len(per_deal),
        "n_failed_deals": sum(1 for row in per_deal if row.get("error")),
        "k": ks.pop(),
        "coverage_waterfall": coverage.waterfall_counts(per_deal),
        "per_deal": per_deal,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", nargs="+", help="Partial results.json files to merge (same discovery_mode, disjoint deal_ids).")
    parser.add_argument("--output-json", required=True, help="Combined results JSON path.")
    parser.add_argument("--results-md", default=str(evaluator.RESULTS_PATH), help="Combined results Markdown path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    merged = merge_batches([Path(p) for p in args.batches])
    _write_results_json(merged, args.output_json)
    _write_results_md(merged, args.results_md)
    complete = merged["n_deals"] - merged["n_failed_deals"]
    logger.info(
        f"Merged {len(args.batches)} batches -> {merged['n_deals']} deals "
        f"({complete} succeeded), mean Precision@{merged['k']} {merged['mean_precision'] * 100:.1f}%"
    )
    print(f"Wrote merged results to {args.output_json} and {args.results_md}")


if __name__ == "__main__":
    main()
