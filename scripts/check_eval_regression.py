"""Regression gate for the manual-deals benchmark.

Compares a fresh eval run (scripts.evaluate_manual_deals output JSON) against
the committed baseline for the same discovery mode and exits non-zero on a
regression, so benchmark drops are caught at the moment a change is made
rather than noticed in a later report.

Usage:
    python -m scripts.evaluate_manual_deals            # writes results.json
    python -m scripts.check_eval_regression           # compare vs. baseline
    python -m scripts.check_eval_regression --update-baseline  # bless new numbers

Hard failures (exit 1):
  - any deal failed to evaluate
  - mean Precision@K drops more than --tolerance (absolute)
  - mean reachable (ranking-layer) precision drops more than 5pp
  - total coverage-waterfall hits drop
  - low_confidence_filtered appears/increases in the waterfall
  - a deal that had hits in the baseline loses all of them
  - a baseline deal disappears from the run

Everything else (per-deal precision moves within tolerance, loss-stage
shuffles, new deals) is reported as informational. Market data drifts daily —
comps' multiples and filter outcomes can flip without any code change — so
the default tolerance is deliberately non-zero; tighten it for cached runs.

The baseline records a dev/holdout split: dev deals drove development
decisions, holdout deals never did. The gate prints both means so holdout
degradation is visible even when the overall mean holds up.
"""
import argparse
import datetime
import json
import statistics
import sys
from pathlib import Path

BASELINE_DIR = Path("eval/baselines")
DEFAULT_RESULTS = Path("outputs/eval/manual_deals/results.json")

# Deals added in v2 (the 16-deal benchmark) that never drove development
# iterations. New deals
# default to holdout; promote to dev deliberately once they have been used
# to make a tuning decision.
DEFAULT_HOLDOUT = (
    "perficient-inc-2024",
    "squarespace-inc-2024",
    "smartsheet-inc-2024",
    "stericycle-inc-2024",
    "duckhorn-portfolio-inc-2024",
    "r1-rcm-inc-de-2024",
)


def _baseline_path(discovery_mode: str) -> Path:
    return BASELINE_DIR / f"manual_deals.{discovery_mode}.baseline.json"


def _split_means(per_deal: dict[str, dict], holdout: set[str]) -> tuple[float | None, float | None]:
    dev_vals = [d["precision"] for i, d in per_deal.items() if i not in holdout and d["precision"] is not None]
    hold_vals = [d["precision"] for i, d in per_deal.items() if i in holdout and d["precision"] is not None]
    return (
        statistics.mean(dev_vals) if dev_vals else None,
        statistics.mean(hold_vals) if hold_vals else None,
    )


def build_baseline(results: dict, holdout_ids: tuple[str, ...] = DEFAULT_HOLDOUT) -> dict:
    per_deal = {
        row["deal_id"]: {
            "precision": row["precision"],
            "hits": sorted(row.get("hits") or []),
            "n_eligible": len(row.get("eligible_ground_truth_tickers") or []),
        }
        for row in results["per_deal"]
    }
    deal_ids = set(per_deal)
    return {
        "discovery_mode": results["discovery_mode"],
        "k": results["k"],
        "created": datetime.date.today().isoformat(),
        "n_deals": results["n_deals"],
        "metrics": {
            "mean_precision": results["mean_precision"],
            "median_precision": results["median_precision"],
            "mean_reachable_precision": results["mean_reachable_precision"],
        },
        "coverage_waterfall": results["coverage_waterfall"],
        "splits": {
            "dev": sorted(deal_ids - set(holdout_ids)),
            "holdout": sorted(deal_ids & set(holdout_ids)),
        },
        "per_deal": per_deal,
    }


def compare(results: dict, baseline: dict, tolerance: float) -> tuple[list[str], list[str]]:
    """Returns (hard_failures, notes)."""
    failures: list[str] = []
    notes: list[str] = []

    if results["discovery_mode"] != baseline["discovery_mode"] or results["k"] != baseline["k"]:
        failures.append(
            f"run is {results['discovery_mode']}@K={results['k']} but baseline is "
            f"{baseline['discovery_mode']}@K={baseline['k']} — compare like against like"
        )
        return failures, notes

    if results.get("n_failed_deals"):
        failures.append(f"{results['n_failed_deals']} deal(s) failed to evaluate")

    base_metrics = baseline["metrics"]
    mean_drop = base_metrics["mean_precision"] - results["mean_precision"]
    if mean_drop > tolerance:
        failures.append(
            f"mean Precision@{results['k']} dropped {mean_drop * 100:.1f}pp "
            f"({base_metrics['mean_precision'] * 100:.1f}% -> {results['mean_precision'] * 100:.1f}%)"
        )
    else:
        notes.append(
            f"mean Precision@{results['k']}: {base_metrics['mean_precision'] * 100:.1f}% -> "
            f"{results['mean_precision'] * 100:.1f}%"
        )

    reach_drop = base_metrics["mean_reachable_precision"] - results["mean_reachable_precision"]
    if reach_drop > 0.05:
        failures.append(
            f"reachable (ranking-layer) precision dropped {reach_drop * 100:.1f}pp "
            f"({base_metrics['mean_reachable_precision'] * 100:.1f}% -> "
            f"{results['mean_reachable_precision'] * 100:.1f}%)"
        )

    base_wf = baseline["coverage_waterfall"]
    run_wf = results["coverage_waterfall"]
    if run_wf.get("hit", 0) < base_wf.get("hit", 0):
        failures.append(f"waterfall hits dropped {base_wf.get('hit', 0)} -> {run_wf.get('hit', 0)}")
    base_lc = base_wf.get("low_confidence_filtered", 0)
    run_lc = run_wf.get("low_confidence_filtered", 0)
    if run_lc > base_lc:
        failures.append(
            f"low_confidence_filtered increased {base_lc} -> {run_lc} — the hard-exclusion "
            "gate is eating banker comps again (see the incomplete-profile split)"
        )

    run_by_deal = {row["deal_id"]: row for row in results["per_deal"]}
    for deal_id, base_deal in baseline["per_deal"].items():
        run_deal = run_by_deal.get(deal_id)
        if run_deal is None:
            failures.append(f"{deal_id}: present in baseline but missing from this run")
            continue
        if base_deal["hits"] and not run_deal.get("hits"):
            failures.append(f"{deal_id}: lost all hits (baseline had {', '.join(base_deal['hits'])})")
            continue
        bp, rp = base_deal["precision"], run_deal["precision"]
        if bp is not None and rp is not None and abs(rp - bp) > 1e-9:
            notes.append(f"{deal_id}: P@{results['k']} {bp * 100:.1f}% -> {rp * 100:.1f}%")
    for deal_id in run_by_deal.keys() - baseline["per_deal"].keys():
        notes.append(f"{deal_id}: new deal, not in baseline (run --update-baseline to include)")

    holdout = set(baseline["splits"]["holdout"])
    per_deal_now = {
        row["deal_id"]: {"precision": row["precision"]}
        for row in results["per_deal"]
    }
    dev_mean, hold_mean = _split_means(per_deal_now, holdout)
    if dev_mean is not None and hold_mean is not None:
        notes.append(f"dev mean {dev_mean * 100:.1f}% | holdout mean {hold_mean * 100:.1f}%")

    return failures, notes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a manual-deals eval run against the committed baseline.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="Eval results JSON from scripts.evaluate_manual_deals.")
    parser.add_argument("--baseline", default=None, help="Baseline JSON path (default: eval/baselines/manual_deals.<mode>.baseline.json).")
    parser.add_argument(
        "--tolerance", type=float, default=0.01,
        help="Allowed absolute drop in mean precision before failing (default 0.01 = 1pp; market data drifts daily).",
    )
    parser.add_argument("--update-baseline", action="store_true", help="Overwrite the baseline with this run's numbers.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    baseline_path = Path(args.baseline) if args.baseline else _baseline_path(results["discovery_mode"])

    if args.update_baseline:
        existing_holdout = DEFAULT_HOLDOUT
        if baseline_path.exists():
            existing_holdout = tuple(json.loads(baseline_path.read_text(encoding="utf-8"))["splits"]["holdout"])
        baseline = build_baseline(results, holdout_ids=existing_holdout)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(baseline, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Baseline updated: {baseline_path} ({baseline['n_deals']} deals, "
              f"mean P@{baseline['k']} {baseline['metrics']['mean_precision'] * 100:.1f}%)")
        return 0

    if not baseline_path.exists():
        print(f"No baseline at {baseline_path} — run with --update-baseline to create one.")
        return 1

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    failures, notes = compare(results, baseline, args.tolerance)

    for note in notes:
        print(f"  note: {note}")
    if failures:
        print(f"\nREGRESSION vs {baseline_path} (created {baseline.get('created')}):")
        for failure in failures:
            print(f"  FAIL: {failure}")
        return 1
    print(f"\nOK — no regression vs {baseline_path} (created {baseline.get('created')}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
