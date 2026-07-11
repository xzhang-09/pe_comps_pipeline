"""
Penalty-parameter tuning — close the eval loop.

The soft-penalty magnitudes in config.yaml's scorer.ranking_penalties are
hand-picked defaults. This searches that space for the values that maximize
precision@k against the fairness-opinion ground truth, turning "we can measure
selection quality" into "we use the measurement to improve it."

The objective is eval.evaluator._evaluate_precision_at_k, which already maps a
penalties dict (plus fixed comp data) to a mean precision. The tuner treats that
as a black-box scalar objective and is otherwise pure search, so its core is
unit-testable with a stub objective and reusable for any scalar metric. This is
practical precisely because of the enrichment/scoring split (src/pipeline.py):
the universe is enriched and scored once, and only the penalties vary per
candidate — no re-fetching, no re-extraction.

Note: _evaluate_precision_at_k recomputes sub-sector embeddings on every call, so
a large grid over data WITH sub_sector descriptions issues many identical
embedding requests. Precomputing those similarities once and injecting them is
the natural next optimization; until then, keep the grid modest for live runs.
"""
import itertools
from collections.abc import Callable
from dataclasses import dataclass

from eval.evaluator import _evaluate_precision_at_k
from src import get_logger

logger = get_logger(__name__)

# Sweeps the four penalty magnitudes (the most consequential params); the
# sub-sector similarity threshold and the size-penalty free range are held at
# their config defaults. 3^4 = 81 candidates.
DEFAULT_GRID: dict[str, list[float]] = {
    "business_model_penalty": [0.3, 0.6, 0.9],
    "customer_type_penalty": [0.25, 0.5, 0.75],
    # Max penalty at similarity 0; the effective penalty for typical drift
    # (similarity 0.35-0.45 vs a ~0.5 threshold) is roughly a quarter of
    # this — see report_selection._subsector_mismatch_penalty's graded ramp.
    "subsector_mismatch_penalty": [0.5, 1.0, 2.0],
    "size_penalty_per_extra_log10": [0.5, 1.0, 1.5],
}


@dataclass(frozen=True)
class TuneResult:
    best_penalties: dict
    best_score: float
    ranked: list[tuple[dict, float]]  # every candidate, best score first


def grid_candidates(base_penalties: dict, grid: dict[str, list[float]]) -> list[dict]:
    """Cartesian product over the swept params, each combination merged onto
    base_penalties so params not in the grid keep their base values."""
    keys = list(grid)
    return [
        {**base_penalties, **dict(zip(keys, combo))}
        for combo in itertools.product(*(grid[k] for k in keys))
    ]


def tune_penalties(objective: Callable[[dict], float], candidates: list[dict]) -> TuneResult:
    """Evaluate objective(penalties) for each candidate and keep the highest
    (higher = better). Pure search — no knowledge of where the score comes from,
    so it works with the real precision objective or a stub."""
    if not candidates:
        raise ValueError("No candidate penalty sets to tune over")
    scored = [(candidate, objective(candidate)) for candidate in candidates]
    scored.sort(key=lambda cs: cs[1], reverse=True)
    logger.info(f"Tuned {len(scored)} candidates; best score {scored[0][1]:.4f}")
    return TuneResult(best_penalties=scored[0][0], best_score=scored[0][1], ranked=scored)


def precision_objective(
    ground_truth: dict[str, list[str]],
    company_scores,
    llm_features: dict,
    companies_by_ticker: dict,
    embedding_model: str,
) -> Callable[[dict], float]:
    """Wrap eval's precision@k as a `penalties -> mean precision` objective with
    the comp data held fixed — exactly the inputs run_evaluation already
    assembles. Tuning then varies only the penalties."""
    def objective(penalties: dict) -> float:
        return _evaluate_precision_at_k(
            ground_truth, company_scores, llm_features, companies_by_ticker, penalties, embedding_model,
        )["mean"]
    return objective


def generate_tuning_report(result: TuneResult, baseline_score: float | None = None) -> str:
    """Human-readable summary of a tuning run."""
    lines = [
        "# Penalty Tuning Results",
        "",
        f"Best mean precision@k: {result.best_score * 100:.1f}% (over {len(result.ranked)} candidates)",
    ]
    if baseline_score is not None:
        lines.append(f"Starting penalties:    {baseline_score * 100:.1f}%")
        lines.append(f"Improvement:           {(result.best_score - baseline_score) * 100:+.1f} pts")
    lines += ["", "Best penalties:"]
    lines += [f"- {key}: {value}" for key, value in sorted(result.best_penalties.items())]
    return "\n".join(lines) + "\n"
