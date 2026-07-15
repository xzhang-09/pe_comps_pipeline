import argparse
import time
from pathlib import Path

import yaml

from src import feature_builder, fetcher, get_logger, llm_analyzer, reporter, scorer, universe_builder

logger = get_logger(__name__)

REQUIRED_DIRECTORIES = ("data/cache", "data/checkpoints", "data/models", "outputs", "logs", "eval")


def _ensure_directories() -> None:
    for directory in REQUIRED_DIRECTORIES:
        Path(directory).mkdir(parents=True, exist_ok=True)


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(
    config_path: str = "config.yaml",
    force_retrain: bool = False,
) -> dict:
    """
    Run the full pipeline end-to-end.

    Returns the output paths dict from reporter.generate().
    """
    start_time = time.time()
    _ensure_directories()
    config = _load_config(config_path)

    current_step = None
    try:
        current_step = "STEP 1/6: Building universe"
        logger.info(current_step)
        tickers = universe_builder.build(config)
        logger.info(f"Universe: {len(tickers)} candidates")

        current_step = "STEP 2/6: Fetching financial data"
        logger.info(current_step)
        companies = fetcher.fetch_batch(tickers, config)
        n_valid = sum(1 for c in companies if c.get("ev_ebitda") is not None)
        logger.info(f"Fetched: {n_valid}/{len(companies)} companies with valid data")

        current_step = "STEP 3/6: Running LLM analysis"
        logger.info(current_step)
        llm_features = llm_analyzer.analyze_batch(companies, config)
        target_llm_features = llm_analyzer.analyze_target(config)
        n_success = sum(1 for v in llm_features.values() if not v.get("extraction_failed"))
        n_failed = sum(1 for v in llm_features.values() if v.get("extraction_failed"))
        n_low = sum(1 for v in llm_features.values() if v.get("low_confidence_flag"))
        logger.info(f"LLM extracted: {n_success} | failed: {n_failed} | low confidence: {n_low}")

        current_step = "STEP 4/6: Building feature matrix"
        logger.info(current_step)
        feature_matrix, ev_ebitda_raw, imputation_medians = feature_builder.build(companies, llm_features)
        logger.info(f"Feature matrix: {feature_matrix.shape[0]} rows x {feature_matrix.shape[1]} columns")

        current_step = "STEP 5/6: Training model and scoring"
        logger.info(current_step)
        scorer_results = scorer.run(
            feature_matrix, config["target_company"], target_llm_features, imputation_medians,
            force_retrain=force_retrain,
        )
        target_prediction = scorer_results["target_prediction"]
        cv_mae = scorer_results.get("cv_mae_median", scorer_results.get("cv_rmse_multiple_space"))
        logger.info(
            f"CV RMSE: ±{cv_mae:.1f}x | "
            f"Target prediction: {target_prediction['range_low']:.1f}x — {target_prediction['range_high']:.1f}x"
        )

        current_step = "STEP 6/6: Generating report"
        logger.info(current_step)
        output_paths = reporter.generate(
            scorer_results, companies, llm_features, target_llm_features, imputation_medians, config,
        )
        logger.info(f"Report saved: {output_paths['csv']}, {output_paths['html']}")

    except Exception:
        logger.error(f"Pipeline failed during: {current_step}", exc_info=True)
        print(f"Pipeline failed during: {current_step} — see logs/pipeline.log for details.")
        raise

    with open(output_paths["csv"], "r", encoding="utf-8") as f:
        n_comps = max(sum(1 for _ in f) - 1, 0)  # minus header row

    elapsed = time.time() - start_time
    target_name = config["target_company"]["name"]

    summary = (
        "=" * 60 + "\n"
        "PIPELINE COMPLETE\n"
        f"Target: {target_name}\n"
        f"Comparable companies found: {n_comps}\n"
        f"Predicted EV/EBITDA range: {target_prediction['range_low']:.1f}x — {target_prediction['range_high']:.1f}x\n"
        f"Report: {output_paths['html']}\n"
        f"Run time: {elapsed:.0f}s\n"
        + "=" * 60
    )
    print(summary)
    logger.info(summary)

    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PE Comparable Company Analysis Pipeline")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force XGBoost model retraining even if saved model exists",
    )
    args = parser.parse_args()
    run_pipeline(args.config, args.force_retrain)
