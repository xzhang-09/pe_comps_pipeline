# Evaluation Status

This project does not publish an audited ground-truth Precision@K benchmark.

The evaluation framework is kept in this directory so a validated benchmark can
be run later without rebuilding the harness from scratch:

- `eval/evaluator.py` evaluates Precision@K once a reliable ground-truth peer
  set is available. Its Top-K selection logic is kept aligned with the report
  selection logic.
- `eval/ground_truth_builder.py` is an experimental builder for extracting
  valuation comps from merger proxy / fairness opinion "Selected Companies
  Analysis" sections. This is the intended ground-truth source, but it still
  needs live SEC-response validation before results should be published.
- Manual review samples are not published in this repository. If needed, LLM
  extraction quality can be reviewed from freshly generated run artifacts.

Current report quality is assessed directionally through the run diagnostics and
the LLM-assisted comp-fit review in `outputs/comps_report.html`. Those signals
are useful for reviewing a run, but they are not audited ground truth.

## Future Work

- Validate SEC full-text search and filing parsing against live merger filings.
- Build a target-specific ground-truth set from selected-companies analyses.
- Rerun Precision@K using the current SIC-based universe and ranking logic.
- Publish benchmark numbers only after the ground-truth source has been
  validated and documented.
