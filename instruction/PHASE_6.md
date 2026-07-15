# Phase 6: Report, Pipeline, and Project Completion

## Project Context

You are building a PE Comparable Company Analysis Pipeline. This tool helps
PE analysts automatically build a comparable company set (comps) for valuing
private companies, producing a structured benchmarking report with valuation
multiples, financial distributions, and an XGBoost-predicted EV/EBITDA range.

This is Phase 6 of 6 — the final phase. Wire everything together.

---

## What Already Exists (Do Not Modify)

- `src/universe_builder.py` — ticker list builder
- `src/fetcher.py` — financial data with caching, retry, logging
- `src/data_quality.py` — data quality report
- `src/llm_analyzer.py` — LLM extraction with checkpoint, judge, retry
- `src/feature_builder.py` — feature matrix construction
- `src/scorer.py` — XGBoost model, SHAP, target prediction
- `eval/ground_truth_builder.py` + `eval/evaluator.py` — evaluation
- `eval/results.md` — actual evaluation numbers from Phase 5
- All 32 previous tests passing
- `data/cache/` — ~300 company JSONs
- `data/checkpoints/llm_checkpoint.json` — LLM results
- `data/models/xgb_model.json` + `data/models/imputation_medians.json`

---

## What To Build in Phase 6

- `src/reporter.py`
- `src/pipeline.py`
- `tests/test_reporter.py` (3 tests)
- `tests/test_pipeline.py` (3 tests)
- `README.md`
- `src/templates/report.html` (Jinja2 template)

---

## src/reporter.py

### Purpose

Take the outputs of scorer.run() and the raw company data, select the
Top 15 comparable companies, and generate a CSV and HTML report.

### How to select Top 15 comps

Apply two filters in sequence:

**Filter 1 — LLM business model alignment (hard filter):**
Remove companies whose `business_model` extracted by LLM does not match
the target company's business model (from `analyze_target` output).
If the target's business_model is None or "other", skip this filter.

Also remove companies flagged as `low_confidence_flag=True` in LLM extraction.

**Filter 2 — residual ranking (soft sort):**
From the remaining companies, sort by `residual_abs` ascending (smallest
residual first — these are the companies whose multiples are best explained
by fundamentals).

Take the top 15 from this sorted list.

**Edge case:** If fewer than 15 companies remain after Filter 1, relax the
filter: allow companies with different business_model but same customer_type
(B2B vs B2C distinction is more important than exact model type).
If still fewer than 10, log a warning and return however many are available.

### Report sections

**Section 1: Target Company Summary**
- Company name and description (from config)
- Predicted EV/EBITDA range: `{range_low:.1f}x — {range_high:.1f}x (center: {predicted:.1f}x)`
- Number of comparable companies analyzed
- Pipeline run timestamp

**Section 2: Valuation Multiple Distribution**
Calculate from Top 15 comps only:

| Metric     | 25th pct | Median | 75th pct | Mean |
|------------|----------|--------|----------|------|
| EV/EBITDA  | X.Xx     | X.Xx   | X.Xx     | X.Xx |
| EV/Revenue | X.Xx     | X.Xx   | X.Xx     | X.Xx |

**Section 3: Financial Benchmarks**
For each metric, show the distribution across Top 15 AND the target's position:

| Metric            | Target Est. | P25   | Median | P75   | Target Percentile |
|-------------------|-------------|-------|--------|-------|-------------------|
| EBITDA Margin     | 18.0%       | 14%   | 19%    | 24%   | 47th              |
| Revenue Growth    | 8.0%        | 5%    | 9%     | 14%   | 44th              |
| Gross Margin      | 42.0%       | 35%   | 43%    | 51%   | 47th              |
| Net Debt/EBITDA   | 2.1x        | 1.2x  | 2.3x   | 3.4x  | 45th              |

Target percentile: where the target's estimate falls in the Top 15 distribution.
Calculate as: `scipy.stats.percentileofscore(top15_values, target_value)`

**Section 4: Top 15 Comparable Companies**

| Rank | Ticker | Company | EV/EBITDA | EBITDA Margin | Revenue ($M) | Rev Growth | Business Model | Sub-sector | Judge Score |
|------|--------|---------|-----------|---------------|--------------|------------|----------------|------------|-------------|

Sort by residual_abs ascending (same order as selection).
Flag companies with judge_score < 3 with a note: "(low confidence)"

**Section 5: Model Diagnostics**
- Cross-validation RMSE: ±X.Xx EV/EBITDA turns
- Training companies: N
- Top 5 features by SHAP importance (feature name + mean SHAP value)
- Evaluation metrics from eval/results.md if file exists:
  "Pipeline validated against SEC proxy peer groups: Mean Precision@15 = XX%"

**Section 6: Data Notes**
- N companies had low LLM confidence and were excluded
- N tickers failed data fetch (see outputs/failed_tickers.csv)
- Standard disclaimer: "Analysis based on public company data via yfinance
  and SEC EDGAR. For reference purposes only."

### CSV output

One row per comparable company. Columns:
`rank, ticker, company_name, ev_ebitda_actual, ev_ebitda_predicted,
residual_abs, ebitda_margin, gross_margin, revenue_ttm_usd_mm,
revenue_cagr_3yr, net_debt_ebitda, business_model, customer_type,
capital_intensity, sub_sector_description, judge_score, low_confidence_flag`

Write to `outputs/comps_report.csv`.

### HTML output

Use Jinja2 to render the template at `src/templates/report.html`.
Write to `outputs/comps_report.html`.

The HTML must:
- Be fully self-contained (no external CSS or JS dependencies, no CDN links)
- Work when opened as a local file in any browser
- Include all 6 sections from above
- Style requirements (all inline CSS):
  - White background, dark gray text (#333)
  - Tables with alternating row colors (#f9f9f9 and white)
  - Median row in valuation multiples table in bold
  - Top-ranked company row in comps table highlighted in light blue (#e8f4f8)
  - Section headers use a dark blue color (#1a3a5c)
  - Low confidence companies shown in italic gray text

### Public interface

```python
def generate(
    scorer_results: dict,
    companies: list[dict],
    llm_features: dict[str, dict],
    config: dict,
) -> dict[str, str]:
    """
    Select Top 15 comps and generate reports.

    Returns:
        {"csv": "outputs/comps_report.csv", "html": "outputs/comps_report.html"}
    """
```

---

## src/templates/report.html

Create a Jinja2 template that renders the 6-section HTML report.
All CSS must be in a `<style>` block in the `<head>`.
All data is passed in as Jinja2 template variables.

Key template variables to expect:
- `target_name`, `target_description`, `target_prediction`
- `valuation_multiples` (dict with P25/median/P75/mean for each metric)
- `financial_benchmarks` (list of benchmark rows)
- `top15` (list of company dicts)
- `model_diagnostics` (dict with CV RMSE, feature importance, etc.)
- `data_notes` (dict with counts of excluded/failed companies)
- `generated_at` (timestamp string)

---

## src/pipeline.py

### Purpose

The single entry point that orchestrates all 6 steps in sequence.
Users run this file directly.

### Logging setup

At the top of the pipeline, before any step runs:
- Create `logs/` directory if it does not exist
- Set up a `RotatingFileHandler` with max 5MB per file, keep 7 backups
- Set up a `StreamHandler` to also print to console
- Format: `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`
- Set root logger level to INFO

### Step sequence

```
STEP 1/6: Building universe
STEP 2/6: Fetching financial data
STEP 3/6: Running LLM analysis
STEP 4/6: Building feature matrix
STEP 5/6: Training model and scoring
STEP 6/6: Generating report
```

Log the step header before each step starts.
Log a completion summary after each step:
- Step 1: `Universe: {N} candidates`
- Step 2: `Fetched: {N_valid}/{N_total} companies with valid data`
- Step 3: `LLM extracted: {N_success} | failed: {N_failed} | low confidence: {N_low}`
- Step 4: `Feature matrix: {N_rows} rows × {N_cols} columns`
- Step 5: `CV RMSE: ±{rmse:.1f}x | Target prediction: {low:.1f}x — {high:.1f}x`
- Step 6: `Report saved: {csv_path}, {html_path}`

### Error handling

If any step fails with an unhandled exception:
- Log the full traceback at ERROR level
- Print a human-readable message explaining which step failed
- Re-raise the exception (do not swallow errors silently)

### Directory setup

At startup, ensure these directories exist (create if missing):
`data/cache/`, `data/checkpoints/`, `data/models/`,
`outputs/`, `logs/`, `eval/`

### Final summary

After all 6 steps, print a final summary block:
```
============================================================
PIPELINE COMPLETE
Target: {target_name}
Comparable companies found: {N}
Predicted EV/EBITDA range: {low:.1f}x — {high:.1f}x
Report: {html_path}
Run time: {elapsed_seconds:.0f}s
============================================================
```

### Command line interface

```python
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="PE Comparable Company Analysis Pipeline"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force XGBoost model retraining even if saved model exists"
    )
    args = parser.parse_args()
    run_pipeline(args.config, args.force_retrain)
```

### Public interface

```python
def run_pipeline(
    config_path: str = "config.yaml",
    force_retrain: bool = False,
) -> dict:
    """
    Run the full pipeline end-to-end.

    Returns the output paths dict from reporter.generate().
    """
```

---

## tests/test_reporter.py

Write these 3 tests. Mock scorer_results and company data.

1. `test_top15_selected_correctly`
   Provide 30 companies where 15 match the target business_model and 15 do not.
   Assert the returned CSV contains exactly 15 rows.

2. `test_csv_file_created`
   Run generate() with sample data.
   Assert `outputs/comps_report.csv` exists.

3. `test_html_file_created`
   Run generate() with sample data.
   Assert `outputs/comps_report.html` exists.
   Assert the file contains the target company name.

---

## tests/test_pipeline.py

Write these 3 tests. Mock all 6 src modules.

1. `test_pipeline_calls_all_six_steps`
   Mock each module's main function (universe_builder.build, fetcher.fetch_batch, etc.)
   Run run_pipeline().
   Assert each of the 6 mocked functions was called exactly once.

2. `test_pipeline_logs_step_headers`
   Mock all modules to return valid dummy data.
   Run run_pipeline().
   Assert `logs/pipeline.log` contains "STEP 1/6" through "STEP 6/6".

3. `test_pipeline_creates_required_directories`
   Delete `outputs/` and `logs/` directories before running.
   Run run_pipeline() (with mocked modules).
   Assert both directories exist after the run.

---

## README.md

Write a complete README with these sections:

### Project Overview
Two paragraphs:
- What the tool does and what problem it solves for PE analysts
- The technical approach (yfinance + EDGAR + OpenAI + XGBoost)

### Quick Start
```bash
git clone <repo-url>
cd pe_comps_pipeline
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"

# Edit config.yaml with your target company details
# Then run:
python -m src.pipeline

# Force model retraining:
python -m src.pipeline --force-retrain

# Run tests:
pytest tests/ -v
```

### Configuration Guide
Explain every field in config.yaml in plain English.

### Understanding the Output
Describe what each of the 6 report sections shows and how to use it.

### Architecture
ASCII diagram showing the 6-step data flow:
```
config.yaml → [1] Universe Builder → [2] Fetcher → [3] LLM Analyzer
                                                           ↓
                                     [6] Reporter ← [5] Scorer ← [4] Feature Builder
```

### Evaluation Results
Copy the key numbers from `eval/results.md`:
- Precision@15 against SEC proxy peer groups
- LLM extraction consistency rates
- Manual review accuracy rates

### Known Limitations
This section is important for credibility. Include:
- Data coverage: public companies only. The data layer can be replaced
  with Capital IQ or PitchBook for private company coverage.
- yfinance data quality: EBITDA and other derived fields have ~30% missing
  rates; outliers are filtered automatically.
- Model scale: XGBoost is trained on ~200 companies per sector. Predictions
  are directional estimates, not precise valuations.
- LLM extraction: business description quality varies. Companies with
  very short or generic descriptions may have unreliable extractions.
- Evaluation: Precision@15 of ~50% means about half of the pipeline's
  selections match the company's own disclosed peer group. The remaining
  selections may still be useful but are not validated by ground truth.

### Cost Estimate
- yfinance: free
- SEC EDGAR: free
- OpenAI API for 300 companies: approximately $3-5 per full pipeline run
- Cached runs (model already trained, LLM checkpoint exists): < $0.10

---

## What NOT to do

- Do NOT change the business logic in any Phase 1-5 module
- Do NOT add external CSS libraries to the HTML report
- Do NOT suppress exceptions in pipeline.py — let them surface clearly
- Do NOT include any API keys in the code
- Do NOT skip writing README.md — it is required

---

## Definition of Done for Phase 6

**Tests:**
- `pytest tests/ -v` passes with 0 failures (32 existing + 6 new = 38 total)

**Full end-to-end run (fresh start):**

1. Delete `data/cache/`, `data/checkpoints/`, `data/models/`, `outputs/`, `logs/`
2. Run `python -m src.pipeline`
3. Verify it completes without errors
4. Check `logs/pipeline.log` contains all 6 step headers with completion summaries

**Cached run test:**

5. Run `python -m src.pipeline` again immediately
6. Verify it completes significantly faster (cache + checkpoint + saved model)
7. Verify outputs are identical to the first run

**Report quality check:**

8. Open `outputs/comps_report.html` in a browser:
   - All 6 sections visible
   - Tables render correctly with alternating colors
   - Top-ranked company is highlighted
   - Predicted EV/EBITDA range is shown with a reasonable value (3x–30x)

9. Open `outputs/comps_report.csv` in Excel or a text editor:
   - 15 rows (one per comp)
   - All expected columns present
   - No encoding errors

**Sanity check on output content:**

10. Look at the top 3 companies in the report.
    Google each company name and verify:
    - They are real companies
    - Their business is plausibly similar to the target (industrial manufacturing)
    - None are obviously wrong (e.g., consumer retail, financial services)

11. Check `README.md`:
    - Installation steps work as written
    - Known limitations section is present and honest
    - Evaluation results section contains actual numbers from eval/results.md

**Final git state:**

12. All files committed to git
13. No API keys in any file
14. `data/cache/`, `data/checkpoints/`, `data/models/`, `outputs/`, `logs/`
    are all in `.gitignore`
