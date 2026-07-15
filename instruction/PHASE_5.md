# Phase 5: Evaluation Module

## Project Context

You are building a PE Comparable Company Analysis Pipeline. This tool helps
PE analysts build a comparable company set (comps) for valuing private companies.
The pipeline is now functionally complete: data fetching, LLM extraction, and
XGBoost modeling all work. Phase 5 adds an evaluation layer to measure whether
the pipeline actually selects good comparable companies.

This is Phase 5 of 6.

---

## What Already Exists (Do Not Modify)

- `src/universe_builder.py`, `src/fetcher.py`, `src/data_quality.py`
- `src/llm_analyzer.py` — LLM extraction with checkpoint and judge
- `src/feature_builder.py` — merges financial + LLM features into matrix
- `src/scorer.py` — XGBoost model, SHAP importance, target prediction
- `data/cache/` — ~300 company JSON files
- `data/checkpoints/llm_checkpoint.json` — LLM results for all companies
- `data/models/xgb_model.json` — trained model
- All 27 previous tests passing

---

## What To Build in Phase 5

One new directory and three new files:

```
eval/
├── __init__.py
├── ground_truth_builder.py   # Fetch peer groups from SEC proxy statements
├── evaluator.py              # Calculate Precision@K and consistency metrics
└── results.md                # Written after running evaluation (not code)
```

One new test file:
```
tests/test_evaluator.py
```

---

## Background: What is a Compensation Peer Group?

US public companies are required to disclose in their annual proxy statement
(SEC form DEF 14A) a list of peer companies they use for executive compensation
benchmarking. This list is called the "Compensation Peer Group."

It is typically 12–20 companies that the board considers to be the most
comparable businesses in terms of size, industry, and business model.

This is the closest thing to a publicly available "official comparable company
list" — reviewed and approved by the board of directors, disclosed to the SEC.

We use this as ground truth to evaluate whether our pipeline's Top 15
overlaps with what the company itself considers comparable.

---

## eval/ground_truth_builder.py

### Purpose

For a list of "test companies" (public companies whose proxy statements we will
read), automatically extract their Compensation Peer Group lists from SEC EDGAR.

### What to build

**Step 1: Find the DEF 14A filing for a given ticker**

Use the SEC EDGAR API:
```
https://data.sec.gov/submissions/CIK{cik_number}.json
```

To find a company's CIK number, use:
```
https://efts.sec.gov/LATEST/search-index?q="{ticker}"&forms=DEF+14A
```

Set User-Agent header: `"PE-Comps-Pipeline research@example.com"`

**Step 2: Download the DEF 14A document text**

From the EDGAR filing index, get the URL of the actual DEF 14A document.
Download the text content (these are HTML files, often large).
Only take the first 200,000 characters to keep it manageable.

**Step 3: Use LLM to extract the peer group**

Send the document text to `gpt-4.1-mini` with this prompt:

```
You are extracting a compensation peer group from a proxy statement (DEF 14A).

Find the section called "Compensation Peer Group", "Peer Group",
"Benchmarking Peer Group", or similar. Extract the list of company names
in that peer group.

Document text (may be truncated):
{document_text[:150000]}

Return ONLY a JSON array of company name strings.
Example: ["Company A", "Company B", "Company C"]
If you cannot find a peer group list, return an empty array: []
Do not include any other text.
```

**Step 4: Map company names to tickers**

The peer group is returned as company names, not tickers.
Map names to tickers using this approach:
- Search the SEC EDGAR company search endpoint:
  `https://efts.sec.gov/LATEST/search-index?q="{company_name}"&forms=10-K`
- Take the first result's ticker if the name match looks reasonable
- If no match found, skip that company (it may be private or delisted)

Log which companies could not be mapped.

### Hardcoded test company list

Use these 15 companies as test cases (they are large industrials/healthcare/
tech hardware companies with well-documented proxy statements):

```python
TEST_TICKERS = [
    "MMM",   # 3M — diversified industrials
    "HON",   # Honeywell
    "EMR",   # Emerson Electric
    "ITW",   # Illinois Tool Works
    "PH",    # Parker Hannifin
    "AME",   # AMETEK
    "IEX",   # IDEX Corporation
    "RRX",   # Rexnord
    "XYL",   # Xylem
    "GNRC",  # Generac Holdings
    "MDT",   # Medtronic — healthcare
    "SYK",   # Stryker
    "ZBH",   # Zimmer Biomet
    "EW",    # Edwards Lifesciences
    "NTAP",  # NetApp — tech hardware
]
```

### Caching

Cache each company's extracted peer group to avoid re-fetching:
- File: `data/cache/peer_group_{ticker}.json`
- Format: `{"ticker": "MMM", "peer_group_tickers": ["HON", "EMR", ...], "extracted_at": "..."}`

### Public interface

```python
def build_ground_truth(
    test_tickers: list[str],
    config: dict,
) -> dict[str, list[str]]:
    """
    For each test ticker, extract its compensation peer group from SEC EDGAR.

    Returns:
        dict mapping ticker -> list of peer tickers
        Example: {"MMM": ["HON", "EMR", "SWK", ...], "HON": [...], ...}

    Companies whose peer group could not be extracted are omitted from results.
    Caches results to data/cache/peer_group_{ticker}.json.
    """
```

---

## eval/evaluator.py

### Purpose

Given the ground truth peer groups, evaluate how well the pipeline's
Top N selection overlaps with the official peer groups. Also evaluate
LLM extraction consistency.

### Evaluation 1: Precision@K against peer groups

For each test company in the ground truth:

1. Run the pipeline's comp selection on that company:
   - Use that company's own data as the "target" (treat it as if it were private)
   - Get the pipeline's Top K companies (use K=15)
   - Exclude the target company itself from both ground truth and pipeline output

2. Calculate overlap:
   - `hits = len(set(pipeline_top_k) ∩ set(ground_truth_peers))`
   - `precision_at_k = hits / len(ground_truth_peers)`
   - (We divide by ground truth size, not K, because ground truth size varies)

3. Aggregate across all test companies:
   - Mean Precision@K
   - Median Precision@K
   - Minimum and maximum

Log each test company's result:
```
INFO: MMM — Ground truth: 14 peers, Pipeline hits: 7/14, Precision: 50.0%
```

### Evaluation 2: LLM extraction consistency

For a random sample of 30 companies from the dataset:
- Run the LLM extraction twice (two separate API calls, same prompt)
- Compare the two outputs field by field
- Calculate agreement rate per field:
  `agreement_rate = number of companies where both runs returned same value / 30`

This reveals which fields are reliably extracted and which are unstable.

Use `gpt-4.1-mini` for consistency testing (cheaper than gpt-4.1).

### Evaluation 3: Manual review support

Generate a formatted text file that makes manual review easy:
- File: `eval/manual_review_sample.txt`
- Pick 15 random companies from the dataset
- For each company, show:
  ```
  === TICKER: MMM — 3M Company ===
  Business Description (first 300 chars):
  {description}

  LLM Extraction:
    business_model: manufacturing
    revenue_recurrence: medium
    customer_type: B2B
    capital_intensity: asset_heavy
    primary_value_driver: technology
    sub_sector_description: diversified manufacturer of industrial and consumer products
    confidence: 4
    judge_score: 4

  Your assessment (fill in manually):
    business_model correct? [Y/N]: ___
    revenue_recurrence correct? [Y/N]: ___
    customer_type correct? [Y/N]: ___
  ```
- Leave blank lines for manual annotation

This file is for the developer to fill in by hand. Do not try to auto-fill it.

### Public interface

```python
def run_evaluation(
    ground_truth: dict[str, list[str]],
    companies: list[dict],
    llm_features: dict[str, dict],
    scorer_results: dict,
    config: dict,
) -> dict:
    """
    Run all three evaluations.

    Returns:
    {
        "precision_at_k": {
            "mean": float,
            "median": float,
            "min": float,
            "max": float,
            "per_company": dict[str, float],
        },
        "llm_consistency": {
            "business_model_agreement": float,
            "revenue_recurrence_agreement": float,
            "customer_type_agreement": float,
            "capital_intensity_agreement": float,
            "primary_value_driver_agreement": float,
        },
        "n_test_companies": int,
        "n_consistency_samples": int,
    }
    """
```

Also implement:

```python
def generate_eval_report(eval_results: dict) -> str:
    """
    Format evaluation results as a readable text report.
    Write to eval/results.md.
    Return the text content.
    """
```

---

## tests/test_evaluator.py

Write these 5 tests. Mock all API and file I/O calls.

1. `test_precision_at_k_correct_calculation`
   Ground truth for ticker "AAA": ["HON", "EMR", "SWK"]
   Pipeline top 15 for "AAA" includes "HON" and "EMR" but not "SWK"
   Assert precision = 2/3 = 0.667

2. `test_precision_zero_when_no_overlap`
   Ground truth: ["HON", "EMR"]
   Pipeline top 15: ["MMM", "GE"] (no overlap)
   Assert precision = 0.0

3. `test_precision_one_when_full_overlap`
   Ground truth: ["HON", "EMR"]
   Pipeline top 15 includes both "HON" and "EMR"
   Assert precision = 1.0

4. `test_target_excluded_from_evaluation`
   Test company is "MMM". "MMM" appears in ground truth.
   Assert "MMM" is excluded from both pipeline output and ground truth
   before calculating precision.

5. `test_eval_report_written_to_file`
   Run generate_eval_report with sample results dict.
   Assert `eval/results.md` file was created.

---

## After Tests Pass: Run the Actual Evaluation

Run the full evaluation pipeline manually:

1. Call `ground_truth_builder.build_ground_truth(TEST_TICKERS, config)`
   - Expected: at least 10 of 15 test companies return non-empty peer groups
   - If fewer than 8 succeed, the EDGAR fetch or LLM extraction needs debugging

2. Call `evaluator.run_evaluation(...)`

3. Call `evaluator.generate_eval_report(results)` to write `eval/results.md`

4. Manually fill in `eval/manual_review_sample.txt` by reviewing 15 samples
   (this is the human part — takes 1-2 hours)

---

## eval/results.md (write this manually after running evaluation)

This is NOT generated code — write it yourself after you have the numbers.
Format it as a markdown file documenting your evaluation findings:

```markdown
# Evaluation Results

## Precision@15 vs SEC Proxy Peer Groups
- Mean: XX%
- Median: XX%
- Test companies: N

Interpretation: [1-2 sentences explaining what this means]

## LLM Extraction Consistency (30-company sample)
- business_model agreement: XX%
- revenue_recurrence agreement: XX%
- customer_type agreement: XX%
- capital_intensity agreement: XX%
- primary_value_driver agreement: XX%

## Manual Review Results (15 companies)
- business_model accuracy: X/15 (XX%)
- revenue_recurrence accuracy: X/15 (XX%)
- customer_type accuracy: X/15 (XX%)

## Key Findings
- [What worked well]
- [What did not work well]
- [Any unexpected patterns]
```

---

## What NOT to do

- Do NOT modify Phase 1-4 modules
- Do NOT make real API calls in tests
- Do NOT try to automate the manual review step
- Do NOT skip the manual review — those numbers are needed for answering Q3

---

## Definition of Done for Phase 5

1. `pytest tests/ -v` passes with 0 failures (27 existing + 5 new = 32 total)

2. Ground truth built for at least 10 of the 15 test companies:
   - `data/cache/peer_group_{ticker}.json` files exist for at least 10 tickers
   - Each contains at least 8 peer tickers

3. Evaluation metrics recorded:
   - `eval/results.md` exists and contains actual numbers (not placeholders)
   - Mean Precision@15 is documented (acceptable range: 40-70%)
   - LLM consistency rates are documented for all 5 fields

4. Manual review completed:
   - `eval/manual_review_sample.txt` has been filled in by hand
   - Accuracy rates are in `eval/results.md`
   - business_model accuracy should be at least 75% to be useful

5. If Mean Precision@15 is below 35%:
   - Investigate whether the LLM filtering step is working correctly
   - Check whether the residual-based ranking logic in scorer.py is
     selecting companies with extreme multiples rather than representative ones
   - Do NOT proceed to Phase 6 until this is debugged
