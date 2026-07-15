# Phase 3: LLM Business Model Extraction

## Project Context

You are building a PE Comparable Company Analysis Pipeline. This tool helps
PE analysts automatically build a comparable company set (comps) for valuing
private companies. The pipeline fetches public company financials + business
descriptions, uses LLM to extract structured business model features, then
uses XGBoost + LLM similarity to rank the most relevant comps.

This is Phase 3 of 6.

---

## What Already Exists (Do Not Modify)

- `src/universe_builder.py` — builds 300-ticker candidate list across 3 industries
- `src/fetcher.py` — fetches financials + business descriptions, with caching
- `src/data_quality.py` — generates data quality report
- `data/cache/` — ~300 JSON files with financial data and business descriptions
- `outputs/data_quality_report.txt` — shows field completeness statistics
- All previous tests passing

---

## What To Build in Phase 3

One new module: `src/llm_analyzer.py`
One new test file: `tests/test_llm_analyzer.py`

---

## src/llm_analyzer.py

### Purpose

For each company in the dataset that has a non-null `business_description`,
call OpenAI's API to extract 5 structured fields that describe the company's
business model. Then call a cheaper model to judge the quality of each
extraction.

### The 5 fields to extract

```python
{
    "business_model": one of:
        "manufacturing" | "services" | "SaaS" | "distribution" |
        "marketplace" | "other",

    "revenue_recurrence": one of:
        "high" | "medium" | "low",
        # high = long-term contracts, subscriptions, recurring orders
        # medium = mix of recurring and project-based
        # low = project-based, one-time, transactional

    "customer_type": one of:
        "B2B" | "B2C" | "B2G" | "mixed",

    "capital_intensity": one of:
        "asset_heavy" | "moderate" | "asset_light",
        # asset_heavy = manufacturing, mining, utilities
        # asset_light = software, consulting, distribution

    "primary_value_driver": one of:
        "technology" | "scale" | "relationships" | "brand" | "other",

    "sub_sector_description": str,
        # One sentence in the model's own words describing what this company
        # specifically does within its industry.
        # Example: "specialty industrial fastener distributor serving aerospace OEMs"
        # This field is free-form text, not constrained to a fixed list.

    "confidence": int between 1 and 5
        # How clearly does the description support the extraction?
        # 5 = very explicit, 1 = largely inferred or ambiguous
}
```

### System prompt (use exactly this)

```
You are a private equity analyst extracting structured information from
company business descriptions to support comparable company analysis.

Rules:
- Output ONLY valid JSON. No explanation, no markdown fences, no preamble.
- Use null for any field where the description provides insufficient
  information to make a reliable determination.
- Base ALL answers strictly on the provided text.
  Do not use any outside knowledge about the company.
- confidence: integer 1-5 reflecting how explicitly the description
  supports your extraction (5 = very explicit).
```

### User prompt template (use exactly this structure)

```
Extract structured fields from this company business description.

Company: {company_name}
Description: {business_description}

Return JSON with exactly these fields:
{
  "business_model": "manufacturing"|"services"|"SaaS"|"distribution"|"marketplace"|"other",
  "revenue_recurrence": "high"|"medium"|"low",
  "customer_type": "B2B"|"B2C"|"B2G"|"mixed",
  "capital_intensity": "asset_heavy"|"moderate"|"asset_light",
  "primary_value_driver": "technology"|"scale"|"relationships"|"brand"|"other",
  "sub_sector_description": "<one sentence describing specific sub-sector>",
  "confidence": <integer 1-5>
}
```

### LLM-as-judge prompt (use exactly this structure)

```
You are reviewing a business model extraction for accuracy.

Original description (first 500 chars): {business_description[:500]}
Extraction result: {extraction_json}

Rate the accuracy of the extraction on a scale of 1-5:
5 = All fields are clearly supported by the text
4 = Most fields accurate, minor uncertainty on 1-2 fields
3 = Partially accurate, some fields unclear or questionable
2 = Several fields appear inaccurate or unsupported
1 = Extraction is largely inaccurate or hallucinated

Return ONLY this JSON (no other text):
{{"score": <integer 1-5>, "reason": "<one sentence>"}}
```

### OpenAI API usage

- Use `openai.OpenAI()` — API key from `OPENAI_API_KEY` environment variable
- Use the responses API: `client.responses.create(...)`
- Never hardcode the API key
- For extraction: model = config['llm']['extraction_model'] (gpt-4.1)
- For judge: model = config['llm']['judge_model'] (gpt-4.1-mini)
- temperature = config['llm']['temperature'] (0)
- max_tokens = config['llm']['max_tokens'] (500)

### JSON parsing — implement robustly

The LLM sometimes wraps output in markdown fences. Handle all these cases:

1. Clean JSON: `{"business_model": "manufacturing", ...}`
   → parse directly

2. Markdown fences: ` ```json\n{...}\n``` `
   → strip the fences, then parse

3. Completely invalid (LLM returned an explanation instead of JSON)
   → log a warning, return None for this company

Never raise an exception from the parsing function — always return
either a valid dict or None.

### Rate limit handling

Wrap all API calls with tenacity:
- Catch `openai.RateLimitError` specifically
- On rate limit: wait 60 seconds before retry
- Maximum 3 retries
- Log a warning when rate limit is hit: `WARNING: Rate limit hit, waiting 60s`

### Checkpoint system — MANDATORY

The LLM step is the slowest and most expensive step. If the program crashes
or is interrupted, it must be able to resume from where it stopped.

- Checkpoint file location: `data/checkpoints/llm_checkpoint.json`
- Format: `{"MMM": {...extraction_result...}, "HON": {...}, ...}`
- On startup: load the checkpoint file if it exists
- Before processing each company: check if its ticker is already in the
  checkpoint — if yes, skip it entirely
- After every `batch_size` companies processed: save the checkpoint to disk
- Also save the checkpoint when the batch function completes

### Output structure per company

The function should return a dict that includes both the extraction and judge:

```python
{
    "business_model": str | None,
    "revenue_recurrence": str | None,
    "customer_type": str | None,
    "capital_intensity": str | None,
    "primary_value_driver": str | None,
    "sub_sector_description": str | None,
    "confidence": int | None,
    "judge_score": int | None,
    "judge_reason": str | None,
    "low_confidence_flag": bool,  # True if judge_score < config threshold
    "extraction_failed": bool,    # True if JSON parsing failed entirely
}
```

### Companies to skip

Skip a company entirely (do not call the API) if:
- `business_description` is None
- `business_description` has fewer than 100 characters
- The ticker is already in the checkpoint

Log skipped companies at DEBUG level: `DEBUG: MMM — skipped (no description)`

### Public interface

```python
def analyze_batch(companies: list[dict], config: dict) -> dict[str, dict]:
    """
    Extract business model features for each company.

    Args:
        companies: list of company dicts from fetcher (must have 'ticker'
                   and 'business_description' fields)
        config: full config dict

    Returns:
        dict mapping ticker -> extraction result dict
        Tickers that were skipped or failed have extraction_failed=True

    Side effects:
        Writes/updates data/checkpoints/llm_checkpoint.json
        Logs progress to pipeline.log
    """
```

Also implement a separate function for analyzing the target company
from the config (used later in Phase 6):

```python
def analyze_target(config: dict) -> dict:
    """
    Run LLM extraction on the target company description from config.
    Returns the same extraction result dict format.
    Does NOT use the checkpoint (target company is always re-analyzed).
    """
```

---

## tests/test_llm_analyzer.py

Write these 7 tests. Mock `openai.OpenAI` — never make real API calls.

1. `test_valid_json_parsed_correctly`
   Mock the API to return a valid JSON string with all 7 fields.
   Assert the returned dict has `business_model` and `judge_score` populated.

2. `test_markdown_fenced_json_parsed`
   Mock the API to return ` ```json\n{"business_model": "manufacturing",...}\n``` `
   Assert the result is successfully parsed (business_model is not None).

3. `test_invalid_json_sets_extraction_failed`
   Mock the API to return "I cannot determine the business model from this text."
   Assert the returned dict has `extraction_failed=True`.

4. `test_low_judge_score_sets_flag`
   Mock extraction to return valid JSON.
   Mock judge to return `{"score": 2, "reason": "unclear description"}`.
   Assert `low_confidence_flag=True` in the result.

5. `test_high_judge_score_does_not_set_flag`
   Mock judge to return score=4.
   Assert `low_confidence_flag=False`.

6. `test_checkpoint_skips_analyzed_ticker`
   Pre-populate the checkpoint with ticker "AAA".
   Call analyze_batch with a company list containing "AAA".
   Assert the OpenAI client was never called for "AAA".

7. `test_company_without_description_skipped`
   Pass a company with `business_description=None`.
   Assert the OpenAI client was never called.
   Assert the returned dict for that ticker has `extraction_failed=True`.

---

## What NOT to do

- Do NOT build feature engineering or ML in this phase
- Do NOT make real OpenAI API calls in tests
- Do NOT hardcode the API key
- Do NOT delete the checkpoint file between test runs
  (tests should mock file I/O or use a temp directory)
- Do NOT implement the full pipeline.py orchestrator yet
- Do NOT call analyze_batch on all 300 companies yet —
  wait until tests pass first, then test on 20-30 companies

---

## Definition of Done for Phase 3

**Step 1 — tests pass:**
- `pytest tests/ -v` passes with 0 failures (10 existing + 7 new = 17 total)

**Step 2 — small batch test (20 companies, real API):**
- Set `batch_size: 5` in config temporarily
- Run analyze_batch on 20 companies (manually call the function in a script)
- Check `data/checkpoints/llm_checkpoint.json` — should have 20 entries
- Interrupt the script mid-run, restart it — the restarted run should
  skip companies already in the checkpoint
- Check 5 random results manually: do the extracted fields make sense
  given the company's actual business? (Look up the company name if needed)
- Judge scores should mostly be 3-5. If most are 1-2, the prompt needs adjustment.

**Step 3 — full batch (all 300 companies, real API):**
- Run analyze_batch on all companies
- Log the final summary:
  - How many companies were successfully extracted
  - How many failed JSON parsing
  - How many were flagged low confidence (judge < 3)
  - Approximate total API cost
- Expected: success rate above 85%, low confidence below 25%
- Expected API cost: under $5 for 300 companies

**Data quality check:**
- Open `data/checkpoints/llm_checkpoint.json`
- Verify entries have the expected structure (all 9 keys present)
- Check that `sub_sector_description` contains useful free-form text
  (not just None for most companies)
