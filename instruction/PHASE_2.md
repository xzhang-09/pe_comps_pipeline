# Phase 2: Scale Up + Data Quality Report

## Project Context

You are building a PE Comparable Company Analysis Pipeline. This tool helps
PE analysts automatically build a comparable company set for valuing private
companies, by pulling public company financials and business descriptions,
then using LLM + ML to rank the most relevant comps.

This is Phase 2 of 6.

---

## What Already Exists (Do Not Modify)

- `src/universe_builder.py` — builds candidate ticker list
- `src/fetcher.py` — fetches financials + business descriptions with caching,
  retry, logging, and failed ticker tracking
- `tests/test_fetcher.py` — 6 passing tests
- `data/cache/` — contains ~50 JSON files from Phase 1
- `config.yaml` — currently has `max_candidates: 50`

All Phase 1 tests must still pass at the end of this phase.

---

## What To Build in Phase 2

Three things:
1. Update `config.yaml` to expand to 300 candidates
2. Add multi-industry support to `universe_builder.py`
3. Build a new `src/data_quality.py` module that generates a data quality report

---

## Task 1: Update config.yaml

Change `max_candidates` from 50 to 300.

Add a new `industries` section under `universe`:

```yaml
universe:
  max_candidates: 300
  min_revenue_usd_mm: 30
  max_revenue_usd_mm: 800
  min_ebitda_margin: 0.05
  industries:
    - gics_sector: "20"
      label: "Industrials"
    - gics_sector: "35"
      label: "Healthcare Equipment"
    - gics_sector: "45"
      label: "Technology Hardware"
```

This tells the pipeline to pull candidates from three different industries,
roughly 100 per industry, to reach the 300 total target.

---

## Task 2: Update universe_builder.py

### Changes needed

The current `build()` function pulls from one sector. Update it to:

- Read the `industries` list from config
- For each industry in the list, get a set of candidate tickers
- Merge all tickers, deduplicate
- Apply the same market cap filter (> $200M) as before
- Truncate to `max_candidates` total
- Log how many tickers came from each industry

### Expanded hardcoded fallback lists

Add fallback ticker lists for the two new sectors:

**Healthcare Equipment (GICS 35) — at least 30 tickers:**
ABT, MDT, SYK, BSX, ZBH, EW, HOLX, DXCM, ISRG, RMD, BAX, BDX,
HAE, ICU, AMED, NVCR, RGEN, IART, MMSI, NPWT, OSIS, ATRC, NTRA,
INSP, SWAV, AXNX, TNDM, NVST, ALGN, VRTX, and at least 5 more.

**Technology Hardware (GICS 45) — at least 30 tickers:**
AAPL, DELL, HPQ, HPE, NTAP, STX, WDC, PSTG, SMCI, JNPR, CSCO,
ANET, CIEN, VIAV, CLFD, ATEN, ARLO, CALX, DIGI, LIQT, PCTI, PCTEL,
SIFY, SMSI, SPOK, SYNA, TTEC, UTSI, VNET, XTLB, and at least 5 more.

---

## Task 3: Build src/data_quality.py

### Purpose

After fetching all 300 companies, generate a data quality report that documents:
- How complete the data is (field-by-field missing rates)
- Whether EV/EBITDA distribution is reasonable
- Industry breakdown of the dataset
- How many companies have business descriptions

This report is for your own use during development — it tells you what to
expect in Phase 4 when you build the ML model.

### What to calculate

**For each numeric field** (revenue_ttm_usd_mm, ebitda_margin, gross_margin,
revenue_cagr_3yr, net_debt_ebitda, capex_revenue, ev_ebitda, ev_revenue):
- Count of non-null values
- Missing rate as a percentage
- Min, 25th percentile, median, 75th percentile, max (excluding nulls)

**For ev_ebitda specifically:**
- Distribution after removing outliers (already set to None by fetcher)
- Flag if median ev_ebitda is outside the range 6x–20x (would indicate
  data quality problem)

**For business_description:**
- Count of non-null values
- Average character length of non-null descriptions
- Count of descriptions shorter than 200 characters (likely truncated)

**Industry breakdown:**
- How many companies per GICS sector in the dataset

### Output

Write a plain text report to `outputs/data_quality_report.txt`.

Format it so it is readable without any special software — use dashes and
spaces to align columns, not any visualization library.

Example format:
```
=== DATA QUALITY REPORT ===
Generated: 2025-01-15 09:23:41
Total companies fetched: 287 / 300 attempted

--- FIELD COMPLETENESS ---
Field                   Non-null    Missing%    Median
revenue_ttm_usd_mm      281         6.3%        245.0
ebitda_margin           201         30.0%       0.18
ev_ebitda               198         31.0%       12.4
...

--- EV/EBITDA DISTRIBUTION ---
Min: 3.2x   P25: 8.1x   Median: 12.4x   P75: 17.8x   Max: 48.3x
Status: OK (median within expected range 6x-20x)

--- BUSINESS DESCRIPTIONS ---
Non-null: 261 / 287 (91%)
Avg length: 847 chars
Short (<200 chars): 14 companies

--- INDUSTRY BREAKDOWN ---
Industrials (20):         98 companies
Healthcare Equipment (35): 94 companies
Technology Hardware (45):  95 companies
```

### Interface

```python
def generate_report(companies: list[dict]) -> str:
    """
    Generate data quality report from list of company dicts.
    Writes report to outputs/data_quality_report.txt.
    Returns the report text as a string.
    """
```

Add a simple runner at the bottom:
```python
if __name__ == "__main__":
    # Load all cache files from data/cache/
    # Pass to generate_report()
    # Print the report to console as well
```

---

## Task 4: Add tests/test_data_quality.py

Write these 4 tests:

1. `test_report_contains_all_fields`
   Pass a list of 5 sample companies.
   Assert the returned string contains "ebitda_margin" and "ev_ebitda".

2. `test_missing_rate_calculation`
   Pass 4 companies where 2 have ev_ebitda=None and 2 have valid values.
   Assert the reported missing rate for ev_ebitda is "50.0%".

3. `test_report_file_written`
   Run generate_report with sample data.
   Assert that outputs/data_quality_report.txt was created.

4. `test_short_description_count`
   Pass companies where 2 have descriptions shorter than 200 characters.
   Assert the report counts 2 short descriptions.

---

## Task 5: Run the full data fetch and generate the report

After implementing the above, run the actual fetch:

1. Call `universe_builder.build(config)` to get ~300 tickers
2. Call `fetcher.fetch_batch(tickers, config)` to fetch all companies
   (this will use caches from Phase 1 for tickers already fetched)
3. Call `data_quality.generate_report(companies)` to produce the report
4. Review `outputs/data_quality_report.txt` and note:
   - The actual missing rate for `ev_ebitda` (critical — this determines
     how many training samples the ML model will have)
   - Whether median EV/EBITDA is in the 6x–20x range
   - Whether at least 200 companies have valid business descriptions

---

## What NOT to do

- Do NOT modify fetcher.py's core logic (only config changes affect it)
- Do NOT implement LLM or ML in this phase
- Do NOT delete existing cache files — they should be reused
- Do NOT use any visualization libraries (matplotlib, seaborn) in the
  data quality report — plain text only
- Do NOT modify the existing 6 tests in test_fetcher.py

---

## Definition of Done for Phase 2

1. `pytest tests/ -v` passes with 0 failures (original 6 + 4 new = 10 total)
2. `outputs/data_quality_report.txt` exists and is readable
3. Report shows at least 200 companies with valid ev_ebitda values
   (if fewer than 150, there is a data problem — investigate before Phase 4)
4. Report shows at least 230 companies with valid business descriptions
5. Median ev_ebitda in the report is between 6x and 25x
6. `data/cache/` contains approximately 300 JSON files
7. `logs/pipeline.log` shows entries from 3 different GICS sectors
