# Phase 1: Data Foundation

## Project Context

You are building a PE Comparable Company Analysis Pipeline. The full project
will help PE analysts automatically build a set of comparable public companies
to benchmark a private target company for valuation purposes.

This is Phase 1 of 6. In this phase you are only building the data foundation:
fetching financial data and business descriptions for 50 public companies.
Do NOT build LLM, ML, or reporting functionality yet.

---

## What Exists

Nothing. This is the first phase. Create the entire project structure from scratch.

---

## What To Build

### Project structure to create

```
pe_comps_pipeline/
├── requirements.txt
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── universe_builder.py
│   └── fetcher.py
├── data/
│   ├── cache/          (empty folder, add .gitkeep)
│   └── checkpoints/    (empty folder, add .gitkeep)
├── outputs/            (empty folder, add .gitkeep)
├── logs/               (empty folder, add .gitkeep)
└── tests/
    ├── conftest.py
    └── test_fetcher.py
```

---

## requirements.txt

Include exactly these packages:

```
yfinance==0.2.54
openai>=1.30.0
xgboost>=2.0.0
shap>=0.45.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
requests>=2.31.0
tenacity>=8.2.0
pyyaml>=6.0.1
jinja2>=3.1.0
matplotlib>=3.8.0
pytest>=8.0.0
pytest-mock>=3.12.0
```

---

## config.yaml

```yaml
target_company:
  name: "Example Manufacturing Co."
  description: >
    Mid-market industrial parts manufacturer serving automotive OEMs
    in North America. Primarily B2B with long-term supply contracts.
    Approximately 70% of revenue is recurring. Asset-heavy operations
    with significant manufacturing facilities.
  gics_sector: "20"
  gics_industry: "2010"
  revenue_usd_mm: 150
  ebitda_margin_estimate: 0.18
  geography: "north_america"

universe:
  max_candidates: 50        # Phase 1: small number for testing
  min_revenue_usd_mm: 30
  max_revenue_usd_mm: 800
  min_ebitda_margin: 0.05

llm:
  extraction_model: "gpt-4.1"
  judge_model: "gpt-4.1-mini"
  temperature: 0
  max_tokens: 500
  batch_size: 20
  judge_threshold: 3

output:
  top_n_comps: 15
  report_formats:
    - csv
    - html
```

---

## src/universe_builder.py

### Purpose
Return a list of ticker symbols for public companies in the configured sector.

### Behavior
- Accept the full config dict as input
- Use the `gics_sector` value from config to filter companies
- The primary method should attempt to use yfinance to get sector tickers
- yfinance's screener is unstable — always fall back to a hardcoded list
- The hardcoded fallback list must contain at least 80 real US-listed industrial
  and manufacturing company tickers. Include real tickers such as:
  MMM, HON, GE, EMR, ITW, PH, ROK, AME, IEX, GNRC, SWK, PNR, RRX,
  XYL, FELE, AIXI, TT, IR, CARR, OTIS, LMT, RTX, GD, NOC, LHX, TDG,
  HII, MOOG, HEICO, TransDigm, ESAB, ITT, WATTS, REXNORD, EVERI,
  ACCO, LYTS, KBAL, GFF, ASTE, HCSG, HI, CSWI, NVT, TRMK, DXPE,
  and at least 30 more real industrial tickers.
- After getting the list, filter out tickers where market cap is below $200M
  (fetch market cap via yfinance info)
- Limit the returned list to config['universe']['max_candidates'] tickers
- Log how many candidates were found before and after filtering

### Interface
```python
def build(config: dict) -> list[str]:
    """
    Return list of candidate ticker symbols.
    Uses yfinance screener with hardcoded fallback.
    Filters out micro-caps below $200M market cap.
    Respects config['universe']['max_candidates'] limit.
    """
```

---

## src/fetcher.py

### Purpose
For each ticker in the candidate list, fetch financial metrics from yfinance
and a business description from SEC EDGAR (fallback to yfinance).

### Data to fetch per company

Return a dict with these exact keys for each company:

```python
{
    "ticker": str,
    "company_name": str,
    "market_cap_usd_mm": float | None,
    "revenue_ttm_usd_mm": float | None,
    "ebitda_margin": float | None,        # EBITDA / Revenue
    "gross_margin": float | None,
    "revenue_cagr_3yr": float | None,     # 3-year revenue CAGR
    "net_debt_ebitda": float | None,
    "capex_revenue": float | None,
    "ev_ebitda": float | None,            # THE ML LABEL — critical field
    "ev_revenue": float | None,
    "gics_sector": str | None,
    "business_description": str | None,
    "description_source": str | None,     # "edgar" or "yfinance"
    "fetch_timestamp": str,               # ISO format datetime string
}
```

### Business description from SEC EDGAR

- Use the EDGAR full-text search endpoint:
  `https://efts.sec.gov/LATEST/search-index?q="{ticker}"&dateRange=custom&startdt=2022-01-01&forms=10-K`
- Set the User-Agent header to: `"PE-Comps-Pipeline research@example.com"`
  (EDGAR requires a user-agent — without it requests get blocked)
- From the returned filing URL, fetch the 10-K document and extract the
  first 1500 characters of the Item 1 (Business) section
- If EDGAR fails for any reason, fall back to `yfinance.Ticker(ticker).info.get("longBusinessSummary")`
- If both fail, set `business_description` to None

### EBITDA margin calculation

yfinance does not always have a clean "ebitda_margin" field. Calculate it:
- Get EBITDA from `yfinance.Ticker(ticker).financials` (row labeled "EBITDA"
  or "Normalized EBITDA")
- Get Revenue from the same financials object
- Calculate margin = EBITDA / Revenue
- Use the most recent fiscal year values

### Revenue CAGR calculation

- Get last 3 years of annual revenue from yfinance financials
- CAGR = (revenue_year_0 / revenue_year_minus_3) ^ (1/3) - 1
- If fewer than 3 years of data available, set to None

### MANDATORY engineering requirements

**Caching — implement exactly as described:**
- Before any API call, check if `data/cache/{ticker}.json` exists
- If it exists, load and return it immediately without any API calls
- After a successful fetch, save the full dict to `data/cache/{ticker}.json`
- Use UTF-8 encoding for all file operations

**Retry with tenacity:**
- Wrap the yfinance fetch in a tenacity retry decorator
- 3 attempts maximum
- Exponential backoff: wait 1s, then 2s, then 4s between attempts
- Catch all exceptions (use `Exception` as the catch-all)

**Failed ticker tracking:**
- When a ticker fails all 3 retries, append a row to `outputs/failed_tickers.csv`
- CSV columns: ticker, error_type, error_message, timestamp
- Use append mode — do not overwrite existing rows on each run
- Create the file with headers if it does not exist

**Data validation — apply after fetching, before caching:**
- If ev_ebitda > 100 or ev_ebitda < 0: set to None and log a warning
- If ebitda_margin > 0.80 or ebitda_margin < -0.50: set to None and log warning
- If revenue_ttm_usd_mm <= 0: set to None and log warning
- If gross_margin > 1.0 or gross_margin < 0: set to None and log warning

**Logging:**
- Every ticker processed: `INFO: Processing 12/50: MMM`
- Cache hit: `INFO: MMM — loaded from cache`
- Missing field: `WARNING: MMM — EBITDA not available, setting None`
- Retry attempt: `WARNING: MMM — attempt 2 failed: {error}. Retrying...`
- Total failure: `ERROR: MMM — failed after 3 retries: {error}`

**Public interface:**
```python
def fetch_batch(tickers: list[str], config: dict) -> list[dict]:
    """
    Fetch financial data for all tickers.
    Returns list of dicts (one per ticker, including failed ones with None values).
    Uses cache — skips API call if cache/{ticker}.json exists.
    Writes failures to outputs/failed_tickers.csv.
    """
```

---

## tests/conftest.py

Define these fixtures:

```python
# sample_company: a complete dict matching the fetcher output schema,
# with all fields populated with realistic values.
# ticker="TEST", ev_ebitda=12.0, ebitda_margin=0.20, etc.

# sample_config: a minimal config dict matching config.yaml structure
# Use max_candidates=10 in the test config
```

---

## tests/test_fetcher.py

Write these 6 tests. Use pytest-mock to mock yfinance. Never make real network calls.

1. `test_cache_hit_skips_api_call`
   Create a cache file for ticker "AAA" before calling fetch_batch.
   Assert that yfinance.Ticker was never called for that ticker.

2. `test_cache_miss_calls_api`
   No cache file exists. Assert that yfinance.Ticker is called.

3. `test_retry_called_three_times_on_failure`
   Mock yfinance.Ticker to always raise an Exception.
   Call the internal fetch function.
   Assert it was retried 3 times total.

4. `test_failed_ticker_written_to_csv`
   Mock yfinance to always fail.
   Run fetch_batch with one ticker.
   Assert that outputs/failed_tickers.csv exists and contains that ticker.

5. `test_ev_ebitda_outlier_set_to_none`
   Mock yfinance to return ev_ebitda=150.
   Assert the returned dict has ev_ebitda=None.

6. `test_negative_ev_ebitda_set_to_none`
   Mock yfinance to return ev_ebitda=-3.
   Assert the returned dict has ev_ebitda=None.

---

## Logging setup

Create a helper function in `src/__init__.py` or at the top of each module:

```python
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        file_handler = RotatingFileHandler(
            "logs/pipeline.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(fmt))
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(fmt))
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger
```

Use `get_logger(__name__)` at the top of every module.

---

## What NOT to do

- Do NOT implement LLM, ML, or reporting in this phase
- Do NOT make real API calls in tests
- Do NOT hardcode API keys anywhere
- Do NOT use print() for status messages — use logging
- Do NOT skip the cache implementation — it is essential for later phases
- Do NOT implement the full pipeline.py yet

---

## Definition of Done for Phase 1

Run this sequence manually to verify:

1. `pip install -r requirements.txt` completes without errors
2. Run a small script that calls `universe_builder.build(config)` and prints the list
   — should return a non-empty list of ticker strings
3. Run a small script that calls `fetcher.fetch_batch(tickers[:5], config)`
   — should complete without crashing
   — `data/cache/` should contain 5 JSON files after the run
   — Running it again should be noticeably faster (cache hits)
4. Manually open 2-3 cache JSON files and verify:
   — `ev_ebitda` field exists (may be None, that is acceptable)
   — `business_description` field exists with real text (not None for most)
   — `ebitda_margin` field exists (may be None)
5. `outputs/failed_tickers.csv` exists
6. `logs/pipeline.log` exists and contains INFO entries
7. `pytest tests/ -v` passes with 0 failures
