# PE Comps Pipeline

## Overview

PE Comps Pipeline helps private equity analysts build a ranked set of public
company comparables for benchmarking a private target company. Given the
target's business description, financial profile, and relevant SIC codes, the
pipeline discovers public peers, enriches them with financial and business-model
features, and produces CSV/HTML comps reports with valuation multiples,
benchmark distributions, and a ranked Top-N comp list selected by business-model
fit and financial-feature similarity to the target.

At a high level, the pipeline:

- discovers candidate tickers dynamically from SEC EDGAR SIC-code filings;
- separates directly relevant SIC codes from adjacent codes used to add
  diversity to the comp pool;
- pulls financial statement data and business descriptions from SEC EDGAR XBRL;
- enriches market capitalization and sector data from Financial Modeling Prep
  (FMP);
- uses OpenAI models to extract structured business-model attributes and run a
  second-pass confidence judge;
- scores each company by its standardized financial-feature distance to the
  target (revenue scale, margins, growth, leverage, capex intensity); and
- ranks comps tier-first (Core / Secondary / Review-Exclude), ordering within
  each tier by that distance plus penalties for business-model, customer-type,
  end-market, and revenue-scale mismatch.

## Sample Output

A full sample run (synthetic target, non-confidential) is committed under
[`docs/samples/`](docs/samples/):

- [`sample_report.html`](docs/samples/sample_report.html) — the generated HTML
  comps report (tiered comp table, implied-valuation football field,
  benchmarks, near-miss audit trail, provenance footer).
- [`sample_report.csv`](docs/samples/sample_report.csv) — the matching CSV.
- [`ui_screenshot.png`](docs/samples/ui_screenshot.png) — the sample target
  entry UI.

<img src="docs/samples/ui_screenshot.png" alt="Sample target entry UI" width="720">

## Key Capabilities

- Multi-source financial-data ingestion with caching, retries, and failure
  reporting.
- LLM-assisted structured extraction with confidence scoring and checkpointed
  batch processing.
- Feature engineering across financial metrics and categorical business-model
  attributes.
- Comp selection via LLM-derived hard/soft business-attribute filters combined
  with nearest-neighbor distance on standardized financial features (rationale
  under [5] Scorer below).
- A reproducible CLI workflow with pytest coverage, Ruff linting, targeted
  mypy checks on core typed modules, and GitHub Actions CI.

## Quick Start

Python 3.11 is the recommended local runtime (the package supports
>=3.10,<3.14):

```bash
git clone <repo-url>
cd pe_comps_pipeline
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For a reproducible install (the exact versions CI tests against), use the lock
file instead of the loose ranges:

```bash
pip install -r requirements.lock
pip install -e . --no-deps
```

You can validate the codebase without API keys. `mypy` uses the targeted module
set configured in `pyproject.toml`, not a whole-repository strict pass:

```bash
ruff check .
mypy
pytest tests/ -v --cov --cov-fail-under=70
```

To run the full pipeline, copy [`.env.example`](.env.example) to `.env` in the
project root and fill in your API keys (`.env` is gitignored — never commit
keys). `SEC_IDENTITY` is used as the SEC EDGAR User-Agent / contact identity:

```bash
cp .env.example .env
```

python-dotenv loads `.env` automatically; no manual `export` needed.

Edit `config.yaml` with your target company details, then:

```bash
pe-comps                       # run the pipeline (= python -m src.pipeline)
pe-comps-ui                    # optional web UI at http://127.0.0.1:7860
pe-comps --suggest-sic-codes   # advisory SIC suggestions from the description
```

If you don't already know which SIC codes fit the target, leave
`primary_sic_codes: []` as a placeholder and run `--suggest-sic-codes` first —
suggestions are validated against SEC's official SIC list and never modify
`config.yaml`. All environment variables, including optional quota/parallelism
knobs, are documented in [`docs/configuration.md`](docs/configuration.md).

For tuning runs where the SIC universe and fetched company data stay the same,
save the expensive discovery/fetch/LLM extraction layer once and reuse it while
adjusting target estimates or scorer weights:

```bash
pe-comps --save-universe outputs/universe/demo
pe-comps --reuse-universe outputs/universe/demo
```

## Configuration

All settings live in `config.yaml`. For a normal run, the fields you usually
edit are:

- `target_company.name` / `description`
- `target_company.revenue_usd_mm` / `ebitda_margin_estimate`
- `target_company.primary_sic_codes` / `adjacent_sic_codes`
- `output.top_n_comps`

The minimum useful setup is a target name, business description, revenue,
EBITDA-margin estimate, and at least one primary SIC code. If the SIC choice is
unclear, start with `pe-comps --suggest-sic-codes`, review the advisory output,
then paste the selected codes into `config.yaml`.

The full field reference — every `config.yaml` section, the Web UI, and all
environment variables (`SEC_IDENTITY`, `FMP_DAILY_PROFILE_BUDGET`,
`EDGAR_FETCH_WORKERS`, ...) — is in
[`docs/configuration.md`](docs/configuration.md).

## Understanding the Output

The run produces `outputs/comps_report.csv` / `outputs/comps_report.html`. Use
the CSV for downstream analysis and the HTML report for review or sharing. The
HTML report has 8 sections: selection snapshot, tiered Top-N comp table,
qualitative fit review, valuation & financial benchmarks (with implied-EV
ranges and a football field), near-miss audit trail, selection diagnostics,
data notes, and methodology notes. Start with the Top-N table, near-miss audit
trail, and data notes when checking whether the comp set is usable.
Section-by-section detail is in
[`docs/report_guide.md`](docs/report_guide.md).

## Architecture

```
config.yaml
   |
   v
[1] Universe Builder -> [2] Fetcher -> [3] LLM Analyzer
                                             |
                                             v
                              [4] Feature Builder -> [5] Scorer
                                                        |
                                                        v
                                                   [6] Reporter
```

1. **Universe Builder** (`src/universe_builder.py`) — discovers candidate
   tickers by SIC code via SEC EDGAR (`src/sic_universe_builder.py`), split
   into primary/adjacent buckets per `primary_allocation_pct`, then applies
   seed ticker expansion and `must_include_tickers` / `exclude_tickers`
   analyst overrides.
2. **Fetcher** (`src/fetcher.py`) — financials + business description from
   SEC EDGAR (via `edgartools`); market cap, sector, and a description
   fallback from FMP's `/profile` endpoint — the only FMP endpoint this
   pipeline calls, by design. EV/EBITDA and EV/Revenue are derived from that
   market cap plus an EDGAR-sourced enterprise-value bridge (the bridge and
   FMP-endpoint choice are detailed in
   [`docs/data_layer.md`](docs/data_layer.md)). Caches every company to
   `data/cache/{ticker}.json` (layered TTLs: quarterly for fundamentals,
   daily for market data). After fetching, `pipeline.py` applies the
   market-cap floor and the configured revenue/EBITDA-margin bands — no
   additional FMP calls, and missing values never disqualify a company.
3. **LLM Analyzer** (`src/llm_analyzer.py`) — OpenAI extracts business-model
   fields per company, citing a verbatim `evidence_quote` from the source
   description, then a second model pass judges extraction quality. The
   quote is checked programmatically (substring match, whitespace-
   normalized) rather than trusted — a hallucinated quote, or missing core
   business-profile fields, forces `low_confidence_flag=True` regardless of
   the judge score. Checkpoints to
   `data/checkpoints/llm_checkpoint.json` for resumable runs; entries are
   invalidated by content (description hash + extraction model), not just
   ticker. `suggest_sic_codes()` is a separate, advisory-only entry point
   whose suggestions are validated against `src/sec_sic_codes.csv`, a
   vendored copy of SEC's official SIC list.
4. **Feature Builder** (`src/feature_builder.py`) — builds the 6-column
   financial-feature matrix (revenue scale, margins, growth, leverage, capex
   intensity) used
   for distance scoring, with the target's `ev_ebitda` as a label column
   (kept for reference; no regression is fit against it).
5. **Scorer** (`src/scorer.py`) — standardizes each company's financial
   features and computes its (optionally weighted — see
   `scorer.feature_weights`) Euclidean distance to the target's standardized
   profile. This distance-to-target is what the report ranks comps by; it's
   directly interpretable per company and doesn't depend on fitting a model
   to a sub-100-row training pool.
6. **Reporter** (`src/reporter.py`) — assembles and renders the CSV/HTML
   report. Selection semantics (soft penalties, tiering, audit trail) live
   in `src/report_selection.py`, valuation math (implied-EV ranges, size
   screens, dispersion diagnostics) in `src/report_valuation.py`, and the
   SVG charts in `src/report_charts.py`; `reporter.py` composes them into
   the report context and writes the outputs.

`src/fmp_client.py` and `src/sic_universe_builder.py` aren't numbered above —
they're thin data-source clients called *by* the numbered stage modules, not
stages themselves. The naming convention: a module named after a pipeline
*role* (`fetcher`, `universe_builder`, `scorer`, ...) is something
`pipeline.py` calls directly; a module named after a *data source*
(`fmp_client`, `sic_universe_builder`) is a dependency that a role module
calls into.

Cross-cutting infrastructure: `src/json_store.py` (atomic JSON writes with
corrupt-file tolerance for every cache/checkpoint), `src/run_lock.py`
(cross-process single-runner lock shared by CLI and UI), `src/paths.py`
(project-root-anchored paths so runs work from any working directory), and
`src/defaults.py` (shared defaults such as the SEC identity).

## Evaluation

This repository includes a reviewed manual ground-truth data set for
fairness-opinion comparable-company evaluation. The harness separates discovery
coverage from ranking quality, so a missed banker comp can be attributed to the
SIC universe, financial filters, data availability, or the final Top-K ranking
step.

- `eval/evaluator.py` is the Precision@K harness. Its Top-K selection shares
  the production ranking core (`src/report_selection.py`), so the evaluation
  cannot drift from what the report actually does.
- `eval/ground_truth/manual_deals.json` contains reviewed "Selected Companies
  Analysis" observations with filing URLs, advisors, target financials, and
  still-public flags for banker-selected comps
  (`manual_deals.example.json` is the template for adding more).
- `eval/ground_truth_builder.py` is experimental: it prefills
  `eval/ground_truth/manual_deals.review.json` from merger-proxy filings; a
  human must verify every field before promoting entries into
  `manual_deals.json`.
- The generated report uses run diagnostics and LLM-assisted comp-fit review
  as directional quality signals, not audited ground truth.

The discovery ladder — each rung a separately measured upgrade on the same
10-deal benchmark — currently stands at:

| Discovery mode | Mean P@15 | Discovery-layer losses | Status |
| --- | ---: | ---: | --- |
| `single-sic` | 8.2% | 58 | baseline |
| `sic+embedding` | 9.8% | 54 | optional |
| `suggest-sic` | 19.3% | 43 | **recommended** |
| `suggest-sic+embedding` | 15.3% | 42 | experimental |

The hybrid rung is a deliberate negative result: coverage improved but
ranking diluted, and a three-point ablation traced the binding constraint to
semantic ranking quality rather than corpus capacity — the full analysis,
evaluation caveats, and what was tried and retired are in
[`docs/known_limitations_and_roadmap.md`](docs/known_limitations_and_roadmap.md).

Per-rung results and coverage waterfalls are published in
[`eval/results.md`](eval/results.md) and its mode-suffixed siblings. The
methodology and discovery-ladder analysis are documented in
[`docs/eval_coverage_analysis.md`](docs/eval_coverage_analysis.md).

## Scripts

`scripts/data_quality.py` is a standalone diagnostic, not part of the
pipeline run — it scans every per-ticker file already in `data/cache/` and
writes a field-completeness / EV-EBITDA-distribution report to
`outputs/data_quality_report.txt`:

```bash
python -m scripts.data_quality
```

`scripts/prefill_manual_deal.py` and `scripts/evaluate_manual_deals.py`
support the evaluation workflow below:

```bash
python -m scripts.prefill_manual_deal <edgar-filing-url>
python -m scripts.evaluate_manual_deals
```

## Data Layer and Costs

Free-tier stack: SEC EDGAR (free) + FMP `/profile` (free, quota-capped) +
OpenAI. A full ~150-company run has recently been around $0.40-0.80 with the
default models, but cost depends on model pricing, candidate count, and cache
hit rate; cached reruns cost near zero.
EBITDA-coverage analysis, paid-FMP / CapIQ / Bloomberg upgrade paths, and the
coverage-by-source table are in [`docs/data_layer.md`](docs/data_layer.md).

## Scope and Tradeoffs

- **Discovery model**: SIC-code based with optional LLM-suggested-SIC and
  embedding channels (`universe.discovery_mode`) — explainable and
  reproducible, but hybrid manufacturing/services targets can need adjacent
  SIC codes or seed tickers. Guardrails catch zero-yield and over-broad SIC
  choices before the run burns API calls. The embedding channel is measured
  but experimental — see
  [`docs/known_limitations_and_roadmap.md`](docs/known_limitations_and_roadmap.md).
- **Data coverage**: US public companies with EDGAR 10-K filings — reproducible
  on public data, but non-US comps are excluded upstream by the domicile
  filter. Coverage limits and paid-data upgrade paths are in
  [`docs/data_layer.md`](docs/data_layer.md).
- **Single-runner workflow**: CLI and UI share `outputs/` and checkpoint files,
  so one run per checkout is enforced by `outputs/.run.lock`
  (`src/run_lock.py`); stale locks from crashed runs are reclaimed
  automatically.
- **Comp pool scale**: with well under 100 companies in a typical run, the
  z-scoring behind the distance ranking shifts with whatever is in the pool.
  Tightening adjacent SIC codes helps more than adding candidate rows.
- **LLM extraction**: quality depends on the underlying business description;
  the judge step plus verbatim-quote verification flags weak extractions via
  `low_confidence_flag`.
- **FMP quota / fetch speed**: the free tier is quota-capped and EDGAR fetching
  is serial by default. `FMP_DAILY_PROFILE_BUDGET` turns the quota into explicit
  controlled degradation; `EDGAR_FETCH_WORKERS` enables opt-in parallel fetching
  — see [`docs/configuration.md`](docs/configuration.md#environment-variables).

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
