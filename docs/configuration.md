# Configuration Guide

All settings live in `config.yaml` (validated against `src/config_schema.py`;
unknown fields are rejected). This page is the full field reference — for the
minimal set most runs need, see the [README's Configuration
section](../README.md#configuration).

## Web UI

For internal users who should not edit YAML or run CLI commands directly, start
the Gradio UI:

```bash
pe-comps-ui
```

The server listens on `http://127.0.0.1:7860` by default. API keys are not
entered in the page; keep `OPENAI_API_KEY`, `FMP_API_KEY`, and `SEC_IDENTITY` in
the server environment or `.env`. Each UI run writes its generated config and
copies its HTML/CSV outputs into `outputs/ui_runs/<run-id>/`, so reports from
separate runs do not overwrite each other.

## Environment variables

Set these in `.env` in the project root (gitignored; loaded automatically by
python-dotenv) or in the server environment:

- `OPENAI_API_KEY` (required): LLM extraction, judging, and embeddings.
- `FMP_API_KEY` (required): FMP `/profile` lookups. `FMP_API_KEY_ALTERNATE`
  (optional) is tried on 402/403/429 responses.
- `SEC_IDENTITY` (strongly recommended): SEC EDGAR User-Agent/contact identity,
  e.g. `"Your Name your.email@example.com"`. A placeholder is used with a
  logged warning if unset.
- `FMP_DAILY_PROFILE_BUDGET` (optional): hard cap on FMP calls per run. Once a
  run has made `n` FMP calls it stops calling FMP, logs a warning per skipped
  ticker, and leaves the affected records' market data as None (they then drop
  out of EV-multiple scoring) — a controlled degradation rather than a mid-run
  surprise at the provider's quota wall.
- `EDGAR_FETCH_WORKERS` (optional): EDGAR fundamentals are fetched serially by
  default; set this to fetch uncached records with `n` threads. The FMP
  market-data layer stays serialized regardless, so its rate limit/budget
  accounting is unaffected. Opt-in because SEC courtesy rate limits apply —
  keep `n` modest (2-4) and watch for 429s in the log.

## `config.yaml` schema

### `target_company` — the private company you're valuing

- `name`, `description`: used in the report and sent to the LLM extractor.
- `revenue_usd_mm`, `ebitda_margin_estimate`, and the optional
  `gross_margin_estimate`, `revenue_cagr_3yr_estimate`,
  `net_debt_ebitda_estimate`, `capex_revenue_estimate`: your estimates for
  the target's six financial features. Each one you provide feeds directly
  into the financial-feature distance used for comp scoring. The latter four
  are optional because most analysts have revenue and EBITDA margin on hand
  first, but supplying the others (from a CIM or management accounts) gives
  a sharper match.
- Any feature you leave unset is **excluded from the distance entirely**, not
  silently filled with a peer median and scored as if it were a real signal —
  otherwise an imputed feature pulls the ranking toward whichever comps sit
  near the pool median on that axis, which is spurious. The distance is
  computed only over the features you actually provided
  (`scorer._target_financial_row` reports these as the "observed" features;
  `scorer._distance_to_target` zeroes the rest). An imputed median value is
  still used wherever a complete target row is needed for display (e.g. the
  benchmark table), specifically the median within companies sharing the
  target's own `business_model` (from `analyze_target()`), not the whole comp
  pool, so a light-asset target doesn't get a traditional manufacturer's
  `capex_revenue` imputed onto it — falling back to the pool-wide median if
  that business-model group is too small
  (`feature_builder.MIN_GROUP_SIZE_FOR_IMPUTATION`) to trust its own median.
  If you provide *no* financial features at all, scoring falls back to the
  old behaviour (all six features, fully imputed) so the run still produces a
  ranking.
- `geography`: descriptive only.
- `primary_sic_codes` / `adjacent_sic_codes`: SIC codes used to discover
  candidates from SEC EDGAR (`src/sic_universe_builder.py`). `primary`
  should be the SIC codes most directly matching the target's actual
  business — these are the ones expected to actually surface as comps.
  `adjacent` is a broader, related set included only to add training-data
  volume/diversity for the comp pool; candidates from it are not expected
  to be selected as comps themselves. These are filled in manually — if
  you don't already know which codes fit, run `pe-comps --suggest-sic-codes`
  for advisory candidates. Suggestions are LLM-generated from `description`
  and then validated against the vendored SEC SIC list; the analyst still
  decides whether each valid code is a good primary/adjacent business fit.

### `universe`

- `max_candidates`: upper bound on the total candidate pool taken from SIC
  discovery, split between the primary/adjacent buckets per
  `primary_allocation_pct`. Market-cap filtering happens later, after
  fetching (see the README's Architecture step 2), so the final count of
  companies that reach LLM analysis/scoring can be lower than this.
- `primary_allocation_pct`: what fraction of `max_candidates` is reserved
  for the primary bucket vs. the adjacent bucket (e.g. `0.5` splits it
  evenly). If a bucket's actual candidate count is below its quota, the
  unused quota is simply not filled — it's not redistributed to the other
  bucket.
- `min_revenue_usd_mm` / `max_revenue_usd_mm` / `min_ebitda_margin`: optional
  hard filters applied after financial data is fetched and market-cap
  filtering runs. Missing revenue or margin values are kept so data gaps do
  not silently discard candidates.
- `discovery_mode`: `"sic"` (default) uses only primary/adjacent SIC discovery.
  `"sic+embedding"` keeps the SIC pool and adds an extra semantic-discovery
  bucket from companies in the same 2-digit SIC families as the configured
  primary/adjacent codes.
- `embedding_top_n` / `embedding_min_similarity`: cap and threshold for that
  optional semantic-discovery bucket. `max_candidates` still controls the SIC
  primary/adjacent quota; `embedding_top_n` is an explicit additional recall
  budget.
- `seed_tickers`: optional public tickers for targets whose SIC codes miss
  relevant hybrid/service-like peers. Each seed's SEC SIC code is added to
  adjacent discovery, and the seed itself is injected as an analyst-specified
  candidate.
- `must_include_tickers` / `exclude_tickers`: optional analyst override lists.
  Must-include tickers are injected into the candidate pool even if SIC
  discovery misses them; exclude tickers are removed from both SIC-discovered
  and analyst-specified candidates. Must-include tickers still must pass data
  quality and low-confidence filters, and the report notes any that are
  excluded.
- `analyst_approved_tickers`: optional audit-only approval list. These tickers
  do not bypass data quality, ranking, tiering, or caveat logic; when an
  approved ticker appears in the selected comp set, the report records the
  analyst approval while still showing the model's tier and fit notes. If an
  approved ticker is not selected, the report notes that too.
- `sic_clusters`: optional `{sic_code: label}` map attached to each fetched
  company as `industry_cluster` — a human-readable grouping used only for
  diagnostics/segmentation, not read by the comp ranking.
- `allow_broad_sic_codes`: escape hatch for the SIC preflight. Before any
  per-company SEC lookups, every configured SIC code is checked for filer
  count: codes yielding no usable tickers always abort the run, and codes
  matching more than 500 SEC filers abort unless this is set to `true` —
  an over-broad code mostly fetches companies unrelated to the target and
  burns the SEC request budget.

### `llm`

- `extraction_model` / `judge_model`: OpenAI models for business-model
  extraction and judging (e.g. `gpt-4.1` / `gpt-4.1-mini`).
- `embedding_model`: OpenAI embedding model for the sub-sector mismatch
  penalty (e.g. `text-embedding-3-small`) — see `scorer.ranking_penalties`
  below.
- `temperature`, `max_tokens`: passed to every OpenAI call.
- `batch_size`: how many companies to process between checkpoint saves.
- `judge_threshold`: judge scores below this set `low_confidence_flag=True`,
  which hard-excludes the company from the final Top-N (see Reporter logic).
  `low_confidence_flag` is also set independently of the judge score if the
  extraction's `evidence_quote` can't be verified or core business-profile
  fields are missing — see the README's LLM Analyzer stage description.

### `output`

- `top_n_comps`: Top-N size for the report.
- `report_formats`: which report formats to produce (`csv`, `html`, or both).
- `min_comps_warning`: when the eligible comp pool (after the
  low-confidence filter) is smaller than this (default 15), the HTML report
  gets a sample-size warning banner and the CLI/UI status echoes it —
  distance rankings and multiple distributions computed on a handful of
  comps are directional at best. Soft warning only; the run still completes.
- `prepared_by` / `confidential`: report footer attribution and
  confidentiality banner.

### `valuation`

- `size_marketability_discount` / `discount_note`: net private-company haircut
  applied to the comp-derived implied EV range, with the free-text rationale
  shown next to it. `0.0` disables the adjusted range.
- `include_operating_leases_in_ev`: capitalize ASC 842 operating-lease
  liabilities into enterprise value (off by default to keep EV/EBITDA
  consistent with EBITDA being net of lease cost).

### `scorer`

- `feature_weights`: `{business_model: {financial_feature: weight}}` — lets
  different industries weight the 6 financial features differently in the
  distance calculation (e.g. growth matters more for SaaS, EBITDA margin
  matters more for asset-heavy manufacturing) instead of treating all 6 as
  equally informative regardless of target. Looked up by the **target's**
  `business_model` so every candidate is measured against the same ruler;
  features missing from a template default to weight `1.0`; an unmatched
  or null `business_model` falls back to the `default` template (or all
  `1.0` if there's no `default` template either). Leave this empty
  (the default) to use unweighted Euclidean distance.
- `ranking_penalties`: magnitudes for the soft penalties the report's Top-N
  selection adds to a candidate's financial distance —
  `business_model_penalty`, `customer_type_penalty`,
  `subsector_similarity_threshold` / `subsector_mismatch_penalty` (a graded
  ramp: the penalty grows linearly as the embedding similarity falls below
  the threshold and reaches `subsector_mismatch_penalty` only at similarity
  0 — see `report_selection._subsector_mismatch_penalty`),
  `size_penalty_free_log10_range` / `size_penalty_per_extra_log10` (see
  `report_selection._size_mismatch_penalty`). Independently of the ranking,
  any nonzero end-market shortfall or a revenue gap beyond 10x
  (`reporter.CORE_MAX_LOG10_REVENUE_GAP`) caps a comp at the Secondary
  tier. `eval/evaluator.py`'s
  `_select_top_k` reads this same config block (passed through
  `run_evaluation()`) and shares the ranking core in
  `src/report_selection.py`, so the evaluation harness can't drift out of
  sync with what the production report actually does. See
  `src/config_schema.py`'s `RankingPenaltiesConfig` for defaults and the
  reasoning behind each one.
- `llm_rerank`: optional listwise LLM re-ranker. It is off by default
  (`enabled: false`). When enabled, the deterministic ranking is still computed
  first; the LLM can only reorder the first `rerank_window` tickers, and an
  invalid response (missing, duplicate, or extra ticker) is discarded in favor
  of the deterministic order.
