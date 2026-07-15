# Data Layer: Coverage, Costs, and Upgrade Paths

The pipeline's default data stack is free-tier: SEC EDGAR XBRL for
fundamentals plus FMP's `/profile` endpoint for market cap/sector. This page
documents what that stack can and cannot cover, what a run costs, and the
designed-in upgrade paths to paid data sources.

## EBITDA coverage on the free-tier stack

Roughly 54% of SIC-based candidates end up with a usable `ev_ebitda` label in
a typical run. Four distinct causes account for most of the missing coverage:

1. *Legitimately negative EBITDA* (~30% of candidates). Pre-revenue or
   turnaround companies — common in adjacent SIC buckets used to widen the
   pool — have negative operating income. EV/EBITDA is not a meaningful
   multiple for these companies; they are correctly excluded from scoring
   but still appear in the raw candidate count.

2. *Operating income not tagged as `OperatingIncomeLoss` in XBRL* (~10%
   of candidates). Large diversified industrials and automotive OEMs —
   including companies like Deere, Parker Hannifin, Johnson Controls,
   Baker Hughes, and most major Tier-1 auto suppliers — structure their
   income statements so that the EBIT subtotal is absorbed into or
   presented below `PretaxIncomeLoss` without a separate XBRL tag for
   operating income. `edgartools` normalises XBRL filings to a fixed
   concept vocabulary; when a company omits the `OperatingIncomeLoss`
   subtotal, `_ebitda()` in `fetcher.py` returns `None` and no EV/EBITDA
   multiple can be derived. A fallback approximation
   (`PretaxIncomeLoss + InterestExpense`) is intentionally *not*
   implemented: for companies with material financial-services subsidiaries
   or large non-operating items the gap from true operating income can run
   to billions, and mixing two different EBIT definitions in the same comp
   table would corrupt the multiple distribution without any visible
   warning. The right fix is a data source that normalises this
   consistently — see the upgrade paths below.

3. *D&A not reported as a distinct cash-flow line item* (~3–5% of
   candidates). Some large manufacturers (e.g. BorgWarner) file a
   condensed indirect-method cash-flow statement in XBRL that shows only
   the section subtotals. `edgartools` cannot extract a standalone D&A
   figure from these filings, so EBITDA cannot be computed even when
   operating income is present. This is a filing-presentation choice, not
   an edgartools bug, and is not reliably fixable from EDGAR alone.

4. *GAAP EBITDA vs. the adjusted EBITDA used in comp analysis.* The
   pipeline computes GAAP EBITDA (`OperatingIncomeLoss` + D&A). In a year
   with large one-time charges this diverges sharply from the adjusted
   EBITDA that bankers and analysts quote: Ducommun's FY2025 GAAP
   operating income is -$32mm on a $107mm litigation settlement, Rogers'
   is -$45mm on $97mm of restructuring/impairment charges — both healthy
   businesses on an adjusted basis, both excluded by the EBITDA-margin
   filter and unusable for an EV/EBITDA multiple in that year. Reliable
   add-backs are not derivable from EDGAR XBRL (one-off lines often carry
   no normalized concept), and mixing GAAP and adjusted definitions
   per-company would corrupt the multiple distribution, so this is
   deliberately not patched — it is another instance of the paid-data-layer
   boundary described below.

Outliers (EV/EBITDA > 100× or negative) are automatically set to `None`
rather than included; these are typically loss-making companies whose
multiple would be arithmetically valid but economically meaningless.

## Using a paid FMP plan

This pipeline only calls FMP's `/profile` endpoint (`src/fmp_client.py`),
which is available on FMP's free tier for every symbol. FMP's
`/key-metrics` and `/enterprise-values` endpoints return direct EV/EBITDA
and EV/Revenue multiples without needing to derive them from EDGAR
fundamentals, but return HTTP 402 on the free tier for anything beyond a
handful of demo mega-cap tickers — so this pipeline doesn't call them, and
derives EV/EBITDA/EV/Revenue from `/profile`'s market cap plus SEC EDGAR
XBRL fundamentals instead (see `fetcher._enrich_with_fmp_data`). If you have
a paid FMP plan with access to those endpoints, you can get more direct,
likely more complete multiples by adding calls to them in
`src/fmp_client.py` and wiring the response into `record["ev_ebitda"]` /
`record["ev_revenue"]` / `record["enterprise_value_usd_mm"]` in
`fetcher._enrich_with_fmp_data`.

## Upgrading the data layer

The free-tier stack is sufficient for a first pass but has the real coverage
limits described above. The fetcher is designed so data-source upgrades
are localised: `_fetch_single()` handles EDGAR fundamentals and
`_enrich_with_fmp_data()` handles market data; swapping either one out
doesn't require touching LLM extraction, scoring, or reporting.

**Paid FMP (Starter / Professional plan)**
FMP's `/key-metrics` and `/enterprise-values` endpoints return
provider-normalised EV/EBITDA and EV/Revenue directly, bypassing the
EDGAR-derivation chain entirely. This resolves the operating-income and
D&A tagging gaps: FMP normalises EBITDA consistently across filers,
including the large-cap industrials and automotive OEMs that EDGAR XBRL
cannot cover with the current logic. To wire this in, add calls to those
endpoints in `src/fmp_client.py` and write the result directly into
`record["ev_ebitda"]`, `record["ev_ebitda_source_value"]`, and
`record["enterprise_value_usd_mm"]` in `_enrich_with_fmp_data()` —
the rest of the pipeline reads those fields without caring how they were
populated.

**CapIQ / Refinitiv / Bloomberg**
Terminal-grade providers normalise EBITDA, EBIT, and all EV bridge
components consistently across global filers and fiscal-year conventions.
The cleanest integration point is to write a `_fetch_from_premium_source()`
function that mirrors the output schema of `_fetch_single()` (see the
`dict` it returns), then call it from `pipeline.py` before the EDGAR fetch
so its results take priority for any ticker the premium source covers.
Fields not supplied by the premium source fall back to EDGAR as before.

**Coverage expectations by data source**

| Source | ev_ebitda coverage (typical run) | Notes |
|---|---|---|
| EDGAR XBRL + FMP free | ~54% | Current default |
| EDGAR XBRL + FMP paid key-metrics | ~75–80% | Adds normalised multiples for the XBRL-gap companies |
| CapIQ / Bloomberg | ~90%+ | Handles non-standard filers, foreign companies, fiscal-year edge cases |

## Cost estimate

- SEC EDGAR: free.
- FMP free tier: free, but capped at a daily request quota (paid plans lift
  this, and also unlock the `/key-metrics`/`/enterprise-values` endpoints —
  see "Using a paid FMP plan" above). Set `FMP_DAILY_PROFILE_BUDGET` to make
  the quota an explicit, controlled degradation — see
  [configuration.md](configuration.md#environment-variables).
- OpenAI API for ~150 companies (extraction + judge + target): measured at
  roughly $0.40-0.80 per full run during development with `gpt-4.1` /
  `gpt-4.1-mini`. The sub-sector mismatch penalty adds one batched
  `text-embedding-3-small` call per report (target + every eligible
  candidate's `sub_sector_description`, a one-sentence string each) — at
  most a few cents even for a large candidate pool, not per-company.
- Optional semantic discovery (`universe.discovery_mode: sic+embedding`) uses
  `text-embedding-3-small` over cached or freshly fetched business descriptions
  for companies in the same 2-digit SIC families as the target's configured
  SIC codes. The embedding vectors are stored under `data/cache/` through
  `src/embedding_store.py`, separate from fetcher's financial cache so a
  description-only corpus entry cannot masquerade as a fresh fundamentals
  record.
- Cached runs (LLM checkpoint exists, FMP/EDGAR data cached): low incremental
  cost — only the target's own LLM extraction re-runs; comp scoring itself
  (financial-feature distance) is cheap to recompute every run and isn't cached.
