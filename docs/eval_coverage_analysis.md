# Manual Eval Coverage Analysis

> **Historical note (superseded).** This records the *first* manual-deals
> benchmark iteration — a 9-deal set, 2026-07-03/05 — before two later
> rounds of expansion (10 deals with the discovery ladder and coverage
> waterfall; then 16 deals with a dev/holdout split). Every number below is
> superseded; none of it reflects the current benchmark. For current
> methodology, numbers, and open items, see
> [`known_limitations_and_roadmap.md`](known_limitations_and_roadmap.md).
> Kept for its still-relevant diagnostic framing (discovery-vs-ranking
> attribution) and the GAAP-vs-adjusted-EBITDA example (Ducommun/Rogers)
> near the end, which isn't duplicated elsewhere.

This note captures why the first full manual-deals evaluation produced
`0.0%` Precision@15 and what to improve next. It is based on the run written to
`eval/results.md` and `outputs/eval/manual_deals/results.json` on 2026-07-03.

## Current Result

- Benchmark: 9 reviewed fairness-opinion deals from
  `eval/ground_truth/manual_deals.json`.
- Metric: Precision@15.
- Mean Precision@15: `0.0%`.
- Median Precision@15: `0.0%`.
- Hits: `0`.
- Ranking misses: `0`.
- Pre-scoring misses: all eligible banker-selected comps.

The result is therefore not primarily a ranking failure. The ground-truth
companies are being lost before they reach `company_scores`.

## Evidence Breakdown

Across the 9 deals, the evaluator found 37 still-public banker-selected comps
eligible for the denominator. Their pre-scoring outcomes were:

| Reason | Count | Tickers |
| --- | ---: | --- |
| Not in target SIC universe | 18 | `6395.T`, `6723.T`, `AHT.L`, `CGCBV.HE`, `DDD`, `HRI`, `IFX.DE`, `KN`, `KRNT`, `MTW`, `NOD.OL`, `PAL.VI`, `PWG.PA`, `QCOM`, `SSYS`, `ST`, `TEX`, `URI` |
| Revenue above `max_revenue_usd_mm: 800` | 17 | `ADI`, `AMD`, `AVGO`, `CRUS`, `INTC`, `MCHP`, `MRVL`, `MTSI`, `NVDA`, `NXPI`, `ON`, `QRVO`, `SMTC`, `STM`, `SWKS`, `SYNA`, `TXN` |
| EBITDA margin below `min_ebitda_margin: 5%` | 1 | `SLAB` |
| No valid EV/EBITDA label | 1 | `LSCC` |

Observed final scored pools were also very small:

- SIC `3559` deals generally selected only `ERII`, `VECO`, `CRCT`, and `AZTA`.
- SIC `3674` deals generally selected only `SHLS` and `SKYT`.

That makes Precision@15 mostly a coverage diagnostic, not a useful ranking
diagnostic yet.

## Core Problems

### 1. Production Hard Filters Are Too Strict for Banker Benchmarks

The current config has:

- `min_revenue_usd_mm: 30`
- `max_revenue_usd_mm: 800`
- `min_ebitda_margin: 0.05`

These filters fit the current demo/private-company use case, but banker
fairness opinions frequently include larger public category leaders. The
semiconductor deals show this clearly: `NVDA`, `TXN`, `ADI`, `AVGO`, `NXPI`,
`MCHP`, and similar companies enter SIC `3674`, but are removed by the revenue
cap before scoring.

### 2. Single Target SIC Discovery Is Too Narrow

The eval runner currently uses each deal's `target_sic` as the only primary SIC
and clears adjacent SIC codes. This avoids answer leakage, but it is too narrow
for many fairness-opinion comp sets.

Examples:

- `3559` deals include 3D-printing, equipment-rental, crane, and industrial
  technology comps outside the target SIC.
- Some banker-selected comps are strategic or end-market peers rather than
  exact SIC peers.

### 3. Current Data Layer Is US/EDGAR-Centric

Many banker comps are non-US public companies, including tickers with suffixes
such as `.HE`, `.VI`, `.T`, `.L`, `.DE`, `.PA`, and `.OL`.

The current pipeline relies on EDGAR 10-K data and filters non-US domiciles
when FMP reports a country. Those companies are legitimate banker comps, but
they are outside the current data contract.

### 4. EV/EBITDA Label Availability Is a Hard Gate

Even when a ticker is discovered and fetched, it must survive the feature
builder's EV/EBITDA label requirement to enter `company_scores`. This can drop
companies with missing XBRL fields, outlier multiples, or unavailable EBITDA.

This is economically defensible for valuation ranking, but it reduces recall in
the benchmark denominator unless reported explicitly.

## Improvement Plan

### 1. Add a Coverage Waterfall to the Eval Output

Before changing ranking logic, the manual eval should explain where every
ground-truth ticker is lost.

Add per-ticker stages to the runner output:

- `not_in_sic_universe`
- `fetch_failed_or_no_cache`
- `market_cap_filtered`
- `revenue_filtered`
- `ebitda_margin_filtered`
- `non_us_filtered`
- `no_valid_ev_ebitda`
- `low_confidence_filtered`
- `ranked_but_not_top_k`
- `hit`

Then summarize the counts in `eval/results.md`. This turns a single `0.0%`
number into a useful discovery-vs-ranking diagnosis.

### 2. Add Strict vs Benchmark Eval Profiles

Run two profiles instead of treating the production demo config as the only
benchmark configuration.

`strict` profile:

- Uses `config.yaml` filters as-is.
- Measures current production/demo behavior.

`benchmark` profile:

- Keeps target SIC discovery and broad-SIC preflight override.
- Relaxes or disables `max_revenue_usd_mm`.
- Relaxes or disables `min_ebitda_margin`.
- Keeps US/EDGAR constraints unless a non-US data source is added.

This separates "the demo config intentionally filters out large companies" from
"the ranking method cannot recover banker comps even when they enter the pool."

### 3. Add Non-Leaking Discovery Expansion

Do not use banker-selected answer tickers or their SIC codes to construct the
universe; that would leak the answer.

Acceptable expansion sources:

- SEC-validated `suggest_sic_codes()` output from the target business
  description.
- Analyst-provided seed tickers that are independent of the banker list.
- Manually reviewed adjacent SIC codes documented per deal, sourced from target
  business analysis rather than answer comps.

The goal is to let mixed or adjacent business models enter discovery without
teaching the runner the answer set.

### 4. Tune Ranking Only After Coverage Exists

Penalty tuning is not useful while every eligible comp is missing before
ranking. Tuning should start only after a meaningful number of ground-truth
tickers appear in `company_scores`.

Once coverage exists, split misses into:

- Discovery misses: never reached `company_scores`.
- Ranking misses: reached `company_scores` but missed Top-K.

Only the second category is a ranking/penalty problem.

## Recommended Implementation Order

1. **Coverage waterfall**
   Add stage attribution to `scripts/evaluate_manual_deals.py`, write the
   summary into JSON and `eval/results.md`, and test with a mocked runner.

2. **Strict vs benchmark profiles**
   Add `--profile strict|benchmark` to the runner. Re-run both profiles and
   compare the coverage waterfall and Precision@15.

3. **Discovery expansion experiment**
   Add an optional non-leaking adjacent-SIC source, preferably using
   SEC-validated `suggest_sic_codes()` or manually reviewed target-derived SICs.

4. **Ranking/tuner work**
   Only tune penalties after the benchmark profile shows ground-truth tickers
   surviving into `company_scores`.

5. **Data-source expansion**
   Treat non-US comps as a separate larger project. They require a non-EDGAR
   financial data source or a different benchmark denominator.


## Re-test Results (2026-07-05)

Everything above describes the first benchmark run (9 deals, mean Precision@15 `0.0%`).
This section closes the loop after the fixes.

### What changed between the runs

1. **Benchmark re-curated** — the out-of-domain deals (large-cap semiconductors,
   2008-vintage WJ with a mislabeled EDGAR SIC) were retired and later removed
   from the repository; the benchmark is now 10 audited 2023-2024
   mid-market industrial deals with per-comp `us_filer` flags and per-deal
   audit notes (`eval/ground_truth/manual_deals.json`).
2. **Denominator fixed** — delisted and non-US-filer banker comps are excluded
   from the Precision denominator (they are unreachable under the documented
   US/EDGAR data contract) but kept in the audit trail.
3. **Runner config fixed** — the revenue band is now relative to each deal's
   target (0.1x-10x, replacing the demo config's absolute $30-800mm band that
   deleted large targets' comps wholesale), and the empty adjacent bucket no
   longer silently burns half the candidate quota
   (`primary_allocation_pct: 1.0`).
4. **Coverage waterfall implemented** (`eval/coverage.py`) — every eligible
   comp is attributed to a loss stage using the production filter predicates;
   `reachable precision` isolates the ranking layer. Includes the two stages
   recommended above plus `truncated_by_max_candidates` and
   `missing_market_cap` (FMP quota misses vs. genuine data gaps).
5. **Discovery ladder implemented** — `--discovery single-sic` (deterministic
   lower bound) vs `--discovery suggest-sic` (SEC-validated LLM-suggested
   codes from the business description only; leakage-free approximation of
   real analyst usage). Results land side by side (`eval/results.md`,
   `eval/results.suggest-sic.md`).

### Results

| Metric | single-sic (baseline) | suggest-sic (rung 2) |
| --- | ---: | ---: |
| Mean Precision@15 | 8.2% | **19.3%** |
| Hits (of 71 eligible) | 5 | 12 |
| `not_in_sic_universe` | 58 (82%) | 43 (61%) |
| `financial_filtered` | 5 | 9 |
| `no_valid_ev_ebitda` | 2 | 2 |
| `low_confidence_filtered` | 1 | 2 |
| `ranked_but_not_top_k` | 0 | 3 |
| Reachable (ranking-layer) precision | 100% (all pools trivial — untested) | **85% (first genuine test)** |
| `missing_market_cap` / `fetch_failed` / truncation | 0 | 0 |

### Conclusions

1. **Discovery is the binding constraint, now quantified.** Even with
   LLM-suggested codes, 61% of banker comps are not reachable through SIC
   taxonomy at all — the hard number behind Known Limitation #1 and the
   quantitative case for the seed-ticker discovery path (roadmap item).
2. **The +11.1pp delta between rungs is the measured value of the
   suggest-SIC flow** — the ladder design (deterministic baseline as the
   ablation control) is what makes that attribution possible.
3. **The ranking layer got its first real test only at rung 2**: 12 hits vs
   3 ranked-out (ITT, FLS for CIRCOR; MTW for Manitex). Penalty tuning was
   correctly deferred — before rung 2 there was nothing to tune against;
   with 3 negative examples it is still directional at best.

   Component decomposition of the three misses (recomputed from the recorded
   discovery codes, cache-deterministic):

   - MTW rank 16 (cutoff 15, missed by 0.04): distance 2.52 of total 2.66 —
     no categorical penalties, subsector similarity 0.83. Lost on raw
     financial distance alone; the #15 slot went to Azenta (life-sciences
     automation) and the Manitex top-15 contains ACMR/ACLS/VECO
     (semiconductor equipment) and HLLY (B2C aftermarket) — financially
     closer shapes, obviously worse businesses.
   - ITT rank 21: distance 2.28 + subsector penalty 0.4 (similarity 0.564,
     just under the 0.6 threshold — a diversified-industrial description
     problem). Even penalty-free it would sit outside the top 15.
   - FLS rank 23: distance 2.74, essentially unpenalized (similarity 0.72) —
     the single most obvious flow-control comp for CIRCOR, ranked out purely
     by financial-feature distance ($4.7bn revenue profile vs $920mm target).

   Diagnosis: within-pool ordering is dominated by financial distance; the
   qualitative penalties (0.4-0.6) are small against the observed residual
   spread (~0.7-2.7 in these pools — wider than the ~0.3-2.0 the config
   comments assumed), so business-mismatched but financially-similar names
   outrank true comps. This is the Case A lesson resurfacing one level up:
   tier-first display fixed the *label* contradiction, but Top-N
   *membership* is still distance-first. Decision: do NOT grid-tune on 3
   negatives (guaranteed overfit; and see conclusion 5 on the size-bias
   trap); revisit penalty scale — or a tier-first membership rule — once the
   benchmark has more ranked-out examples (more deals, or the seed-ticker
   rung which will produce larger pools).
4. **GAAP-vs-adjusted EBITDA divergence surfaced (initially misdiagnosed as
   an XBRL artifact; corrected after inspecting the filings)**: Ducommun's
   FY2025 GAAP operating income is -$32mm because of a one-time $107mm
   litigation settlement; Rogers' is -$45mm from $97mm of
   restructuring/impairment charges. The pipeline's GAAP EBITDA
   (OperatingIncomeLoss + D&A) is arithmetically correct; banker comp
   analysis uses *adjusted* EBITDA that excludes such one-offs, so a
   one-time-charge year pushes a real comp under the margin floor and out
   of the multiple set. Reliable add-backs are not derivable from EDGAR
   XBRL (DCO's litigation line carries no normalized concept), and mixing
   EBITDA definitions per-company would corrupt the comp table — this is
   a data-layer boundary, now with two concrete examples. No code change.
5. **Filter-philosophy divergence is now measurable, not anecdotal**: the
   10x revenue band excluded TDG/SWK/SNA/TEX/ATI — banker-endorsed mega-cap
   anchors that the product's mid-market size discipline deliberately
   rejects. Any penalty/filter tuning against this benchmark must account
   for that bias (do not let the tuner learn to delete the size discipline).
