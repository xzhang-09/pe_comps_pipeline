# Known Limitations and Roadmap

This page is the source of truth for the manual-deals evaluation — what it
measures, current results, caveats, and why the remaining roadmap ideas are
sequenced the way they are. (README only summarizes the headline number and
points here — update results in one place, not two.) The short version:
every mechanical fix we shipped worked as designed, precision moved only
when the *search space* changed (suggest-SIC), and the current binding
constraint is semantic ranking quality — not corpus plumbing.

The ground-truth benchmark has gone through three revisions, referenced
below as **v1**/**v2**/**v3** rather than by date:

- **v1** — 9 deals, no discovery ladder, no coverage waterfall. Full
  write-up archived in [`eval_coverage_analysis.md`](eval_coverage_analysis.md).
- **v2** — 10 deals, all industrial/manufacturing targets; introduced the
  discovery ladder (single-sic / sic+embedding / suggest-sic /
  suggest-sic+embedding) and the coverage waterfall.
- **v3** (current) — 16 deals across 6 industry clusters (adds IT services,
  SaaS, environmental services, consumer beverages, healthcare services);
  introduced a dev(10)/holdout(6) split
  (`scripts.check_eval_regression.DEFAULT_HOLDOUT`) — holdout deals must
  never drive tuning decisions.

Only the `single-sic` discovery mode has been re-measured on v3 so far; the
other three modes below are still v2 results, pending re-confirmation — see
"Pending" below for status.

## What this evaluates

- `eval/evaluator.py` is the Precision@K harness. Its Top-K selection shares
  the production ranking core (`src/report_selection.py`), so the
  evaluation cannot drift from what the report actually does.
- `eval/ground_truth/manual_deals.json` holds the reviewed "Selected
  Companies Analysis" observations (filing URLs, advisors, target
  financials, still-public flags for banker-selected comps) —
  `manual_deals.example.json` is the template for adding more.
- `eval/ground_truth_builder.py` is experimental: it prefills
  `eval/ground_truth/manual_deals.review.json` from merger-proxy filings; a
  human must verify every field before promoting entries into
  `manual_deals.json`.
- The generated report itself uses run diagnostics and LLM-assisted
  comp-fit review as directional quality signals, not audited ground truth
  — this benchmark is the audited check.
- Report-quality fixes validated against this benchmark (not just
  spot-checked on the demo target): the `low_confidence_flag` /
  `profile_incomplete` split and targeted follow-up (`src/llm_analyzer.py`)
  stopped small-cap comps with one terse field from being silently dropped;
  the end-market LLM review (`src/end_market_reviewer.py`) corrects
  Core-tier false positives/negatives that embedding-similarity alone
  missed; the extraction judge's rubric rewrite replaced adjective-only
  scoring (which had collapsed to near-universal 4-5s) with
  behaviorally-anchored score bands and worked examples.
- Regressions are caught by a baseline gate
  (`scripts/check_eval_regression.py`): it hard-fails on mean-precision
  drops beyond a tolerance (default 1pp — market data drifts daily), lost
  waterfall hits, any increase in `low_confidence_filtered`, and any deal
  losing all its hits. Baseline updates
  (`check_eval_regression.py --update-baseline`) are a deliberate,
  reviewable act, not a side effect of running the eval.
- Per-rung results and coverage waterfalls are published in
  `eval/results.md` and its mode-suffixed siblings.

To run it yourself:

```bash
python -m scripts.evaluate_manual_deals      # writes results.json
python -m scripts.check_eval_regression      # non-zero exit on regression
```

## Discovery ladder: measured results

### Confirmed on v3 (16 deals)

| Discovery mode | Mean P@15 | Discovery-layer losses | Status |
| --- | ---: | ---: | --- |
| `single-sic` | 9.3% (dev 9.4% / holdout 9.0% — no gap) | 84 | baseline, config default |

`single-sic` is the only mode re-measured on v3
(`eval/baselines/manual_deals.single-sic.baseline.json`).

### Prior measurement, v2 (10 deals) — pending re-confirmation on v3

| Discovery mode | Mean P@15 | Median | Reachable precision | Discovery-layer losses | Status when measured |
| --- | ---: | ---: | ---: | ---: | --- |
| `sic+embedding` | 9.8% | 0% | 100% | 54 | optional |
| `suggest-sic` | 19.3% | 17.7% | 85% | 43 | recommended |
| `suggest-sic+embedding` | 15.3% | 7.0% | 78.6% | 42 | experimental |

These numbers predate v3 — do not compare them directly against the
confirmed `single-sic` row above; re-run each mode on v3 first (see
"Pending"). `suggest-sic+embedding` (hybrid) did not earn the recommended
slot on v2: it wins on paper coverage (42 vs 43 discovery losses — noise at
this sample size) but loses 6 of 10 deals head-to-head against
`suggest-sic`, because the larger candidate pool dilutes ranking (banker
comps pushed out of Top-15) and feeds more marginal candidates into the
low-confidence filter. This conclusion has not yet been checked against v3.

## Pending: discovery-ladder re-measurement on v3

`sic+embedding`, `suggest-sic`, and `suggest-sic+embedding` need to be
re-run against the full v3 (16-deal) benchmark before their rows above (or
any conclusion drawn from them, including the hybrid-mode negative result)
can be trusted at the current benchmark size. Status: **blocked**, not
merely unscheduled — a re-measurement attempt at `sic+embedding` ran for
several minutes (through the first deal's candidate fetch) before FMP's
free-tier daily quota started rejecting requests with HTTP 429; the run was
stopped rather than let finish with degraded/missing market-cap data that
would misreport as discovery-layer losses.
`scripts.evaluate_manual_deals.evaluate_manual_deals` only writes
`results.json` once, after its full deal loop completes, so the interrupted
attempt left zero usable output despite the elapsed run time.

Resume path once FMP quota resets: batch the run with `--deals` (e.g. 3-4
deals per invocation, each to its own `--output-json` path) and combine the
batches with `scripts/merge_manual_deal_batches.py` — verified byte-for-byte
equivalent to a single unbroken run by round-tripping the committed
single-sic results through a two-batch split/merge. This bounds the cost of
a future quota interruption to one batch instead of the whole benchmark.
Once each mode is re-measured, run
`scripts/check_eval_regression.py --update-baseline` for that mode to create
its (currently missing) `eval/baselines/manual_deals.<mode>.baseline.json`,
and move its row from "prior measurement" to "confirmed" above (and in
README's summary).

## What the coverage waterfall taught us

Three rounds of instrumented fixes on v2, each verified by the same
benchmark:

1. **Correctness fixes first** — candidate-filtered Top-N (historical
   vectors could crowd out the current run), content-keyed vector
   invalidation, and per-ticker discovery-stage attribution.
   Behavior-preserving by design.
2. **Then the hybrid mode** — promoting the eval-only suggest-SIC expansion
   into a production discovery mode, layered under embedding retrieval.
   Coverage moved (cross-industry banker comps finally entered the
   universe), but precision dropped: the bottleneck shifted downstream to
   ranking.
3. **Then a sampling/budget ablation** — three points: sequential
   enumeration at budget 120, stratified round-robin at 120, stratified
   at 300.
   - Stratified sampling alone was **distribution-neutral**: it reshuffled
     *which* companies got truncated (12 → 12 losses), so budget capacity,
     not enumeration order, was the real constraint.
   - Budget 300 eliminated truncation (12 → 3) — and the recovered banker
     comps then failed to rank inside the target's semantic Top-40
     (2 → 10 losses). Precision unchanged at every point.

   Conclusion: cosine similarity over short 10-K business descriptions does
   not align with banker comparability judgment. Further knob-turning
   (top-N, thresholds) would most likely push losses one stage further
   downstream again.

A cheap-channel idea was also tested and rejected before implementation:
FMP's current `stable/stock-peers` endpoint returned **zero** of 26 banker
comps across four probed targets (it lists market-cap neighbors, not
business peers; the industry-screened v4 endpoint is legacy-walled). Ten
API calls of preflight saved the whole build.

A fourth finding, from scoping the ranking-layer tuning work (roadmap item
2, below), on v3: **penalty tuning has no signal on `single-sic` dev data.**
`_select_ranked_tickers` returns the whole ranked list unchanged whenever
the candidate pool is at or below K=15 — penalty magnitudes cannot affect
which tickers get selected if every ticker gets selected regardless.
Checked directly against the v3 `single-sic` results: **all 10 dev deals**
have `selection_trivial=True` (pool sizes 1-7 companies), so mean precision
on dev is mathematically invariant to every penalty candidate a grid search
could try — a tuning run against this data would report a "best"
configuration that is pure tie-breaking noise, not a real optimization.
Only 2 of the 16 deals have non-trivial pools (Squarespace, 47 selectable;
Smartsheet, 46) and both are holdout, so they cannot be used for selection
either. `scripts/tune_ranking_penalties.py` is written but not yet runnable
for a real answer — it needs dev-deal pools from a larger-pool discovery
mode (`suggest-sic` or an embedding channel) once those are re-measured on
v3; see "Pending" above.

## Evaluation design limitations

Read the benchmark numbers with these caveats:

1. **The ladder rungs are measured unevenly across benchmark versions.**
   Only `single-sic` has been re-measured on v3; the suggest-SIC lift and
   the hybrid negative result are still only validated on v2 — see
   "Pending" above. Per-deal variance remains large — a single deal moves
   the mean by 2-3 points.
2. **The dev/holdout split has only been checked for a gap on one rung.**
   On `single-sic`, dev and holdout means match (9.4% / 9.0%) — no
   overfitting signal at that rung. The other three rungs haven't been
   re-measured with the split in place, so whether suggest-SIC's v2 lift
   holds on holdout is still unknown.
3. **Single-advisor labels.** Banker "Selected Companies" lists are one
   advisor's judgment, not the universe of defensible comps — a reasonable
   comp outside the list counts as a miss, so precision systematically
   understates quality.
4. **Present-day data, historical decisions.** The pipeline evaluates
   historical filings using current descriptions and market data; bankers
   chose comps from filing-date information. Some hits/misses reflect
   company drift since the filing, not selection quality.
5. **Small pools degrade Precision@K.** When the scored pool is at or below
   K, "selection" picks everything that survived coverage — flagged as
   `selection_trivial` per deal, but the headline mean still mixes coverage
   and ranking failures. Use the coverage waterfall and reachable precision
   for diagnosis, not the headline number alone.

Data-contract limits (US 10-K filers only, FMP free-tier quota, 1-day
market-data TTL) are documented in [data_layer.md](data_layer.md).

## Roadmap

Ordered by expected value given the findings above:

1. ~~Expand ground truth to 15-20 deals across sectors.~~ **Done (v3)** —
   16 deals across 6 industry clusters, dev(10)/holdout(6) split
   (`eval/ground_truth/manual_deals.json`). Next: re-measure the three
   pending discovery-ladder rungs on v3 — see "Pending" above — which is a
   prerequisite for #2 below (ranking-layer tuning needs a discovery mode
   with non-trivial dev-deal pools).
2. **Ranking layer.** The newly exposed bottleneck: larger pools push banker
   comps out of Top-15 while reachable precision stays high. Candidate-pool
   conditioning or rank-aware weighting is the first untried lever.
   `scripts/tune_ranking_penalties.py` is built for this (grid search +
   dev/holdout-disciplined validation) but currently blocked — see the
   "fourth finding" in "What the coverage waterfall taught us" above.
3. **Richer semantic inputs.** Short profile descriptions hit a quality
   ceiling; full 10-K Item 1 text (products, end markets, competitors) is
   the natural upgrade before changing embedding models.
4. **10-K competitor-mention channel.** Explicitly named competitors from
   Item 1/1A — free, filing-date-consistent (no time travel), and directly
   aligned with how bankers reason. Preferred over BM25 or financial
   neighbors as the next recall channel.
5. **Vector store migration** (Chroma/FAISS/pgvector) only when corpus size
   or query latency demands it — the JSON store's filtered `upsert/query`
   interface is deliberately swappable.

Retired ideas, with evidence: FMP peers channel (no signal in current
endpoint data), corpus-budget increases beyond 120 (truncation eliminated,
precision unchanged), enumeration-order fixes as a recall lever
(distribution-neutral).
