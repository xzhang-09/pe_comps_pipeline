# Known Limitations and Roadmap

This page records what the evaluation can and cannot support, the measured
results of each discovery upgrade, and why the remaining ideas are sequenced
the way they are. The short version: every mechanical fix we shipped worked
as designed, precision moved only when the *search space* changed
(suggest-SIC), and the current binding constraint is semantic ranking
quality — not corpus plumbing.

## Discovery ladder: measured results

All rungs run against the same 10-deal fairness-opinion benchmark
(see [Evaluation](../README.md#evaluation) and `eval/results*.md`).

| Discovery mode | Mean P@15 | Median | Reachable precision | Discovery-layer losses | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `single-sic` | 8.2% | 0% | 100% | 58 | baseline |
| `sic+embedding` | 9.8% | 0% | 100% | 54 | optional |
| `suggest-sic` | 19.3% | 17.7% | 85% | 43 | **recommended** |
| `suggest-sic+embedding` | 15.3% | 7.0% | 78.6% | 42 | experimental |

`suggest-sic+embedding` (hybrid) did not meet its acceptance gate: it wins
on paper coverage (42 vs 43 discovery losses — noise at this sample size)
but loses 6 of 10 deals head-to-head against `suggest-sic`, because the
larger candidate pool dilutes ranking (banker comps pushed out of Top-15)
and feeds more marginal candidates into the low-confidence filter.

## What the coverage waterfall taught us

Three rounds of instrumented fixes, each verified by the same benchmark:

1. **Correctness fixes (P0)** — candidate-filtered Top-N (historical vectors
   could crowd out the current run), content-keyed vector invalidation, and
   per-ticker discovery-stage attribution. Behavior-preserving by design.
2. **Hybrid mode (P1)** — promoting the eval-only suggest-SIC expansion into
   a production discovery mode, layered under embedding retrieval. Coverage
   moved (cross-industry banker comps finally entered the universe), but
   precision dropped: the bottleneck shifted downstream to ranking.
3. **Sampling/budget ablation (P2)** — three points: sequential enumeration
   at budget 120, stratified round-robin at 120, stratified at 300.
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

## Evaluation design limitations

Read the benchmark numbers with these caveats:

1. **Ten deals, one sector cluster.** All ten targets are
   industrial/manufacturing names. Conclusions (including the suggest-SIC
   lift) are unvalidated for software, healthcare, or services targets, and
   per-deal variance is large — a single deal moves the mean by 2–3 points.
2. **No holdout.** The same ten deals drove development iterations
   (guardrail calibration, revenue-band choice, fit-quality labels), so
   published numbers are development-set performance. Any further parameter
   tuning needs a larger ground truth first.
3. **Single-advisor labels.** Banker "Selected Companies" lists are one
   advisor's judgment, not the universe of defensible comps — a reasonable
   comp outside the list counts as a miss, so precision systematically
   understates quality.
4. **Present-day data, historical decisions.** The pipeline evaluates
   2023–24 filings using current descriptions and market data; bankers chose
   comps from filing-date information. Some hits/misses reflect three years
   of company drift, not selection quality.
5. **Small pools degrade Precision@K.** When the scored pool is at or below
   K, "selection" picks everything that survived coverage — flagged as
   `selection_trivial` per deal, but the headline mean still mixes coverage
   and ranking failures. Use the coverage waterfall and reachable precision
   for diagnosis, not the headline number alone.

Data-contract limits (US 10-K filers only, FMP free-tier quota, 1-day
market-data TTL) are documented in [data_layer.md](data_layer.md).

## Roadmap

Ordered by expected value given the findings above:

1. **Expand ground truth to 15–20 deals across sectors.** Prerequisite for
   everything below — the current benchmark cannot reliably measure the
   marginal contribution of any further change.
2. **Ranking layer.** The newly exposed bottleneck: larger pools push banker
   comps out of Top-15 while reachable precision stays high. Candidate-pool
   conditioning or rank-aware weighting is the first untried lever.
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
