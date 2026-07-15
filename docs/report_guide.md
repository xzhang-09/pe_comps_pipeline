# Understanding the Report

`pe-comps` (or `python -m src.pipeline`) produces the formats requested in
`output.report_formats` (`outputs/comps_report.csv`,
`outputs/comps_report.html`, or both). The HTML report has 7 sections:

1. **Target and Selection Snapshot** — target summary, run timestamp,
   selected-comp count, scoring-pool size, eligible-candidate count, data-failure
   count, and EV/EBITDA IQR compression versus the eligible pool.
2. **Top-N Comparable Companies** — ordered tier-first (Core → Secondary →
   Review/Exclude), then within each tier by financial-feature distance to the
   target after soft penalties for business-model, customer-type,
   revenue-scale, and sub-sector mismatch. A comp with a deterministic
   mismatch flag is capped at Secondary even when the qualitative review rates
   it highly, so the tier label never contradicts the fit notes beside it.
3. **Comparable Fit Review** — a directional qualitative review of the selected
   Top-N set, including strongest fits, questionable fits, and potential
   near-miss substitutions. This is not a substitute for transaction-team
   judgment, banker input, or confirmatory diligence.
4. **Valuation and Financial Benchmarks** — a P25/median/P75/mean
   distribution across the selected Top-N for EV/EBITDA, EV/Revenue, EV/EBIT,
   EV/Gross Profit, and P/E, plus the Top-N median FCF yield (cash from
   operations − capex, over EV). EV/EBIT and FCF yield are included because an
   asset-heavy target's D&A is a real economic cost that EV/EBITDA flatters,
   and FCF is the cash a PE buyer actually underwrites to. Financial benchmark
   distributions cover EBITDA margin, revenue growth, gross margin,
   capex/revenue, FCF conversion, net debt/EBITDA, interest coverage, and
   debt/equity; the leverage and FCF-conversion rows show the comp
   distribution only (no target figure, since the private target doesn't
   disclose them). Target percentile is computed empirically within the
   selected comps, not by interpolating from rounded quartiles.
5. **Near-Miss Candidates** — the `report_selection.AUDIT_SIZE` (5 by default)
   candidates just outside the Top-N cutoff, sorted by financial-fit rank, with
   the specific reason each one didn't make it (business-model/customer-type/
   revenue-scale/sub-sector penalty, or simply ranked below on financial
   distance alone).
6. **Selection Diagnostics** — average distance to target across the selected
   Top-N, a multiple-spread check comparing the selected Top-N's EV/EBITDA IQR
   with the eligible pool's IQR, and the financial features that most influenced
   the ranking. A good comp set should converge on a usable multiple — a low
   ratio means the selection is doing real work narrowing the spread; a ratio
   near/above 100% means the Top-N is about as scattered as the pool it was
   drawn from (worth revisiting `scorer.feature_weights` or the soft-penalty
   constants in that case, not a hard pass/fail threshold — see
   `report_valuation._relative_dispersion`).
7. **Data Notes** — how many companies were excluded for weak source support,
   how many companies lacked required market or filing data, the LLM assistance
   note, the external benchmarking caveat, and a standard disclaimer.
8. **Methodology Notes** — how to read the report's derived numbers: that Rank
   (Section 2) is the standardized financial distance plus the soft mismatch
   penalties rather than the EV/EBITDA multiple; that the fit review (Section 3)
   is directional, not a substitute for transaction-team judgment; that the
   target percentile (Section 4) is computed from company-level data and so may
   not match interpolation off the rounded quartiles; that "relative influence"
   (Section 6) values are only comparable to each other within that table; and
   an end-market-scarcity note on why niche targets lean on broader
   same-industry comps.

The CSV has one row per selected comp with the same financial/business-model
fields shown in the report, for further analysis in Excel or elsewhere.

A committed sample (synthetic target) is at
[`samples/sample_report.html`](samples/sample_report.html) /
[`samples/sample_report.csv`](samples/sample_report.csv).
