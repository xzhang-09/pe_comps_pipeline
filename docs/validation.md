# Validation Notes

Two synthetic (non-confidential) target cases were run end-to-end through the
web UI in July 2026 to validate the workflow: LLM SIC suggestion, manual SIC
review, pipeline execution, and report usefulness. This page summarizes what
they showed, what was fixed as a result, and what remains open. Sample output
from the re-run of Case A is in [`samples/`](samples/).

## Case A — Precision Motion Components Co. (pass)

A $180mm-revenue B2B manufacturer of precision motion components for
automotive OEM and industrial end markets (primary SIC 3714/3562).

- The SIC-discovered universe (58 candidates) was plausible; roughly 7 of the
  Top 10 were defensible auto/vehicle/industrial component comps.
- The report's fit notes correctly surfaced B2C/aftermarket/channel
  mismatches on the weaker comps.
- **Original issue:** the #1-ranked comp was a `Review / Exclude` company
  (strong financial-distance fit, but a B2C aftermarket business) — financial
  closeness outranked qualitative fit in the displayed order.

## Case B — Warehouse Automation Systems Co. (fail, by design of the test)

A $220mm-revenue provider of warehouse automation systems — conveyor/sortation
design, installation, controls integration, and maintenance services.

- SIC-only discovery could not find warehouse-automation system integrators:
  the surviving pool was industrial distributors and general manufacturers,
  and the fit notes correctly flagged business-model/end-market mismatch on
  nearly every selected comp.
- Two LLM-suggested SIC codes had to be removed by hand mid-validation: one
  (7373) was operationally far too broad, one (3535) yielded no usable
  tickers. Neither problem was visible before the run started.
- The final pool (9 companies with a valid EV/EBITDA) was too small for the
  distance ranking to be meaningful, and nothing in the report said so.

## Fixes made after validation

1. **Tier-first ranking** — the report now orders comps Core → Secondary →
   Review/Exclude (financial distance within each tier), so a flagged comp can
   no longer appear at rank 1. A comp with a deterministic business-model /
   customer-type / end-market mismatch can also no longer be labeled Core,
   even when the LLM review calls it a strong fit (`reporter._assign_tier`).
2. **SIC preflight** — before any per-company SEC lookups, each configured SIC
   code is checked for filer count: zero-yield codes abort the run with a
   clear message, and codes above a 500-filer threshold abort unless
   `universe.allow_broad_sic_codes: true` is set (`universe_builder`).
3. **Small-sample warning** — when the eligible comp pool falls below
   `output.min_comps_warning` (default 15), a warning banner is stamped on the
   HTML report and echoed in the CLI/UI status.
4. **Failed-run forensics** — a UI run that crashes now writes the traceback
   to `outputs/ui_runs/<run-id>/error.log` next to the config that produced
   it, and the UI status reports the failure reason.

Re-running Case A after these fixes produces the sample report in
[`samples/sample_report.html`](samples/sample_report.html): the top-ranked
comp is now a Core comp, and the flagged comps sit at the bottom of the table
with their reasons attached.

## Known remaining gap

Case B's root cause — SIC-only discovery misses hybrid
manufacturing/services/system-integration targets — is a discovery-mode
limitation, not a ranking bug. The planned fix is a second discovery path
(analyst-provided seed tickers, then description-embedding search over SEC
filers) and is tracked in the README's Known Limitations.
