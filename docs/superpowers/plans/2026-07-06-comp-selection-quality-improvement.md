# Comp Selection Quality Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve actual comp-company selection quality by making replacement decisions explicit, rejecting weak replacement candidates, and surfacing deterministic selection-quality metrics.

**Architecture:** Keep the existing financial-distance ranking and soft-penalty model. Add a conservative selection-quality layer around the existing `_fill_usable_comp_slots()` flow: diagnose Top-N quality, add only usable replacement candidates, and report replacement decisions so weak selected comps are not silently papered over.

**Tech Stack:** Python 3.10+, pytest, existing `src/report_selection.py`, `src/reporter.py`, Jinja report template.

---

## Scope

This first implementation does not rewrite candidate discovery or replace the financial scorer. It improves the already-present second-layer mechanics:

- Core eligibility remains deterministic.
- Review / Exclude rows still remain visible for audit.
- Replacement candidates are now accepted only if they are usable (`core` or `secondary`), not if they also become `review_exclude`.
- The report gains explicit selection-quality metrics and a replacement audit.

Out of scope for this pass:

- New external data sources.
- New SIC expansion logic.
- Full benchmark tuning of penalty constants.
- Replacing embeddings or LLM extraction.

## Files

- Modify `src/reporter.py`
  - Add selection quality summary helper.
  - Extend `_fill_usable_comp_slots()` to return substitution audit decisions.
  - Reject replacement candidates that are themselves `review_exclude`.
  - Pass selection quality and substitution audit into template context.

- Modify `src/templates/report.html`
  - Show a concise selection quality note near the Top comps section.
  - Show substitution audit notes when replacements were accepted or rejected.

- Modify `tests/test_reporter.py`
  - Add tests for replacement rejection.
  - Add tests for selection-quality summary.
  - Add tests that the report shows replacement audit notes.

## Acceptance Criteria

- A replacement candidate that would be `review_exclude` is not accepted as a usable-slot filler.
- Accepted replacements are recorded with ticker and tier.
- Rejected replacements are recorded with ticker and reason.
- The report shows Core / Secondary / Review-Exclude counts and max revenue ratio.
- Existing reporter tests pass.
- Lint passes for modified files.

## Task 1: Add Selection Quality Summary

- [ ] Write a failing unit test in `tests/test_reporter.py` for `_selection_quality_summary()`.
- [ ] Implement `_selection_quality_summary(rows, target_revenue)`.
- [ ] Verify the test passes.

Expected fields:

- `n_total`
- `n_core`
- `n_secondary`
- `n_review_exclude`
- `usable_count`
- `max_revenue_ratio`
- `median_revenue_ratio`

## Task 2: Add Replacement Audit and Reject Weak Replacements

- [ ] Write a failing test showing `_fill_usable_comp_slots()` rejects a replacement candidate that becomes `review_exclude`.
- [ ] Extend `_fill_usable_comp_slots()` to return `(rows, selected, flagged_tickers, substitution_audit)`.
- [ ] For each candidate considered after the initial selected list, annotate a tentative row set.
- [ ] If the candidate tier is `review_exclude`, do not append it to `selected`; record a rejected audit item.
- [ ] If the candidate tier is `core` or `secondary`, append it; record an accepted audit item.
- [ ] Update the single caller in `generate()`.
- [ ] Verify focused tests pass.

Audit item shape:

```python
{
    "ticker": "T015",
    "company_name": "T015 Inc.",
    "decision": "accepted" | "rejected",
    "tier": "core" | "secondary" | "review_exclude",
    "reason": "filled usable comp slot" | "candidate also classified as Review / Exclude",
}
```

## Task 3: Display Selection Quality and Substitution Audit

- [ ] Write a failing HTML test checking the report includes `Selection quality check`.
- [ ] Add `selection_quality` and `substitution_audit` to the template context.
- [ ] Update `src/templates/report.html` near Section 2 to show:
  - usable count;
  - tier counts;
  - max revenue ratio when available;
  - accepted/rejected replacement notes.
- [ ] Verify the HTML test passes.

## Task 4: Verification

- [ ] Run `python -m pytest tests/test_reporter.py -v`.
- [ ] Run `python -m ruff check src/reporter.py src/templates/report.html tests/test_reporter.py`.
- [ ] Run `pe-comps`.
- [ ] Inspect `outputs/comps_report.html` for:
  - `Selection quality check`;
  - replacement audit notes if replacements were considered;
  - no regression of the `Mixed / directionally useful with material caveats` label.

