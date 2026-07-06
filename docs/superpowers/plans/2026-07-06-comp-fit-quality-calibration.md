# Comp Fit Quality Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the comps report's quality language match the actual evidence in the selected comp set, so weak scale/end-market fit cannot be described as overly optimistic "Good" support.

**Architecture:** Keep the existing report pipeline intact. Add a deterministic post-LLM calibration layer in `src/reporter.py` that derives display labels and caveat severity from report data, and tighten the LLM review prompt in `src/comp_fit_reviewer.py` so generated scores are less likely to overstate fit. Tests should verify deterministic behavior without network calls.

**Tech Stack:** Python 3.10+, pytest, existing report generation functions, existing Jinja report template.

---

## Current Diagnosis

The current report is readable and directionally useful, but its quality label is too generous for the evidence shown in the same report.

Observed issues:

- `outputs/comps_report.html` displays `65 / 100 (Good / directionally supportive)`.
- The Top 20 set has only 6 Core comps, 9 Secondary comps, and 5 Review / Exclude comps.
- The target revenue is about `$150mm`; Top 20 comp revenue ranges from `$46mm` to `$1104mm`, up to 7.4x larger.
- The report itself says end-market fit is weak for several comps and revenue scale mismatch is significant.
- The selection diagnostics show `Top 20 EV/EBITDA IQR vs. eligible pool = 90%`, meaning selection barely narrows valuation multiple dispersion.
- The LLM review includes a ticker typo, `HLLO`, while the selected ticker is `HLLY`, showing that narrative fields need stronger deterministic validation.

Root causes:

- `src/reporter.py:_fit_label()` maps every score `>=65` to `"Good / directionally supportive"` unless a narrow scale caveat downgrade triggers.
- The downgrade currently depends on text detection in `weaknesses`, rather than deterministic comp-set metrics like Review / Exclude share, scale ratio, and dispersion.
- `src/comp_fit_reviewer.py` asks the LLM to score, but does not explicitly anchor 60-69 to mixed-quality sets with material caveats.
- `_filter_review_scope()` removes invalid ticker rows in structured callouts, but narrative text can still contain ticker typos or references to unavailable tickers.

## Desired User-Facing Outcome

For the current report shape, the displayed language should read closer to:

```text
65 / 100 (Mixed / directionally useful with material caveats)
```

or, if the calibrated score is lowered:

```text
62 / 100 (Mixed / directionally useful with material caveats)
```

The report should preserve the usefulness of the output while making the caveat explicit:

```text
The selected set is useful for initial directional screening, but not strong enough to support a banker-grade comp set without manual substitutions and scale-controlled sensitivity work.
```

## File Structure

Modify:

- `src/reporter.py`
  - Add deterministic comp-fit diagnostics from `top15_rows`, `audit_trail`, and `model_diagnostics`.
  - Replace the current broad `>=65` label behavior with calibrated labels that account for severe caveats.
  - Add narrative ticker-reference validation for review text fields.

- `src/comp_fit_reviewer.py`
  - Update `REVIEW_PROMPT_VERSION`.
  - Tighten scoring rubric and label guidance in `SYSTEM_PROMPT` / `PROMPT_TEMPLATE`.

- `tests/test_reporter.py`
  - Add focused tests for label downgrade, scale/end-market caveat handling, and narrative ticker typo notes.

- `tests/test_comp_fit_reviewer.py`
  - Add a test that the prompt includes explicit score-band calibration guidance.

Do not modify:

- `src/templates/report.html` unless the calibrated data object requires a new field that cannot be expressed through the existing `fit_label`, `scope_notes`, or summary/caveat fields.
- The core scorer/ranker in `src/scorer.py`; this plan is report quality calibration, not ranking algorithm redesign.

## Task 1: Add Deterministic Report Quality Diagnostics

**Files:**

- Modify: `src/reporter.py`
- Test: `tests/test_reporter.py`

- [ ] **Step 1: Write the failing test**

Add this test near the existing comp-fit review tests in `tests/test_reporter.py`:

```python
def test_html_report_downgrades_good_label_when_fit_evidence_has_material_caveats(mocker):
    mocker.patch("src.reporter.comp_fit_reviewer.review_comp_fit", return_value={
        "status": "available",
        "overall_score": 65,
        "review_confidence": "high",
        "summary": "Directional set with material caveats.",
        "strengths": ["Most companies share a broad manufacturing label."],
        "weaknesses": [
            "Revenue scale mismatch is significant, with several comps above 5x target revenue.",
            "End-market fit is weak for several selected comps.",
        ],
        "top_fits": [{"ticker": "T000", "score": 80, "reason": "Best selected fit."}],
        "questionable_fits": [{"ticker": "T001", "score": 50, "reason": "Different end market."}],
        "near_miss_upgrades": [],
    })
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    scorer_results = _scorer_results(company_scores)
    target_llm_features = _llm(business_model="manufacturing", customer_type="B2B")

    reporter.generate(
        scorer_results, companies, llm_features, target_llm_features, _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "65 / 100 (Mixed / directionally useful with material caveats)" in html_text
    assert "65 / 100 (Good / directionally supportive)" not in html_text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_reporter.py::test_html_report_downgrades_good_label_when_fit_evidence_has_material_caveats -v
```

Expected:

```text
FAILED ... assert '65 / 100 (Mixed / directionally useful with material caveats)' in html_text
```

- [ ] **Step 3: Implement the minimal label calibration**

In `src/reporter.py`, replace `_fit_label()` with a caveat-aware version that keeps the old public signature and adds deterministic label names:

```python
def _has_material_fit_caveat(weaknesses: list[str] | None) -> bool:
    if not weaknesses:
        return False
    joined = " ".join(weaknesses).lower()
    material_terms = (
        "significant",
        "weak for several",
        "several comps",
        "scale mismatch",
        "revenue scale",
        "end-market fit is weak",
        "customer type mismatch",
        "review / exclude",
    )
    return any(term in joined for term in material_terms)


def _fit_label(score: float | int | None, weaknesses: list[str] | None = None) -> str | None:
    if score is None:
        return None
    if score >= 80:
        return "Strong"
    if score >= 65:
        if _has_severe_scale_caveat(weaknesses):
            return "Mixed / directionally useful with material scale caveats"
        if _has_material_fit_caveat(weaknesses):
            return "Mixed / directionally useful with material caveats"
        return "Good / directionally supportive"
    if score >= 50:
        return "Mixed / directionally useful with material caveats"
    return "Weak"
```

Rationale:

- This is intentionally conservative and scoped.
- It does not change the LLM's numeric score yet.
- It prevents the most misleading case: a score barely at 65 plus explicit serious weaknesses being called `Good`.

- [ ] **Step 4: Run the focused test**

Run:

```bash
pytest tests/test_reporter.py::test_html_report_downgrades_good_label_when_fit_evidence_has_material_caveats -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Run nearby reporter tests**

Run:

```bash
pytest tests/test_reporter.py::test_html_report_includes_comp_fit_review_when_available tests/test_reporter.py::test_html_report_filters_comp_fit_review_tickers_to_their_scope -v
```

Expected:

```text
2 passed
```

## Task 2: Add Quantitative Caveat Signals to the Label Decision

**Files:**

- Modify: `src/reporter.py`
- Test: `tests/test_reporter.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_reporter.py`:

```python
def test_fit_label_downgrades_when_many_selected_comps_are_review_exclude():
    rows = [
        {"ticker": "A", "tier": "core", "revenue_ttm_usd_mm": 100.0},
        {"ticker": "B", "tier": "secondary", "revenue_ttm_usd_mm": 120.0},
        {"ticker": "C", "tier": "review_exclude", "revenue_ttm_usd_mm": 700.0},
        {"ticker": "D", "tier": "review_exclude", "revenue_ttm_usd_mm": 800.0},
    ]

    diagnostics = reporter._fit_quality_diagnostics(rows, target_revenue=100.0, model_diagnostics={})

    assert diagnostics["review_exclude_share"] == pytest.approx(0.5)
    assert diagnostics["max_revenue_ratio"] == pytest.approx(8.0)
    assert reporter._fit_label(68, [], diagnostics) == "Mixed / directionally useful with material caveats"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_reporter.py::test_fit_label_downgrades_when_many_selected_comps_are_review_exclude -v
```

Expected:

```text
FAILED ... AttributeError: module 'src.reporter' has no attribute '_fit_quality_diagnostics'
```

- [ ] **Step 3: Add diagnostics helper**

In `src/reporter.py`, add this helper above `_fit_label()`:

```python
def _fit_quality_diagnostics(
    rows: list[dict],
    target_revenue: float | int | None,
    model_diagnostics: dict | None,
) -> dict[str, float | int | None]:
    total = len(rows)
    review_exclude_count = sum(1 for row in rows if row.get("tier") == "review_exclude")
    secondary_count = sum(1 for row in rows if row.get("tier") == "secondary")
    review_exclude_share = review_exclude_count / total if total else 0.0
    secondary_or_review_share = (secondary_count + review_exclude_count) / total if total else 0.0

    max_revenue_ratio = None
    if target_revenue:
        ratios = []
        for row in rows:
            revenue = row.get("revenue_ttm_usd_mm")
            if revenue and revenue > 0:
                ratios.append(max(float(revenue) / float(target_revenue), float(target_revenue) / float(revenue)))
        max_revenue_ratio = max(ratios) if ratios else None

    dispersion = None
    if model_diagnostics:
        dispersion = model_diagnostics.get("selected_ev_ebitda_iqr_vs_pool")

    return {
        "review_exclude_count": review_exclude_count,
        "review_exclude_share": review_exclude_share,
        "secondary_or_review_share": secondary_or_review_share,
        "max_revenue_ratio": max_revenue_ratio,
        "selected_ev_ebitda_iqr_vs_pool": dispersion,
    }
```

- [ ] **Step 4: Extend label function signature**

Update `_fit_label()` to accept diagnostics:

```python
def _has_material_quant_caveat(diagnostics: dict[str, float | int | None] | None) -> bool:
    if not diagnostics:
        return False
    review_exclude_share = diagnostics.get("review_exclude_share") or 0
    secondary_or_review_share = diagnostics.get("secondary_or_review_share") or 0
    max_revenue_ratio = diagnostics.get("max_revenue_ratio")
    dispersion = diagnostics.get("selected_ev_ebitda_iqr_vs_pool")
    return (
        review_exclude_share >= 0.25
        or secondary_or_review_share >= 0.65
        or (max_revenue_ratio is not None and max_revenue_ratio >= 5.0)
        or (dispersion is not None and dispersion >= 0.85)
    )


def _fit_label(
    score: float | int | None,
    weaknesses: list[str] | None = None,
    diagnostics: dict[str, float | int | None] | None = None,
) -> str | None:
    if score is None:
        return None
    if score >= 80:
        return "Strong"
    if score >= 65:
        if _has_severe_scale_caveat(weaknesses):
            return "Mixed / directionally useful with material scale caveats"
        if _has_material_fit_caveat(weaknesses) or _has_material_quant_caveat(diagnostics):
            return "Mixed / directionally useful with material caveats"
        return "Good / directionally supportive"
    if score >= 50:
        return "Mixed / directionally useful with material caveats"
    return "Weak"
```

- [ ] **Step 5: Wire diagnostics into `generate()`**

In `src/reporter.py`, move fit-label assignment until after `top15_rows` has final tiers, or recompute it after `_fill_usable_comp_slots()`.

Use this pattern after `top15_rows, top15, flagged_tickers = _fill_usable_comp_slots(...)`:

```python
    fit_quality_diagnostics = _fit_quality_diagnostics(top15_rows, target_revenue, model_diagnostics)
    comp_fit_review["fit_quality_diagnostics"] = fit_quality_diagnostics
    comp_fit_review["fit_label"] = (
        _fit_label(
            comp_fit_review.get("overall_score"),
            comp_fit_review.get("weaknesses"),
            fit_quality_diagnostics,
        )
        if comp_fit_review.get("status") == "available" else None
    )
```

Remove the earlier pre-fill `comp_fit_review["fit_label"] = ...` block so the displayed label is based on final table tiers.

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest tests/test_reporter.py::test_fit_label_downgrades_when_many_selected_comps_are_review_exclude tests/test_reporter.py::test_html_report_downgrades_good_label_when_fit_evidence_has_material_caveats -v
```

Expected:

```text
2 passed
```

## Task 3: Validate Narrative Ticker References

**Files:**

- Modify: `src/reporter.py`
- Test: `tests/test_reporter.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_reporter.py`:

```python
def test_html_report_notes_invalid_ticker_references_in_review_narrative(mocker):
    mocker.patch("src.reporter.comp_fit_reviewer.review_comp_fit", return_value={
        "status": "available",
        "overall_score": 65,
        "review_confidence": "high",
        "summary": "Directional set.",
        "strengths": [],
        "weaknesses": ["One selected comp (HLLO) serves B2C customers."],
        "top_fits": [],
        "questionable_fits": [],
        "near_miss_upgrades": [],
    })
    companies, llm_features, company_scores = _build_sample(n=30, n_matching=15)
    llm_features["T001"] = _llm(customer_type="B2C")
    scorer_results = _scorer_results(company_scores)

    reporter.generate(
        scorer_results, companies, llm_features, _llm(business_model="manufacturing", customer_type="B2B"),
        _imputation_medians(), _sample_config(),
    )

    html_text = reporter.HTML_PATH.read_text(encoding="utf-8")
    assert "Review narrative referenced unavailable ticker HLLO" in html_text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_reporter.py::test_html_report_notes_invalid_ticker_references_in_review_narrative -v
```

Expected:

```text
FAILED ... assert 'Review narrative referenced unavailable ticker HLLO' in html_text
```

- [ ] **Step 3: Add narrative ticker extraction helper**

In `src/reporter.py`, add `import re` only if not already present. Then add helpers near `_filter_review_scope()`:

```python
TICKER_REFERENCE_PATTERN = re.compile(r"\b[A-Z]{2,5}\b")
NARRATIVE_TICKER_ALLOWLIST = {
    "B2B",
    "B2C",
    "EV",
    "EBITDA",
    "OEM",
    "OEMs",
    "SEC",
    "FMP",
    "LLM",
    "AI",
}


def _review_narrative_text(review: dict) -> str:
    parts = [str(review.get("summary") or "")]
    for key in ("strengths", "weaknesses"):
        parts.extend(str(item) for item in review.get(key, []) if item)
    return " ".join(parts)


def _invalid_review_ticker_references(review: dict, allowed_tickers: set[str]) -> list[str]:
    candidates = set(TICKER_REFERENCE_PATTERN.findall(_review_narrative_text(review)))
    invalid = candidates - allowed_tickers - NARRATIVE_TICKER_ALLOWLIST
    return sorted(invalid)
```

- [ ] **Step 4: Wire narrative validation into `_filter_review_scope()`**

At the end of `_filter_review_scope()`, before assigning `review["scope_notes"]`, add:

```python
    invalid_narrative_tickers = _invalid_review_ticker_references(
        review,
        allowed_tickers=selected_tickers | near_miss_tickers,
    )
    for ticker in invalid_narrative_tickers:
        notes.append(f"Review narrative referenced unavailable ticker {ticker}.")
```

Keep existing scope notes behavior.

- [ ] **Step 5: Run focused test**

Run:

```bash
pytest tests/test_reporter.py::test_html_report_notes_invalid_ticker_references_in_review_narrative -v
```

Expected:

```text
PASSED
```

## Task 4: Tighten LLM Score-Band Guidance

**Files:**

- Modify: `src/comp_fit_reviewer.py`
- Test: `tests/test_comp_fit_reviewer.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_comp_fit_reviewer.py`:

```python
def test_review_prompt_contains_score_band_calibration_guidance():
    assert "Score-band calibration" in comp_fit_reviewer.SYSTEM_PROMPT
    assert "60-69" in comp_fit_reviewer.SYSTEM_PROMPT
    assert "material caveats" in comp_fit_reviewer.SYSTEM_PROMPT
    assert "Do not call a set good" in comp_fit_reviewer.SYSTEM_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_comp_fit_reviewer.py::test_review_prompt_contains_score_band_calibration_guidance -v
```

Expected:

```text
FAILED ... AssertionError
```

- [ ] **Step 3: Update prompt version**

In `src/comp_fit_reviewer.py`, update:

```python
REVIEW_PROMPT_VERSION = "analyst_memo_v8_score_calibrated"
```

- [ ] **Step 4: Add score-band rules to `SYSTEM_PROMPT`**

Append this block inside `SYSTEM_PROMPT` after the existing scale-ratio rules:

```text
- Score-band calibration:
  - 80-100 means strong comp-set support with only minor caveats.
  - 70-79 means useful support, but not banker-grade without analyst review.
  - 60-69 means mixed directional support with material caveats.
  - 50-59 means weak/mixed support; use only as a rough screen.
  - Below 50 means weak support.
- Do not call a set good if several selected comps have clearly different end
  markets, if 25% or more of selected comps would require review/exclusion, or
  if multiple selected comps exceed 5x target revenue. In those cases, scores
  in the 60-69 range are usually more appropriate unless there are unusually
  strong offsetting reasons.
```

- [ ] **Step 5: Run prompt test**

Run:

```bash
pytest tests/test_comp_fit_reviewer.py::test_review_prompt_contains_score_band_calibration_guidance -v
```

Expected:

```text
PASSED
```

## Task 5: Update Existing Tests for Label Text

**Files:**

- Modify: `tests/test_reporter.py`

- [ ] **Step 1: Run existing reporter tests to find expected-label failures**

Run:

```bash
pytest tests/test_reporter.py -v
```

Expected:

```text
Some tests may fail only where they assert the old "Good / directionally supportive" wording for a caveat-heavy report.
```

- [ ] **Step 2: Update only stale assertions**

If a test sets `overall_score` to `70` or `84` without material caveats, keep existing expectations.

If a test sets `overall_score` to `65-79` and deliberately includes material caveats, update the assertion to:

```python
assert "Mixed / directionally useful with material caveats" in html_text
```

Do not weaken assertions to generic substrings like `"directionally"` unless the test is not about label calibration.

- [ ] **Step 3: Re-run reporter tests**

Run:

```bash
pytest tests/test_reporter.py -v
```

Expected:

```text
All tests in tests/test_reporter.py pass.
```

## Task 6: Regenerate the Current Sample Report for Manual Review

**Files:**

- Modify by command output: `outputs/comps_report.html`, `outputs/comps_report.csv`, `outputs/comp_fit_review.json`

- [ ] **Step 1: Run the pipeline with existing config**

Run:

```bash
pe-comps
```

Expected:

```text
Pipeline completes and writes outputs/comps_report.html.
```

- [ ] **Step 2: Inspect the new label**

Run:

```bash
rg -n "Comp fit:|Overall Top|Review narrative|Good / directionally|Mixed / directionally" outputs/comps_report.html
```

Expected for the current Precision Motion report:

```text
Comp fit: 65 / 100 (Mixed / directionally useful with material caveats)
Overall Top 20 Fit: 65 / 100 (Mixed / directionally useful with material caveats)
```

If `HLLO` still appears in narrative text, expected:

```text
Review narrative referenced unavailable ticker HLLO.
```

- [ ] **Step 3: Manually review readability**

Open `outputs/comps_report.html` and verify:

- The Executive Summary no longer suggests the comp set is stronger than the caveats support.
- Section 3's label matches the weaknesses immediately below it.
- The report still remains useful as a directional screen.
- The Data Notes still clearly say outputs are not final diligence conclusions.

## Task 7: Final Verification

**Files:**

- No new modifications expected.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/test_reporter.py tests/test_comp_fit_reviewer.py -v
```

Expected:

```text
All selected tests pass.
```

- [ ] **Step 2: Run lint**

Run:

```bash
ruff check src/reporter.py src/comp_fit_reviewer.py tests/test_reporter.py tests/test_comp_fit_reviewer.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 3: Review git diff**

Run:

```bash
git diff -- src/reporter.py src/comp_fit_reviewer.py tests/test_reporter.py tests/test_comp_fit_reviewer.py docs/superpowers/plans/2026-07-06-comp-fit-quality-calibration.md
```

Expected:

- Changes are scoped to label calibration, ticker-reference validation, prompt guidance, and tests.
- No unrelated refactors.
- No secrets or `.env` values appear.

## Acceptance Criteria

The fix is acceptable when:

- A score of `65` with material caveats no longer displays as `Good / directionally supportive`.
- A clean score of `70+` without material caveats can still display as `Good / directionally supportive`.
- Quantitative evidence can trigger downgrade even if the LLM weaknesses are too mild.
- Narrative ticker typos such as `HLLO` are surfaced as review notes.
- The comp-fit prompt explicitly calibrates 60-69 as mixed directional support with material caveats.
- Existing report sections remain readable and no template redesign is required.

## Future Work Not Included

These are deliberately out of scope for the first fix:

- Re-ranking the comp set itself.
- Replacing the LLM reviewer with a fully deterministic scoring model.
- Adding an audited external benchmark for comp selection.
- Changing valuation methodology or private-company discount assumptions.
- Rebuilding the report design.

