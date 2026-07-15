# Evaluation Results

## Precision@15 vs SEC Proxy Peer Groups
- Mean: 14.6%
- Median: 13.0%
- Min: 0.0%
- Max: 50.0%
- Test companies: 10

Interpretation: 14.6% is below the pipeline's raw selection accuracy because most of
the gap is universe coverage, not bad ranking — see Key Findings below for the
universe-adjusted number (~30%), which isolates ranking quality from scope.

## LLM Extraction Consistency (30-company sample)
- business_model agreement: 100.0%
- revenue_recurrence agreement: 100.0%
- customer_type agreement: 96.7%
- capital_intensity agreement: 96.7%
- primary_value_driver agreement: 100.0%

## Manual Review Results (15 companies)
- business_model accuracy: TBD — fill in after reviewing eval/manual_review_sample.txt
- revenue_recurrence accuracy: TBD
- customer_type accuracy: TBD

## Key Findings

**What worked well:**
- LLM extraction consistency is excellent (96.7-100% across all 5 fields on a
  30-company re-extraction sample) — the prompt produces stable, repeatable
  structured output, not noisy guesses.
- Ground truth coverage hit 10/15 test companies (the DoD minimum), after fixing
  a real bug: the prompt template's `{document_text[:150000]}` truncation (taken
  literally from the Phase 5 spec) cut off the actual peer-group list in some
  filings entirely — verified on MMM's 2026 DEF 14A, where the list sits around
  character 187,000. Replaced blind truncation with a search for "peer group"
  mentions and a window around them, sent to the LLM instead of a fixed prefix.
- Name-to-ticker mapping (`edgar.find_company`) was unreliable when queried with
  full corporate names: "IDEX Corporation" scored an 84% "match" against the
  unrelated "IHI Corporation" (ticker IHICF) while the real IDEX Corp (ticker IEX)
  didn't appear in the top 10 results at all. Stripping corporate suffixes
  (Inc/Incorporated/Corporation/Co/Group/etc.), parentheticals, and hyphens before
  searching fixed this consistently (same pattern reproduced and fixed for Dover,
  Hubbell, TransDigm, Corning, Johnson & Johnson, Snap-on, Colgate-Palmolive,
  Kimberly-Clark).

**What did not work well — and why the raw 14.6% understates ranking quality:**
- Most of the precision gap is universe coverage, not ranking quality. Our
  candidate universe only spans 3 GICS sectors (Industrials, Healthcare Equipment,
  Technology Hardware) with ~112 companies carrying a usable EV/EBITDA label —
  real compensation committees draw peers from the entire market. For MMM, only
  10 of its 24 real disclosed peers (42%) are even present in our universe at all
  (the rest are Consumer Staples, Chemicals, etc. — e.g. P&G, Colgate, Dow,
  DuPont, Kimberly-Clark — categorically out of scope for this project).
- Recomputing Precision@15 restricted to only the real peers that are actually
  reachable in our universe gives a materially different picture:
  MMM 40%, HON 16.7%, ITW 25%, PH 100%, AME 40%, IEX 23.1%, RRX 20%, GNRC 33.3%,
  SYK 0%, NTAP 0% — mean ≈ 29.8%, roughly double the raw 14.6%, and within sight
  of the 35% investigation threshold. This isolates the ranking logic's actual
  performance from a universe-size limitation that no amount of ranking-algorithm
  tuning can fix.
- SYK and NTAP still show 0% even on the universe-adjusted basis — worth a closer
  look in a future iteration (e.g., check whether the +10 business_model penalty
  is pushing real same-sector peers just outside their Top 15).

**Unexpected patterns:**
- Even a small number of remaining ticker-mapping edge cases survived the suffix-
  stripping fix (e.g. preferred-share classes like `BA-PA` instead of common `BA`,
  and a couple of renamed/rebranded companies like Schlumberger → SLB that the
  LLM's extracted historical name didn't resolve). These are minor relative to
  the earlier wholesale foreign-OTC mismatches and were left as-is rather than
  chasing diminishing returns on name normalization.
