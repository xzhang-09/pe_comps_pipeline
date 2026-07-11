"""Prefill manual-deal review entries from specific DEFM14A/S-4 filing URLs.

Usage:
    python -m scripts.prefill_manual_deal <filing-url> [<filing-url> ...]

For each EDGAR document URL this extracts the target metadata, fairness
advisor, selected-companies list, and projected financials into
eval/ground_truth/manual_deals.review.json, including a ready-to-copy
`suggested_manual_deal` block per deal. Everything it writes is a prefill:
a human must verify each field against the filing before promoting the deal
into eval/ground_truth/manual_deals.json.
"""
import argparse
from pathlib import Path

import yaml

from eval import ground_truth_builder


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefill manual-deal review entries from EDGAR filing URLs.")
    parser.add_argument("urls", nargs="+", help="EDGAR document URLs (…/Archives/edgar/data/<cik>/<accession>/<doc>.htm)")
    parser.add_argument("--config", default="config.yaml", help="Pipeline config path (for LLM settings).")
    parser.add_argument(
        "--output",
        default=str(ground_truth_builder.MANUAL_DEAL_REVIEW_PATH),
        help="Review JSON path; entries are merged in by filing URL.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    entries = ground_truth_builder.build_manual_deal_review_from_urls(
        args.urls, config, Path(args.output),
    )

    for entry in entries:
        deal = entry["suggested_manual_deal"]
        comps = entry["selected_companies"]
        n_mapped = sum(1 for c in comps if c.get("suggested_ticker"))
        n_public = sum(1 for c in comps if c.get("still_public"))
        financials = entry["target_financials"]
        print(f"\n{deal['deal_id']} — {deal['target_name']} (SIC {deal['target_sic']}, {entry['filing'].get('form_used')} {deal.get('filing_date')})")
        print(f"  Advisor: {deal.get('advisor') or 'NOT FOUND — fill manually'}")
        print(f"  Selected companies: {len(comps)} extracted, {n_mapped} ticker-mapped, {n_public} still US-listed")
        print(
            f"  Target financials: revenue {financials.get('revenue_usd_mm')}mm, "
            f"EBITDA {financials.get('ebitda_usd_mm')}mm ({financials.get('fiscal_year')})"
        )
        unmapped = [c["company_name"] for c in comps if not c.get("suggested_ticker")]
        if unmapped:
            print(f"  Unmapped names (fill tickers manually): {', '.join(unmapped)}")

    print(
        f"\nWrote {len(entries)} review entrie(s) to {args.output}.\n"
        "Every field is a prefill — verify against the filing, set review_status to reviewed,\n"
        "then copy each suggested_manual_deal into eval/ground_truth/manual_deals.json."
    )


if __name__ == "__main__":
    main()
