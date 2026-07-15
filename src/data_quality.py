import json
import statistics
from datetime import datetime
from pathlib import Path

from src import get_logger

logger = get_logger(__name__)

OUTPUT_PATH = Path("outputs/data_quality_report.txt")
CACHE_DIR = Path("data/cache")

NUMERIC_FIELDS = [
    "revenue_ttm_usd_mm",
    "ebitda_margin",
    "gross_margin",
    "revenue_cagr_3yr",
    "net_debt_ebitda",
    "capex_revenue",
    "ev_ebitda",
    "ev_revenue",
]

EV_EBITDA_LOW = 6.0
EV_EBITDA_HIGH = 20.0
SHORT_DESCRIPTION_LENGTH = 200


def _non_null_values(companies: list[dict], field: str) -> list[float]:
    return [c[field] for c in companies if c.get(field) is not None]


def _field_stats(companies: list[dict], field: str) -> dict:
    total = len(companies)
    values = sorted(_non_null_values(companies, field))
    non_null = len(values)
    missing_pct = ((total - non_null) / total * 100) if total else 0.0

    stats = {
        "non_null": non_null,
        "missing_pct": missing_pct,
        "min": None,
        "p25": None,
        "median": None,
        "p75": None,
        "max": None,
    }

    if values:
        stats["min"] = values[0]
        stats["max"] = values[-1]
        stats["median"] = statistics.median(values)
        if len(values) >= 2:
            p25, _, p75 = statistics.quantiles(values, n=4, method="inclusive")
            stats["p25"], stats["p75"] = p25, p75
        else:
            stats["p25"] = stats["p75"] = values[0]

    return stats


def _business_description_stats(companies: list[dict]) -> dict:
    descriptions = [c["business_description"] for c in companies if c.get("business_description")]
    non_null = len(descriptions)
    avg_length = sum(len(d) for d in descriptions) / non_null if non_null else 0.0
    short_count = sum(1 for d in descriptions if len(d) < SHORT_DESCRIPTION_LENGTH)
    return {"non_null": non_null, "avg_length": avg_length, "short_count": short_count}


def _industry_breakdown(companies: list[dict]) -> dict:
    breakdown = {}
    for c in companies:
        sector = c.get("gics_sector") or "Unknown"
        breakdown[sector] = breakdown.get(sector, 0) + 1
    return breakdown


def generate_report(companies: list[dict]) -> str:
    """
    Generate data quality report from list of company dicts.
    Writes report to outputs/data_quality_report.txt.
    Returns the report text as a string.
    """
    total = len(companies)
    lines = [
        "=== DATA QUALITY REPORT ===",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total companies fetched: {total}",
        "",
        "--- FIELD COMPLETENESS ---",
        f"{'Field':<22}{'Non-null':<12}{'Missing%':<12}{'Median':<10}",
    ]

    field_stats = {}
    for field in NUMERIC_FIELDS:
        stats = _field_stats(companies, field)
        field_stats[field] = stats
        median_str = f"{stats['median']:.2f}" if stats["median"] is not None else "N/A"
        missing_str = f"{stats['missing_pct']:.1f}%"
        lines.append(f"{field:<22}{stats['non_null']:<12}{missing_str:<12}{median_str:<10}")

    lines += ["", "--- EV/EBITDA DISTRIBUTION ---"]
    ev_stats = field_stats["ev_ebitda"]
    if ev_stats["non_null"] > 0:
        lines.append(
            f"Min: {ev_stats['min']:.1f}x   P25: {ev_stats['p25']:.1f}x   "
            f"Median: {ev_stats['median']:.1f}x   P75: {ev_stats['p75']:.1f}x   Max: {ev_stats['max']:.1f}x"
        )
        in_range = EV_EBITDA_LOW <= ev_stats["median"] <= EV_EBITDA_HIGH
        if in_range:
            status = "OK (median within expected range 6x-20x)"
        else:
            status = f"WARNING (median {ev_stats['median']:.1f}x outside expected range 6x-20x)"
        lines.append(f"Status: {status}")
    else:
        lines.append("No valid ev_ebitda values available.")

    lines += ["", "--- BUSINESS DESCRIPTIONS ---"]
    desc_stats = _business_description_stats(companies)
    pct = (desc_stats["non_null"] / total * 100) if total else 0.0
    lines.append(f"Non-null: {desc_stats['non_null']} / {total} ({pct:.0f}%)")
    lines.append(f"Avg length: {desc_stats['avg_length']:.0f} chars")
    lines.append(f"Short (<{SHORT_DESCRIPTION_LENGTH} chars): {desc_stats['short_count']} companies")

    lines += ["", "--- INDUSTRY BREAKDOWN ---"]
    breakdown = _industry_breakdown(companies)
    for sector, count in sorted(breakdown.items()):
        lines.append(f"{sector:<12}{count} companies")

    report = "\n".join(lines) + "\n"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Data quality report written to {OUTPUT_PATH}")

    return report


# data/cache/ also holds non-company cache files written by other modules
# — sic_universe_builder's sic_universe_{sic}.json (a list, not a dict) and
# universe_builder's universe_market_cap.json, and ground_truth_builder's
# peer_group_{ticker}.json (a dict, but not a company record). A bare
# glob("*.json") would either crash on the list or silently pollute the
# stats with the dict.
NON_COMPANY_CACHE_PREFIXES = ("sic_universe_", "peer_group_")
NON_COMPANY_CACHE_FILES = ("universe_market_cap.json",)

if __name__ == "__main__":
    companies = []
    for cache_file in sorted(CACHE_DIR.glob("*.json")):
        if cache_file.name in NON_COMPANY_CACHE_FILES or cache_file.name.startswith(NON_COMPANY_CACHE_PREFIXES):
            continue
        with open(cache_file, "r", encoding="utf-8") as f:
            companies.append(json.load(f))

    report_text = generate_report(companies)
    print(report_text)
