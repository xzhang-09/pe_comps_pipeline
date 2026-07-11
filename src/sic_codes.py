import csv
import re
from functools import lru_cache
from pathlib import Path

SIC_CODES_CSV = Path(__file__).with_name("sec_sic_codes.csv")
TITLE_TOKEN_RE = re.compile(r"[A-Z0-9]+")
TITLE_STOPWORDS = {"AND", "OF", "THE", "NO", "NEC"}


@lru_cache(maxsize=1)
def official_sic_codes() -> dict[str, dict[str, str]]:
    """SEC SIC code list vendored from
    https://www.sec.gov/search-filings/standard-industrial-classification-sic-code-list."""
    with SIC_CODES_CSV.open(newline="", encoding="utf-8") as f:
        return {row["sic_code"]: row for row in csv.DictReader(f)}


def _title_tokens(title: str | None) -> set[str]:
    if not title:
        return set()
    return {token for token in TITLE_TOKEN_RE.findall(title.upper()) if token not in TITLE_STOPWORDS}


def title_matches_official(suggested_title: str | None, official_title: str) -> bool:
    suggested = _title_tokens(suggested_title)
    if not suggested:
        return True
    official = _title_tokens(official_title)
    return bool(suggested) and len(suggested & official) / len(suggested) >= 0.5


def validate_sic_suggestion(suggestion: dict) -> dict | None:
    code = str(suggestion.get("sic_code") or "").strip()
    official = official_sic_codes().get(code)
    if official is None:
        return None
    if not title_matches_official(suggestion.get("title"), official["industry_title"]):
        return None
    return {
        **suggestion,
        "sic_code": code,
        "title": official["industry_title"],
        "validated_against_sec": True,
    }
