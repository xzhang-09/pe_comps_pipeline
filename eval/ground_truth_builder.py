import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import edgar
import openai
import requests

from src import get_logger, sic_universe_builder
from src.llm_analyzer import _call_openai, _strip_markdown_fences

logger = get_logger(__name__)

CACHE_DIR = Path("data/cache")
DEFAULT_SEC_IDENTITY = "PE-Comps-Pipeline research@example.com"
EDGAR_IDENTITY = os.environ.get("SEC_IDENTITY", DEFAULT_SEC_IDENTITY)

DOCUMENT_MAX_CHARS = 200_000
MIN_NAME_MATCH_SCORE = 70.0

# A banker's "Selected Companies Analysis" in a merger proxy/fairness opinion
# picks comps specifically to value the target — unlike a DEF 14A
# compensation peer group (chosen by a comp committee to benchmark pay,
# often deliberately including aspirational/larger peers). This is a much
# better ground truth for a tool whose whole job is valuation comparability,
# at the cost of a much smaller candidate pool: it only exists for companies
# that were actually acquired (or did a stock-for-stock deal), not for any
# live public company.
FAIRNESS_OPINION_KEYWORDS = (
    "selected companies analysis", "comparable companies analysis",
    "selected public companies analysis", "selected publicly traded companies",
)
# Filed by the target (DEFM14A, when shareholders must vote) or by the
# acquirer (S-4, for stock-for-stock deals registering the new shares).
FAIRNESS_OPINION_FORMS = ("DEFM14A", "S-4")
KEYWORD_WINDOW_BEFORE = 300
KEYWORD_WINDOW_AFTER = 3000
MAX_PROMPT_CHARS = 60_000

# SEC's public, no-key-required full text search API (the same one backing
# https://www.sec.gov/edgar/search/). Covers filings from 2001 onward.
# NOTE: validate this response shape against live SEC responses before
# publishing benchmark results from the generated ground truth.
FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
FULL_TEXT_SEARCH_PAGE_SIZE = 10
FULL_TEXT_SEARCH_MAX_PAGES = 10
FULL_TEXT_SEARCH_HEADERS = {"User-Agent": EDGAR_IDENTITY}
REQUEST_TIMEOUT_SECONDS = 10
SEC_REQUEST_DELAY_SECONDS = 0.2

# edgar.find_company() does token-overlap fuzzy matching, so normalize generic
# corporate suffixes and punctuation before searching company names.
_CORPORATE_SUFFIX_RE = re.compile(
    r"[,.]?\s*\b(Incorporated|Inc|Corporation|Corp|Company|Co|Holdings?|Group|plc|PLC|Limited|Ltd|LLC)\b\.?\s*$",
    re.IGNORECASE,
)
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)\s*$")
_LEADING_THE_RE = re.compile(r"^The\s+", re.IGNORECASE)


def _normalize_company_name(name: str) -> str:
    cleaned = _PARENTHETICAL_RE.sub("", name.strip()).strip()
    cleaned = _LEADING_THE_RE.sub("", cleaned).strip()
    cleaned = cleaned.replace("-", " ")

    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = _CORPORATE_SUFFIX_RE.sub("", cleaned).strip().rstrip(",.")

    return cleaned or name


FAIRNESS_OPINION_PROMPT_TEMPLATE = """You are extracting a "Selected Companies Analysis" (sometimes called
"Comparable Companies Analysis" or "Selected Public Companies Analysis")
from a merger proxy statement or fairness opinion.

Find the section where a financial advisor lists the public companies they
selected as valuation comparables for the transaction. Extract the list of
company names in that selected-companies list.

Document text (may be truncated):
{document_text}

Return ONLY a JSON array of company name strings.
Example: ["Company A", "Company B", "Company C"]
If you cannot find a selected-companies list, return an empty array: []
Do not include any other text."""

# Manual override list for local experiments. Prefer
# discover_fairness_opinion_candidates() for a dynamically sourced list of real
# acquisitions.
EXAMPLE_MANUAL_TEST_TICKERS = [
    "MMM", "HON", "EMR", "ITW", "PH", "AME", "IEX", "RRX", "XYL", "GNRC",
    "MDT", "SYK", "ZBH", "EW", "NTAP",
]


def _cache_path(identifier: str) -> Path:
    return CACHE_DIR / f"comp_analysis_{identifier}.json"


def _load_cache(identifier: str) -> dict | None:
    path = _cache_path(identifier)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_cache(identifier: str, record: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(identifier), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def _search_full_text(phrase: str, forms: tuple[str, ...], max_results: int) -> list[dict]:
    """
    Search SEC's full text search index for `phrase` within the given form
    types. Returns a list of {"cik", "company_name", "file_date", "form"}
    dicts, one per matching filing, paginated FULL_TEXT_SEARCH_PAGE_SIZE at
    a time up to max_results. Never raises — returns [] on any failure,
    since this is a discovery aid, not something that should block the
    whole ground-truth build over one bad request.
    """
    results = []
    page = 0
    while len(results) < max_results and page < FULL_TEXT_SEARCH_MAX_PAGES:
        try:
            resp = requests.get(
                FULL_TEXT_SEARCH_URL,
                params={
                    "q": f'"{phrase}"',
                    "forms": ",".join(forms),
                    "from": page * FULL_TEXT_SEARCH_PAGE_SIZE,
                },
                headers=FULL_TEXT_SEARCH_HEADERS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Full text search failed for {phrase!r}: {e}")
            break

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break

        for hit in hits:
            source = hit.get("_source", {})
            ciks = source.get("ciks") or []
            if not ciks:
                continue
            results.append({
                "cik": int(ciks[0]),
                "company_name": (source.get("display_names") or [None])[0],
                "file_date": source.get("file_date"),
                "form": source.get("form") or source.get("form_type"),
            })

        if len(hits) < FULL_TEXT_SEARCH_PAGE_SIZE:
            break
        page += 1
        time.sleep(SEC_REQUEST_DELAY_SECONDS)

    return results[:max_results]


def discover_fairness_opinion_candidates(sic_codes: list[str], max_results: int = 20) -> list[dict]:
    """
    Discover real acquisitions/mergers with a "Selected Companies Analysis"
    in their fairness opinion, restricted to the given SIC codes — a
    dynamically-sourced alternative to hardcoding a list of test tickers.

    Returns a list of {"cik", "ticker", "company_name", "sic", "file_date",
    "form"} dicts, one per matching company (deduplicated by CIK across all
    FAIRNESS_OPINION_KEYWORDS phrases and SEC's own form aliases). `ticker`
    is None for companies with no ticker on file (common for delisted
    targets, especially older or thinly-traded ones) — callers should fall
    back to the CIK as the identifier in that case.
    """
    sic_set = set(sic_codes)
    by_cik: dict[int, dict] = {}

    for phrase in FAIRNESS_OPINION_KEYWORDS:
        for hit in _search_full_text(phrase, FAIRNESS_OPINION_FORMS, max_results):
            cik = hit["cik"]
            if cik not in by_cik:
                by_cik[cik] = hit

    candidates = []
    for cik, hit in by_cik.items():
        try:
            profile = sic_universe_builder.fetch_company_profile(cik)
        except Exception as e:
            logger.warning(f"CIK {cik} — profile lookup failed: {e}. Skipping.")
            continue
        if profile is None:
            continue

        sic = profile.get("sic")
        if sic not in sic_set:
            continue

        tickers = [t for t in profile.get("tickers", []) if t]
        candidates.append({
            "cik": cik,
            "ticker": tickers[0] if tickers else None,
            "company_name": hit.get("company_name") or profile.get("name"),
            "sic": sic,
            "file_date": hit.get("file_date"),
            "form": hit.get("form"),
        })

    logger.info(f"Found {len(candidates)} fairness-opinion candidates matching SIC codes {sic_codes}")
    return candidates[:max_results]


def _fetch_fairness_opinion_text(identifier: int | str) -> tuple[str | None, str | None]:
    """
    Latest fairness-opinion-bearing filing (DEFM14A, falling back to S-4)
    for a company, via edgartools — tries each form in FAIRNESS_OPINION_FORMS
    in order since deals close either type depending on consideration
    (cash vs. stock). `identifier` can be a ticker or a CIK; CIK is
    preferred for delisted acquisition targets that may have no ticker on
    file. Returns (text, form_used), or (None, None) if neither form has a
    filing on record.
    """
    company = edgar.Company(identifier)
    for form in FAIRNESS_OPINION_FORMS:
        filing = company.get_filings(form=form).latest()
        if filing is not None:
            return filing.text()[:DOCUMENT_MAX_CHARS], form
    return None, None


def _extract_relevant_windows(text: str) -> str:
    """Find every mention of a fairness-opinion keyword and return the text
    around those mentions (merging overlapping windows), instead of a
    blind prefix truncation that can miss the section entirely — verified
    necessary on a real DEF 14A where the relevant section sat past a
    150,000-char cutoff."""
    text_lower = text.lower()
    positions = set()
    for keyword in FAIRNESS_OPINION_KEYWORDS:
        start = 0
        while True:
            idx = text_lower.find(keyword, start)
            if idx == -1:
                break
            positions.add(idx)
            start = idx + len(keyword)

    if not positions:
        return text[:MAX_PROMPT_CHARS]

    windows = []
    for pos in sorted(positions):
        start = max(0, pos - KEYWORD_WINDOW_BEFORE)
        end = min(len(text), pos + KEYWORD_WINDOW_AFTER)
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))

    combined = "\n...\n".join(text[start:end] for start, end in windows)
    return combined[:MAX_PROMPT_CHARS]


def _parse_json_array(text: str | None, label: str) -> list | None:
    """Like llm_analyzer._parse_json_response, but for prompts that return a
    JSON array rather than an object — that function hard-requires a dict,
    which would reject every valid selected-companies response here."""
    if not text:
        return None
    cleaned = _strip_markdown_fences(text)
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"{label} — failed to parse LLM JSON response: {e}")
        return None
    if not isinstance(parsed, list):
        logger.warning(f"{label} — LLM JSON response was not an array: {cleaned!r}")
        return None
    return parsed


def _extract_selected_companies(client: openai.OpenAI, label: str, document_text: str, config: dict) -> list[str]:
    relevant_text = _extract_relevant_windows(document_text)
    prompt = FAIRNESS_OPINION_PROMPT_TEMPLATE.format(document_text=relevant_text)

    try:
        response_text = _call_openai(
            client, config["llm"]["judge_model"], None, prompt,
            config["llm"]["temperature"], config["llm"]["max_tokens"],
        )
    except Exception as e:
        logger.warning(f"{label} — selected-companies extraction API call failed: {e}")
        return []

    parsed = _parse_json_array(response_text, label)
    if parsed is None:
        logger.warning(f"{label} — selected-companies extraction did not return a JSON array")
        return []

    return [name for name in parsed if isinstance(name, str)]


def _map_name_to_ticker(label: str, company_name: str) -> str | None:
    """
    Map a selected comp's company name to a ticker via edgartools' fuzzy
    company search (cik/ticker/company/score columns) — find_company gives
    an explicit match-quality score to apply the "reasonable match"
    judgment call against. The name is normalized first (see
    _normalize_company_name) since searching with corporate suffixes
    attached produces wrong high-scoring matches.
    """
    query = _normalize_company_name(company_name)
    try:
        results = edgar.find_company(query)
    except Exception as e:
        logger.warning(f"{label} — name lookup failed for {company_name!r}: {e}")
        return None

    if results.results is None or results.results.empty:
        logger.warning(f"{label} — could not map {company_name!r} to a ticker")
        return None

    top = results.results.iloc[0]
    if top["score"] < MIN_NAME_MATCH_SCORE:
        logger.warning(f"{label} — best match for {company_name!r} too weak ({top['score']:.0f}%), skipping")
        return None

    return top["ticker"]


def _candidate_identifier(candidate: dict | str) -> int | str:
    if isinstance(candidate, str):
        return candidate
    return candidate.get("cik") or candidate["ticker"]


def _candidate_label(candidate: dict | str) -> str:
    if isinstance(candidate, str):
        return candidate
    return candidate.get("ticker") or f"CIK{candidate.get('cik')}"


def build_ground_truth(test_candidates: list[dict | str], config: dict) -> dict[str, list[str]]:
    """
    For each test candidate, extract its fairness opinion's selected
    comparable companies from SEC EDGAR. `test_candidates` is either a list
    of tickers (str) or candidate dicts from
    discover_fairness_opinion_candidates() (which may have a CIK but no
    ticker, for delisted targets).

    Returns:
        dict mapping ticker-or-"CIK{n}" label -> list of selected-comp tickers

    Companies whose selected-companies list could not be extracted are
    omitted from results. Caches results to
    data/cache/comp_analysis_{identifier}.json.
    """
    edgar.set_identity(EDGAR_IDENTITY)
    client = openai.OpenAI()
    ground_truth = {}

    for candidate in test_candidates:
        identifier = _candidate_identifier(candidate)
        label = _candidate_label(candidate)
        cache_key = str(identifier)

        cached = _load_cache(cache_key)
        if cached is not None:
            logger.info(f"{label} — selected companies loaded from cache")
            if cached.get("selected_company_tickers"):
                ground_truth[label] = cached["selected_company_tickers"]
            continue

        try:
            document_text, form_used = _fetch_fairness_opinion_text(identifier)
        except Exception as e:
            logger.warning(f"{label} — failed to fetch fairness opinion filing: {e}")
            continue

        if document_text is None:
            logger.warning(f"{label} — no {'/'.join(FAIRNESS_OPINION_FORMS)} filing found")
            continue

        selected_names = _extract_selected_companies(client, label, document_text, config)

        selected_tickers = []
        unmapped = []
        for name in selected_names:
            mapped = _map_name_to_ticker(label, name)
            if mapped:
                selected_tickers.append(mapped)
            else:
                unmapped.append(name)

        if unmapped:
            logger.warning(f"{label} — could not map {len(unmapped)} selected-company names to tickers: {unmapped}")

        record = {
            "identifier": cache_key,
            "label": label,
            "form_used": form_used,
            "selected_company_tickers": selected_tickers,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_cache(cache_key, record)

        if selected_tickers:
            ground_truth[label] = selected_tickers

    return ground_truth
