import html as html_lib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import edgar
import openai
import requests

from src import get_logger, sic_universe_builder
from src.defaults import sec_identity
from src.llm_analyzer import _call_openai_structured
from src.llm_schemas import AdvisorAndSelectedCompanies, DealFinancials, SelectedCompaniesList

logger = get_logger(__name__)

CACHE_DIR = Path("data/cache")
MANUAL_DEAL_REVIEW_PATH = Path("eval/ground_truth/manual_deals.review.json")
MANUAL_REVIEW_FIELDS = [
    "filing_url",
    "advisor",
    "target_financials",
    "business_description",
    "selected_company_tickers",
    "selected_company_still_public_flags",
]
EDGAR_IDENTITY = sec_identity()

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
    # Singular "Company" variant — Centerview's section heading on
    # Squarespace's DEFM14A was "Selected Public Company Analysis", which
    # none of the plural-only keywords above matched, so the extractor fell
    # back to the document prefix and missed the section entirely.
    "selected public company analysis",
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

If you cannot find a selected-companies list, return an empty list."""

# A specific filing document under /Archives/edgar/data/<cik>/<accession-no-dashes>/…
# — the URL shape produced by EDGAR full-text search and filing-index pages.
FILING_URL_RE = re.compile(r"/Archives/edgar/data/(\d+)/(\d{18})/", re.IGNORECASE)

# Section-title keywords for the target's management projections, used to pull
# prompt windows the same way FAIRNESS_OPINION_KEYWORDS does for the comps list.
PROJECTION_KEYWORDS = (
    "prospective financial information",
    "financial projections",
    "management projections",
    "financial forecasts",
    "projected financial information",
)
# Projection tables run longer than a comps name list, so the after-window is
# wider than KEYWORD_WINDOW_AFTER.
PROJECTION_WINDOW_AFTER = 4000

# The advisor + selected-companies JSON for a 15-20 name list overflows the
# pipeline's default llm.max_tokens (500); extraction calls made for deal-review
# prefill use at least this many output tokens.
REVIEW_EXTRACTION_MIN_TOKENS = 1500

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

DEAL_REVIEW_PROMPT_TEMPLATE = """You are reading excerpts from a merger proxy statement (DEFM14A or S-4)
for the acquisition of {target_name}.

Find the "Selected Companies Analysis" (also titled "Comparable Companies
Analysis" or "Selected Public Companies Analysis") presented by the financial
advisor that delivered the fairness opinion to the TARGET's board of
directors, and extract:
1. "advisor": that financial advisor's firm name.
2. "selected_companies": the public company names in that advisor's
   selected-companies list, verbatim, in the order listed.

If more than one advisor presents such an analysis, use the target board's
advisor. If no such analysis appears in the excerpts, use a null advisor and
an empty selected-companies list.

Excerpts (non-contiguous, separated by "..."):
{document_text}"""

DEAL_FINANCIALS_PROMPT_TEMPLATE = """You are reading excerpts from a merger proxy statement for the
acquisition of {target_name}. Find management's financial projections for the
TARGET company (sections titled like "Certain Unaudited Prospective Financial
Information", "Financial Projections", or "Management Projections").

Extract, for the nearest full forecast fiscal year, the TARGET's:
- total revenue, in USD millions
- EBITDA or Adjusted EBITDA, in USD millions

Convert thousands to millions where needed. Use null for anything not
disclosed in the excerpts. Never use the acquirer's figures.

Excerpts (non-contiguous, separated by "..."):
{document_text}

fiscal_year should be a label like "FY2023E"; source_note should be one line
naming the section/table and line items used."""

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


def _search_full_text(
    phrase: str,
    forms: tuple[str, ...],
    max_results: int,
    sic_codes: list[str] | None = None,
) -> list[dict]:
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
        params = {
            "q": f'"{phrase}"',
            "forms": ",".join(forms),
            "from": page * FULL_TEXT_SEARCH_PAGE_SIZE,
        }
        if sic_codes:
            params["sics"] = ",".join(sic_codes)
        try:
            resp = requests.get(
                FULL_TEXT_SEARCH_URL,
                params=params,
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
        for hit in _search_full_text(phrase, FAIRNESS_OPINION_FORMS, max_results, sic_codes=sic_codes):
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


def _extract_relevant_windows(
    text: str,
    keywords: tuple[str, ...] = FAIRNESS_OPINION_KEYWORDS,
    window_before: int = KEYWORD_WINDOW_BEFORE,
    window_after: int = KEYWORD_WINDOW_AFTER,
) -> str:
    """Find every mention of a section keyword and return the text
    around those mentions (merging overlapping windows), instead of a
    blind prefix truncation that can miss the section entirely — verified
    necessary on a real DEF 14A where the relevant section sat past a
    150,000-char cutoff."""
    text_lower = text.lower()
    positions = set()
    for keyword in keywords:
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
        start = max(0, pos - window_before)
        end = min(len(text), pos + window_after)
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))

    combined = "\n...\n".join(text[start:end] for start, end in windows)
    return combined[:MAX_PROMPT_CHARS]


def _extract_selected_companies(client: openai.OpenAI, label: str, document_text: str, config: dict) -> list[str]:
    """
    Structured-outputs extraction (schema-enforced JSON, no hand-rolled
    parsing) — the free-form json.loads() this replaced would occasionally
    fail on a large merged keyword-window prompt (reproduced on a real
    DEFM14A: "Expecting property name enclosed in double quotes"), silently
    dropping every selected company for that filing. See
    src/llm_analyzer._extract_and_judge for the same pattern used by the
    main extraction pipeline.
    """
    relevant_text = _extract_relevant_windows(document_text)
    prompt = FAIRNESS_OPINION_PROMPT_TEMPLATE.format(document_text=relevant_text)

    try:
        result = _call_openai_structured(
            client, config["llm"]["judge_model"], None, prompt,
            config["llm"]["temperature"], config["llm"]["max_tokens"], SelectedCompaniesList,
        )
    except Exception as e:
        logger.warning(f"{label} — selected-companies extraction API call failed: {e}")
        return []

    return [name for name in result.companies if isinstance(name, str) and name.strip()]


def _map_name_to_ticker(label: str, company_name: str) -> str | None:
    """
    Map a selected comp's company name to a ticker via edgartools' fuzzy
    company search (cik/ticker/company/score columns) — find_company gives
    an explicit match-quality score to apply the "reasonable match"
    judgment call against. The name is normalized first (see
    _normalize_company_name) since searching with corporate suffixes
    attached produces wrong high-scoring matches.

    The score alone is not a reliable accept/reject signal near the
    threshold: single-token normalized queries (e.g. "Waystar" from
    "Waystar Holding Corp.") score in the low-to-mid 60s for their own
    exact match, in the same range unrelated companies score for an
    unrelated query (e.g. "Tata Consultancy Services" top-matches "Quanta
    Services, Inc." at 65%). A below-threshold match is accepted anyway
    when the matched company's own name, normalized the same way, is
    identical to the query — that is a much stronger signal than the raw
    score that this is genuinely the same company, not a coincidental
    token overlap with a different one.
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
        exact_normalized_match = _normalize_company_name(top["company"]).casefold() == query.casefold()
        if not exact_normalized_match:
            logger.warning(f"{label} — best match for {company_name!r} too weak ({top['score']:.0f}%), skipping")
            return None
        logger.info(
            f"{label} — accepted below-threshold match for {company_name!r} ({top['score']:.0f}%): "
            f"normalized name is identical to {top['company']!r}"
        )

    return top["ticker"]


def _candidate_identifier(candidate: dict | str) -> int | str:
    if isinstance(candidate, str):
        return candidate
    return candidate.get("cik") or candidate["ticker"]


def _candidate_label(candidate: dict | str) -> str:
    if isinstance(candidate, str):
        return candidate
    return candidate.get("ticker") or f"CIK{candidate.get('cik')}"


def _selected_company_reviews(label: str, selected_names: list[str]) -> list[dict]:
    reviews = []
    for name in selected_names:
        reviews.append({
            "company_name": name,
            "suggested_ticker": _map_name_to_ticker(label, name),
            "still_public": None,
            "include_in_ground_truth": None,
            "review_status": "needs_review",
        })
    return reviews


def _target_review(candidate: dict | str, identifier: int | str, label: str) -> dict:
    if isinstance(candidate, str):
        return {
            "identifier": str(identifier),
            "label": label,
            "ticker": candidate,
            "cik": None,
            "company_name": None,
            "sic": None,
            "source_candidate": None,
        }
    return {
        "identifier": str(identifier),
        "label": label,
        "ticker": candidate.get("ticker"),
        "cik": candidate.get("cik"),
        "company_name": candidate.get("company_name"),
        "sic": candidate.get("sic"),
        "source_candidate": {
            "file_date": candidate.get("file_date"),
            "form": candidate.get("form"),
        },
    }


def build_manual_deal_review(
    test_candidates: list[dict | str],
    config: dict,
    output_path: Path = MANUAL_DEAL_REVIEW_PATH,
) -> list[dict]:
    """
    Prefill a human-review JSON file from fairness-opinion candidates.

    The output is intentionally not the final manual_deals.json benchmark:
    reviewers must confirm tickers, public-status flags, and inclusion before
    copying approved deals into the manual ground-truth file.
    """
    edgar.set_identity(EDGAR_IDENTITY)
    client = openai.OpenAI()
    reviews = []

    for candidate in test_candidates:
        identifier = _candidate_identifier(candidate)
        label = _candidate_label(candidate)
        try:
            document_text, form_used = _fetch_fairness_opinion_text(identifier)
        except Exception as e:
            logger.warning(f"{label} — failed to fetch fairness opinion filing: {e}")
            continue
        if document_text is None:
            logger.warning(f"{label} — no {'/'.join(FAIRNESS_OPINION_FORMS)} filing found")
            continue

        selected_names = _extract_selected_companies(client, label, document_text, config)
        reviews.append({
            "review_status": "needs_review",
            "manual_fields_to_confirm": MANUAL_REVIEW_FIELDS,
            "target": _target_review(candidate, identifier, label),
            "filing": {"form_used": form_used},
            "selected_companies": _selected_company_reviews(label, selected_names),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2)
    return reviews


def _parse_filing_url(url: str) -> tuple[int, str]:
    """CIK and dashed accession number from an EDGAR archive document URL,
    e.g. .../Archives/edgar/data/1091883/000114036123034666/xyz_defm14a.htm
    -> (1091883, "0001140361-23-034666")."""
    match = FILING_URL_RE.search(url)
    if not match:
        raise ValueError(
            f"Not an EDGAR filing document URL (expected /Archives/edgar/data/<cik>/<accession>/…): {url}"
        )
    cik = int(match.group(1))
    raw = match.group(2)
    return cik, f"{raw[:10]}-{raw[10:12]}-{raw[12:]}"


def _html_to_text(document: str) -> str:
    """Plain text from a filing HTML document — good enough for keyword-window
    extraction and LLM prompts; not a layout-faithful render."""
    text = _SCRIPT_STYLE_RE.sub(" ", document)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html_lib.unescape(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _fetch_filing_document(url: str) -> str:
    """The exact filing document at `url`, as plain text. Unlike
    _fetch_fairness_opinion_text (which takes a company's *latest* DEFM14A/S-4),
    this pins the specific document a human chose, and skips the
    DOCUMENT_MAX_CHARS prefix truncation — window extraction bounds the prompt
    instead, so a section past the 200k mark isn't silently lost."""
    resp = requests.get(url, headers=FULL_TEXT_SEARCH_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS * 3)
    resp.raise_for_status()
    return _html_to_text(resp.text)


def _filing_metadata(cik: int, accession: str) -> dict:
    """Target name/SIC/tickers plus this filing's date and form type, from the
    SEC submissions profile. filing_date/form are None when the accession has
    aged out of the profile's ~1000-filing recent window — the reviewer fills
    them from the document header in that case."""
    profile = sic_universe_builder.fetch_company_profile(cik) or {}
    recent = (profile.get("filings") or {}).get("recent") or {}
    accessions = recent.get("accessionNumber") or []
    filing_date = None
    form = None
    if accession in accessions:
        idx = accessions.index(accession)
        dates = recent.get("filingDate") or []
        forms = recent.get("form") or []
        filing_date = dates[idx] if idx < len(dates) else None
        form = forms[idx] if idx < len(forms) else None
    return {
        "company_name": profile.get("name"),
        "sic": str(profile["sic"]) if profile.get("sic") else None,
        "sic_description": profile.get("sicDescription"),
        "tickers": [t for t in profile.get("tickers", []) if t],
        "filing_date": filing_date,
        "form": form,
    }


TARGET_DESCRIPTION_MAX_CHARS = 500


def _fetch_target_description(cik: int) -> str | None:
    """Item 1 Business text from the target's last 10-K on file (EDGAR keeps
    these for delisted companies). Prefill only — the reviewer trims it."""
    try:
        filing = edgar.Company(cik).get_filings(form="10-K").latest()
        if filing is None:
            return None
        text = getattr(filing.obj(), "business", None)
    except Exception as e:
        logger.warning(f"CIK{cik} — could not fetch 10-K business description: {e}")
        return None
    return text[:TARGET_DESCRIPTION_MAX_CHARS] if text else None


_CURRENT_US_LISTINGS: dict[str, int] | None = None


def _current_us_listings() -> dict[str, int]:
    """{ticker: cik} for every currently SEC-listed company, from the same
    company_tickers.json that sic_universe_builder uses — fetched once per
    process. A ticker absent from this map is delisted (or foreign-only)."""
    global _CURRENT_US_LISTINGS
    if _CURRENT_US_LISTINGS is None:
        resp = requests.get(
            sic_universe_builder.COMPANY_TICKERS_URL,
            headers=FULL_TEXT_SEARCH_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        _CURRENT_US_LISTINGS = {
            str(row.get("ticker") or "").strip().upper(): int(row["cik_str"])
            for row in resp.json().values()
            if row.get("ticker")
        }
    return _CURRENT_US_LISTINGS


def _comp_listing_status(ticker: str | None) -> tuple[bool | None, bool | None]:
    """(still_public, us_filer) prefill for a suggested comp ticker.
    still_public: has a current US listing. us_filer: files 10-K (vs a foreign
    private issuer's 20-F) — the flag that decides whether the comp can enter
    the benchmark denominator at all. Both are suggestions for the reviewer,
    None when undeterminable."""
    if not ticker:
        return None, None
    try:
        cik = _current_us_listings().get(ticker.strip().upper())
    except Exception as e:
        logger.warning(f"{ticker} — current-listings lookup failed: {e}")
        return None, None
    if cik is None:
        return False, False
    try:
        time.sleep(SEC_REQUEST_DELAY_SECONDS)
        profile = sic_universe_builder.fetch_company_profile(cik) or {}
    except Exception as e:
        logger.warning(f"{ticker} — submissions profile lookup failed: {e}")
        return True, None
    forms = (profile.get("filings") or {}).get("recent", {}).get("form") or []
    return True, any(str(f).startswith("10-K") for f in forms)


def _review_max_tokens(config: dict) -> int:
    return max(int(config["llm"].get("max_tokens", 500)), REVIEW_EXTRACTION_MIN_TOKENS)


def _extract_advisor_and_companies(client: openai.OpenAI, label: str, target_name: str,
                                   document_text: str, config: dict) -> tuple[str | None, list[str]]:
    """
    Structured-outputs extraction — this is the URL-driven prefill path
    (scripts.prefill_manual_deal -> build_manual_deal_review_from_urls), and
    the free-form json.loads() this replaced could fail on the large merged
    keyword-window prompt (reproduced on a real DEFM14A: "Expecting property
    name enclosed in double quotes"), silently returning no advisor and no
    selected companies for that filing.
    """
    relevant_text = _extract_relevant_windows(document_text)
    prompt = DEAL_REVIEW_PROMPT_TEMPLATE.format(target_name=target_name, document_text=relevant_text)
    try:
        result = _call_openai_structured(
            client, config["llm"]["extraction_model"], None, prompt,
            config["llm"]["temperature"], _review_max_tokens(config), AdvisorAndSelectedCompanies,
        )
    except Exception as e:
        logger.warning(f"{label} — advisor/selected-companies extraction failed: {e}")
        return None, []

    advisor = result.advisor.strip() if result.advisor and result.advisor.strip() else None
    names = [n for n in result.selected_companies if isinstance(n, str) and n.strip()]
    return advisor, names


def _extract_target_financials(client: openai.OpenAI, label: str, target_name: str,
                               document_text: str, config: dict) -> dict:
    """Structured-outputs extraction — see _extract_advisor_and_companies for
    why (same prompt-fragility class, same URL-driven prefill path)."""
    relevant_text = _extract_relevant_windows(
        document_text, keywords=PROJECTION_KEYWORDS, window_after=PROJECTION_WINDOW_AFTER,
    )
    prompt = DEAL_FINANCIALS_PROMPT_TEMPLATE.format(target_name=target_name, document_text=relevant_text)
    empty = {"fiscal_year": None, "revenue_usd_mm": None, "ebitda_usd_mm": None, "source_note": None}
    try:
        result = _call_openai_structured(
            client, config["llm"]["extraction_model"], None, prompt,
            config["llm"]["temperature"], _review_max_tokens(config), DealFinancials,
        )
    except Exception as e:
        logger.warning(f"{label} — target-financials extraction failed: {e}")
        return empty

    return result.model_dump()


def _ebitda_margin(financials: dict) -> float | None:
    revenue = financials.get("revenue_usd_mm")
    ebitda = financials.get("ebitda_usd_mm")
    if not revenue or ebitda is None:
        return None
    return round(ebitda / revenue, 3)


def _deal_slug(company_name: str | None, cik: int, filing_date: str | None) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", (company_name or f"cik{cik}").lower()).strip("-")
    year = (filing_date or "")[:4] or "unknown"
    return f"{base}-{year}"


def _suggested_manual_deal(url: str, cik: int, metadata: dict, advisor: str | None,
                           financials: dict, description: str | None,
                           selected_companies: list[dict]) -> dict:
    """A manual_deals.json-shaped entry the reviewer can copy across once every
    field is verified — review_status stays needs_review until then."""
    source_bits = [b for b in (financials.get("fiscal_year"), financials.get("source_note")) if b]
    return {
        "deal_id": _deal_slug(metadata.get("company_name"), cik, metadata.get("filing_date")),
        "target_ticker": (metadata.get("tickers") or [None])[0] or f"CIK{cik}",
        "target_name": metadata.get("company_name"),
        "target_cik": f"{cik:010d}",
        "target_sic": metadata.get("sic"),
        "business_description": description,
        "target_financials": {
            "revenue_usd_mm": financials.get("revenue_usd_mm"),
            "ebitda_margin_estimate": _ebitda_margin(financials),
            "source": "; ".join(source_bits) + " — LLM-prefilled, verify against the filing" if source_bits else None,
        },
        "filing_url": url,
        "filing_date": metadata.get("filing_date"),
        "advisor": advisor,
        "selected_companies": [
            {
                "company_name": comp["company_name"],
                "ticker": comp.get("suggested_ticker"),
                "still_public": comp.get("still_public"),
                "us_filer": comp.get("us_filer"),
            }
            for comp in selected_companies
        ],
        "review_status": "needs_review",
        "notes": "Auto-prefilled by build_manual_deal_review_from_urls; verify every field against the filing before moving into manual_deals.json.",
    }


def _merge_review_entries(existing: list[dict], new_entries: list[dict]) -> list[dict]:
    """Replace by filing URL, keep everything else (including candidate-flow
    entries, which have no filing.url) in place, append the rest."""
    new_by_url = {e["filing"]["url"]: e for e in new_entries}
    merged = []
    for entry in existing:
        url = (entry.get("filing") or {}).get("url")
        if url in new_by_url:
            merged.append(new_by_url.pop(url))
        else:
            merged.append(entry)
    merged.extend(new_by_url.values())
    return merged


def build_manual_deal_review_from_urls(
    filing_urls: list[str],
    config: dict,
    output_path: Path = MANUAL_DEAL_REVIEW_PATH,
) -> list[dict]:
    """
    Prefill review entries from specific DEFM14A/S-4 document URLs (the flow
    for a human who has already located the exact proxy on EDGAR full-text
    search), extracting everything the manual_deals.json schema needs: target
    metadata (name/SIC/CIK from SEC submissions), filing date/form, the fairness
    advisor and its selected-companies list, the target's projected
    revenue/EBITDA, a 10-K business-description stub, and per-comp
    still_public / us_filer suggestions.

    Entries are merged into `output_path` by filing URL (existing entries for
    other filings are preserved) and each carries a `suggested_manual_deal`
    block shaped exactly like a manual_deals.json record. Everything remains
    review_status="needs_review" until a human verifies it against the filing.
    """
    edgar.set_identity(EDGAR_IDENTITY)
    client = openai.OpenAI()
    new_entries = []

    for url in filing_urls:
        cik, accession = _parse_filing_url(url)
        metadata = _filing_metadata(cik, accession)
        label = (metadata.get("tickers") or [None])[0] or f"CIK{cik}"
        target_name = metadata.get("company_name") or label
        logger.info(f"{label} — prefilling deal review from {url}")

        document_text = _fetch_filing_document(url)
        advisor, selected_names = _extract_advisor_and_companies(client, label, target_name, document_text, config)
        financials = _extract_target_financials(client, label, target_name, document_text, config)
        description = _fetch_target_description(cik)

        selected_companies = []
        for name in selected_names:
            suggested_ticker = _map_name_to_ticker(label, name)
            still_public, us_filer = _comp_listing_status(suggested_ticker)
            selected_companies.append({
                "company_name": name,
                "suggested_ticker": suggested_ticker,
                "still_public": still_public,
                "us_filer": us_filer,
                "include_in_ground_truth": None,
                "review_status": "needs_review",
            })

        new_entries.append({
            "review_status": "needs_review",
            "manual_fields_to_confirm": MANUAL_REVIEW_FIELDS,
            "target": {
                "identifier": str(cik),
                "label": label,
                "ticker": (metadata.get("tickers") or [None])[0],
                "cik": cik,
                "company_name": metadata.get("company_name"),
                "sic": metadata.get("sic"),
                "sic_description": metadata.get("sic_description"),
                "source_candidate": None,
            },
            "filing": {
                "url": url,
                "accession": accession,
                "form_used": metadata.get("form"),
                "filing_date": metadata.get("filing_date"),
            },
            "advisor": advisor,
            "target_financials": financials,
            "business_description": description,
            "selected_companies": selected_companies,
            "suggested_manual_deal": _suggested_manual_deal(
                url, cik, metadata, advisor, financials, description, selected_companies,
            ),
        })

    existing = []
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
    merged = _merge_review_entries(existing, new_entries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    return new_entries


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
