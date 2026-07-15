import json
import re
from datetime import datetime, timezone
from pathlib import Path

import edgar
import openai

from src import get_logger
from src.llm_analyzer import _call_openai, _strip_markdown_fences

logger = get_logger(__name__)

CACHE_DIR = Path("data/cache")
EDGAR_IDENTITY = "PE-Comps-Pipeline research@example.com"

DOCUMENT_MAX_CHARS = 200_000
PROMPT_DOCUMENT_CHARS = 150_000
MIN_NAME_MATCH_SCORE = 70.0

# Blind truncation to the first PROMPT_DOCUMENT_CHARS misses the peer group
# entirely in some filings — verified on MMM's 2026 DEF 14A, where the
# actual peer list sits around char 187,000, well past a 150,000 cutoff.
# Proxy statements bury executive-comp peer discussion deep behind pages of
# governance/director-bio content, so instead of guessing a cutoff we search
# for mentions of the keywords and only send the LLM the text around them.
PEER_GROUP_KEYWORDS = ("peer group", "peer companies", "compensation peer", "benchmarking peer")
KEYWORD_WINDOW_BEFORE = 300
KEYWORD_WINDOW_AFTER = 3000
MAX_PROMPT_CHARS = 60_000

# edgar.find_company() does token-overlap fuzzy matching, which gets fooled
# by generic corporate-suffix words: "IDEX Corporation" scored a 84% "match"
# against "IHI Corporation" (wrong company) while the real IDEX Corp (ticker
# IEX) didn't even appear in the top 10 results. Searching the bare brand
# name instead ("IDEX") finds it at 100%. Verified the same pattern for
# Dover, Hubbell, TransDigm, Corning, Johnson & Johnson, Snap-on, Colgate-
# Palmolive, Kimberly-Clark — stripping suffixes/punctuation before the
# search consistently fixes it.
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

PEER_GROUP_PROMPT_TEMPLATE = """You are extracting a compensation peer group from a proxy statement (DEF 14A).

Find the section called "Compensation Peer Group", "Peer Group",
"Benchmarking Peer Group", or similar. Extract the list of company names
in that peer group.

Document text (may be truncated):
{document_text}

Return ONLY a JSON array of company name strings.
Example: ["Company A", "Company B", "Company C"]
If you cannot find a peer group list, return an empty array: []
Do not include any other text."""

TEST_TICKERS = [
    "MMM",   # 3M — diversified industrials
    "HON",   # Honeywell
    "EMR",   # Emerson Electric
    "ITW",   # Illinois Tool Works
    "PH",    # Parker Hannifin
    "AME",   # AMETEK
    "IEX",   # IDEX Corporation
    "RRX",   # Rexnord
    "XYL",   # Xylem
    "GNRC",  # Generac Holdings
    "MDT",   # Medtronic — healthcare
    "SYK",   # Stryker
    "ZBH",   # Zimmer Biomet
    "EW",    # Edwards Lifesciences
    "NTAP",  # NetApp — tech hardware
]


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"peer_group_{ticker}.json"


def _load_cache(ticker: str) -> dict | None:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache(ticker: str, record: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(ticker), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def _fetch_def14a_text(ticker: str) -> str | None:
    """
    Latest DEF 14A full text via edgartools, rather than the raw EDGAR
    full-text-search API the spec describes — that endpoint does free-text
    relevance search, not exact ticker lookup, so a quoted ticker can match
    an unrelated filing that merely mentions it somewhere (verified
    empirically: searching `"MMM"` surfaced a Cousins Properties filing).
    edgartools' Company(ticker).get_filings() resolves the ticker properly.
    """
    company = edgar.Company(ticker)
    filing = company.get_filings(form="DEF 14A").latest()
    if filing is None:
        return None
    return filing.text()[:DOCUMENT_MAX_CHARS]


def _extract_relevant_windows(text: str) -> str:
    """Find every mention of a peer-group keyword and return the text
    around those mentions (merging overlapping windows), instead of a
    blind prefix truncation that can miss the section entirely."""
    text_lower = text.lower()
    positions = set()
    for keyword in PEER_GROUP_KEYWORDS:
        start = 0
        while True:
            idx = text_lower.find(keyword, start)
            if idx == -1:
                break
            positions.add(idx)
            start = idx + len(keyword)

    if not positions:
        return text[:PROMPT_DOCUMENT_CHARS]

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


def _parse_json_array(text: str | None, ticker: str) -> list | None:
    """Like llm_analyzer._parse_json_response, but for prompts that return a
    JSON array rather than an object — that function hard-requires a dict,
    which would reject every valid peer-group response here."""
    if not text:
        return None
    cleaned = _strip_markdown_fences(text)
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"{ticker} — failed to parse LLM JSON response: {e}")
        return None
    if not isinstance(parsed, list):
        logger.warning(f"{ticker} — LLM JSON response was not an array: {cleaned!r}")
        return None
    return parsed


def _extract_peer_company_names(client: openai.OpenAI, ticker: str, document_text: str, config: dict) -> list[str]:
    relevant_text = _extract_relevant_windows(document_text)
    prompt = PEER_GROUP_PROMPT_TEMPLATE.format(document_text=relevant_text)

    try:
        response_text = _call_openai(
            client, config["llm"]["judge_model"], None, prompt,
            config["llm"]["temperature"], config["llm"]["max_tokens"],
        )
    except Exception as e:
        logger.warning(f"{ticker} — peer group extraction API call failed: {e}")
        return []

    parsed = _parse_json_array(response_text, ticker)
    if parsed is None:
        logger.warning(f"{ticker} — peer group extraction did not return a JSON array")
        return []

    return [name for name in parsed if isinstance(name, str)]


def _map_name_to_ticker(ticker: str, company_name: str) -> str | None:
    """
    Map a peer's company name to a ticker via edgartools' fuzzy company
    search (cik/ticker/company/score columns) rather than the spec's raw
    full-text-search approach — same reliability problem as fetching the
    DEF 14A itself, and find_company gives an explicit match-quality score
    to apply the "reasonable match" judgment call against. The name is
    normalized first (see _normalize_company_name) since searching with
    corporate suffixes attached produces wrong high-scoring matches.
    """
    query = _normalize_company_name(company_name)
    try:
        results = edgar.find_company(query)
    except Exception as e:
        logger.warning(f"{ticker} — name lookup failed for {company_name!r}: {e}")
        return None

    if results.results is None or results.results.empty:
        logger.warning(f"{ticker} — could not map {company_name!r} to a ticker")
        return None

    top = results.results.iloc[0]
    if top["score"] < MIN_NAME_MATCH_SCORE:
        logger.warning(f"{ticker} — best match for {company_name!r} too weak ({top['score']:.0f}%), skipping")
        return None

    return top["ticker"]


def build_ground_truth(test_tickers: list[str], config: dict) -> dict[str, list[str]]:
    """
    For each test ticker, extract its compensation peer group from SEC EDGAR.

    Returns:
        dict mapping ticker -> list of peer tickers

    Companies whose peer group could not be extracted are omitted from results.
    Caches results to data/cache/peer_group_{ticker}.json.
    """
    edgar.set_identity(EDGAR_IDENTITY)
    client = openai.OpenAI()
    ground_truth = {}

    for ticker in test_tickers:
        cached = _load_cache(ticker)
        if cached is not None:
            logger.info(f"{ticker} — peer group loaded from cache")
            if cached.get("peer_group_tickers"):
                ground_truth[ticker] = cached["peer_group_tickers"]
            continue

        try:
            document_text = _fetch_def14a_text(ticker)
        except Exception as e:
            logger.warning(f"{ticker} — failed to fetch DEF 14A: {e}")
            continue

        if document_text is None:
            logger.warning(f"{ticker} — no DEF 14A filing found")
            continue

        peer_names = _extract_peer_company_names(client, ticker, document_text, config)

        peer_tickers = []
        unmapped = []
        for name in peer_names:
            mapped = _map_name_to_ticker(ticker, name)
            if mapped:
                peer_tickers.append(mapped)
            else:
                unmapped.append(name)

        if unmapped:
            logger.warning(f"{ticker} — could not map {len(unmapped)} peer names to tickers: {unmapped}")

        record = {
            "ticker": ticker,
            "peer_group_tickers": peer_tickers,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_cache(ticker, record)

        if peer_tickers:
            ground_truth[ticker] = peer_tickers

    return ground_truth
