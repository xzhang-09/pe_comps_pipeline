"""Regression test for a projections-table extraction bug reproduced on
Squarespace's DEFM14A (https://www.sec.gov/Archives/edgar/data/1496963/000114036124038138/ny20030653x9_defm14a.htm).

PROJECTION_KEYWORDS matched fine (the filing's "Unaudited Prospective
Financial Information" defined term contains "prospective financial
information"), but the nearest keyword match before the actual numbers
table sat ~5,200 characters before the table's Total Revenue/Adjusted
EBITDA rows. With the old 4,000-char PROJECTION_WINDOW_AFTER, that
window closed 61 characters into the table (right after the header, no
numbers) and the following keyword match's window didn't reach back far
enough to cover the gap — so the real table was never in the prompt at
all, and the model filled the gap with numbers from elsewhere in the
document (a garbled $1.0mm revenue / $240mm EBITDA) instead of the real
$1,216mm / $291mm, or an honest null.
"""
import eval.ground_truth_builder as ground_truth_builder
from eval.ground_truth_builder import (
    KEYWORD_WINDOW_BEFORE,
    PROJECTION_KEYWORDS,
    PROJECTION_MAX_PROMPT_CHARS,
    PROJECTION_WINDOW_AFTER,
    _extract_relevant_windows,
)


def _synthetic_filing_with_distant_table() -> str:
    """A keyword match, then a gap wide enough that a 4,000-char
    after-window misses the table, then the table, then a second keyword
    match placed so its 300-char before-window can't reach back to the
    table either — mirroring the real filing's keyword-to-table spacing."""
    keyword = "prospective financial information"
    gap_before_table = "x" * 5300
    table = "Total Revenue $ 1,216 ... Adjusted EBITDA $ 291 ... end of table"
    gap_after_table = "y" * 1000
    second_keyword = "financial projections"
    trailer = "z" * 500
    return keyword + gap_before_table + table + gap_after_table + second_keyword + trailer


def test_old_window_after_missed_the_table():
    """Documents the failure mode: the pre-fix 4,000-char window drops the
    table entirely, from either neighboring keyword match."""
    text = _synthetic_filing_with_distant_table()

    result = _extract_relevant_windows(
        text, keywords=PROJECTION_KEYWORDS, window_before=KEYWORD_WINDOW_BEFORE,
        window_after=4000, max_chars=PROJECTION_MAX_PROMPT_CHARS,
    )

    assert "Total Revenue" not in result
    assert "Adjusted EBITDA" not in result


def test_projection_window_after_reaches_the_table():
    text = _synthetic_filing_with_distant_table()

    result = _extract_relevant_windows(
        text, keywords=PROJECTION_KEYWORDS, window_before=KEYWORD_WINDOW_BEFORE,
        window_after=PROJECTION_WINDOW_AFTER, max_chars=PROJECTION_MAX_PROMPT_CHARS,
    )

    assert "Total Revenue $ 1,216" in result
    assert "Adjusted EBITDA $ 291" in result


def test_extract_target_financials_passes_projection_window_settings(mocker):
    """_extract_target_financials must use the wider window/budget constants
    (not the FAIRNESS_OPINION_KEYWORDS defaults), or this bug regresses
    silently since the function signature still accepts narrower values."""
    captured = {}

    def fake_extract_relevant_windows(text, **kwargs):
        captured.update(kwargs)
        return "windowed text"

    mocker.patch(
        "eval.ground_truth_builder._extract_relevant_windows",
        side_effect=fake_extract_relevant_windows,
    )
    mocker.patch(
        "eval.ground_truth_builder._call_openai_structured",
        return_value=mocker.MagicMock(model_dump=lambda: {}),
    )

    ground_truth_builder._extract_target_financials(
        client=mocker.MagicMock(), label="CIK1496963", target_name="Squarespace, Inc.",
        document_text="some filing text",
        config={"llm": {"extraction_model": "gpt-4.1", "temperature": 0, "max_tokens": 500}},
    )

    assert captured["window_after"] == PROJECTION_WINDOW_AFTER
    assert captured["max_chars"] == PROJECTION_MAX_PROMPT_CHARS
