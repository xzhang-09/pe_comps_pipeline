"""Tests for two prefill bugs fixed together: free-form JSON parsing in the
ground-truth extractors, and the name-to-ticker matcher's below-threshold
false rejections.

Three extractors shared the fragile _call_openai + hand-rolled-json.loads
pattern; all three are covered here since they're the same failure class,
even though the originally reported failure (reproduced on Squarespace's
DEFM14A) turned out to be in _extract_advisor_and_companies — the URL-driven
prefill path (scripts.prefill_manual_deal) — not _extract_selected_companies,
which is a separate, ticker/CIK-candidate-driven path that also had the bug
but wasn't the one observed."""
import pandas as pd

import eval.ground_truth_builder as ground_truth_builder
from src.llm_schemas import AdvisorAndSelectedCompanies, DealFinancials, SelectedCompaniesList


def test_extract_selected_companies_uses_structured_outputs(mocker):
    mocker.patch(
        "eval.ground_truth_builder._call_openai_structured",
        return_value=SelectedCompaniesList(companies=["BigCommerce Holdings, Inc.", "Wix.com Ltd.", ""]),
    )

    result = ground_truth_builder._extract_selected_companies(
        client=mocker.MagicMock(), label="CIK123", document_text="some filing text",
        config={"llm": {"judge_model": "gpt-4.1-mini", "temperature": 0, "max_tokens": 500}},
    )

    assert result == ["BigCommerce Holdings, Inc.", "Wix.com Ltd."]  # blank name dropped


def test_extract_selected_companies_returns_empty_on_api_failure(mocker):
    mocker.patch(
        "eval.ground_truth_builder._call_openai_structured",
        side_effect=RuntimeError("API down"),
    )

    result = ground_truth_builder._extract_selected_companies(
        client=mocker.MagicMock(), label="CIK123", document_text="some filing text",
        config={"llm": {"judge_model": "gpt-4.1-mini", "temperature": 0, "max_tokens": 500}},
    )

    assert result == []


def test_extract_advisor_and_companies_uses_structured_outputs(mocker):
    """This is the extractor on the URL-driven prefill path
    (scripts.prefill_manual_deal -> build_manual_deal_review_from_urls) where
    the free-form JSON parse failure was actually reproduced (Squarespace's
    DEFM14A: 'Expecting property name enclosed in double quotes'), silently
    returning no advisor and no selected companies for that filing."""
    mocker.patch(
        "eval.ground_truth_builder._call_openai_structured",
        return_value=AdvisorAndSelectedCompanies(
            advisor="Centerview Partners LLC",
            selected_companies=["GoDaddy Inc.", "  ", "VeriSign, Inc."],
        ),
    )

    advisor, names = ground_truth_builder._extract_advisor_and_companies(
        client=mocker.MagicMock(), label="CIK1496963", target_name="Squarespace, Inc.",
        document_text="some filing text",
        config={"llm": {"extraction_model": "gpt-4.1", "temperature": 0, "max_tokens": 500}},
    )

    assert advisor == "Centerview Partners LLC"
    assert names == ["GoDaddy Inc.", "VeriSign, Inc."]  # blank name dropped


def test_extract_advisor_and_companies_returns_none_and_empty_on_api_failure(mocker):
    mocker.patch(
        "eval.ground_truth_builder._call_openai_structured",
        side_effect=RuntimeError("API down"),
    )

    advisor, names = ground_truth_builder._extract_advisor_and_companies(
        client=mocker.MagicMock(), label="CIK123", target_name="Target Inc.",
        document_text="some filing text",
        config={"llm": {"extraction_model": "gpt-4.1", "temperature": 0, "max_tokens": 500}},
    )

    assert advisor is None
    assert names == []


def test_extract_target_financials_uses_structured_outputs(mocker):
    mocker.patch(
        "eval.ground_truth_builder._call_openai_structured",
        return_value=DealFinancials(
            fiscal_year="FY2025E", revenue_usd_mm=1374.0, ebitda_usd_mm=371.0,
            source_note="Certain Financial Projections table",
        ),
    )

    financials = ground_truth_builder._extract_target_financials(
        client=mocker.MagicMock(), label="CIK1496963", target_name="Squarespace, Inc.",
        document_text="some filing text",
        config={"llm": {"extraction_model": "gpt-4.1", "temperature": 0, "max_tokens": 500}},
    )

    assert financials == {
        "fiscal_year": "FY2025E", "revenue_usd_mm": 1374.0, "ebitda_usd_mm": 371.0,
        "source_note": "Certain Financial Projections table",
    }


def test_extract_target_financials_returns_empty_dict_on_api_failure(mocker):
    mocker.patch(
        "eval.ground_truth_builder._call_openai_structured",
        side_effect=RuntimeError("API down"),
    )

    financials = ground_truth_builder._extract_target_financials(
        client=mocker.MagicMock(), label="CIK123", target_name="Target Inc.",
        document_text="some filing text",
        config={"llm": {"extraction_model": "gpt-4.1", "temperature": 0, "max_tokens": 500}},
    )

    assert financials == {"fiscal_year": None, "revenue_usd_mm": None, "ebitda_usd_mm": None, "source_note": None}


def _fake_find_company_results(rows: list[dict]):
    fake = type("FakeFindCompanyResults", (), {})()
    fake.results = pd.DataFrame(rows) if rows else None
    return fake


def test_map_name_to_ticker_accepts_below_threshold_exact_normalized_match(mocker):
    """Waystar Holding Corp. (score 63.6, below the 70 threshold) is the
    correct match — edgar's own top hit company name, once normalized the
    same way as the query, is identical to the query."""
    mocker.patch(
        "eval.ground_truth_builder.edgar.find_company",
        return_value=_fake_find_company_results(
            [{"ticker": "WAY", "company": "Waystar Holding Corp.", "score": 63.6}]
        ),
    )

    ticker = ground_truth_builder._map_name_to_ticker("deal-1", "Waystar Holding Corp.")

    assert ticker == "WAY"


def test_map_name_to_ticker_rejects_below_threshold_unrelated_company(mocker):
    """Tata Consultancy Services Limited (score 65.0) top-matches an
    unrelated company (Quanta Services) at a similar score — the exact-name
    fallback must not accept this; scores alone cannot distinguish the two
    cases, so the fallback is name-identity, not a lower cutoff."""
    mocker.patch(
        "eval.ground_truth_builder.edgar.find_company",
        return_value=_fake_find_company_results(
            [{"ticker": "PWR", "company": "QUANTA SERVICES, INC.", "score": 65.0}]
        ),
    )

    ticker = ground_truth_builder._map_name_to_ticker("deal-1", "Tata Consultancy Services Limited")

    assert ticker is None


def test_map_name_to_ticker_accepts_above_threshold_as_before(mocker):
    mocker.patch(
        "eval.ground_truth_builder.edgar.find_company",
        return_value=_fake_find_company_results(
            [{"ticker": "MNTX", "company": "Manitex International, Inc.", "score": 95.0}]
        ),
    )

    ticker = ground_truth_builder._map_name_to_ticker("deal-1", "Manitex International, Inc.")

    assert ticker == "MNTX"


def test_map_name_to_ticker_returns_none_on_empty_results(mocker):
    mocker.patch(
        "eval.ground_truth_builder.edgar.find_company",
        return_value=_fake_find_company_results([]),
    )

    ticker = ground_truth_builder._map_name_to_ticker("deal-1", "Nonexistent Corp")

    assert ticker is None
