import json

import eval.ground_truth_builder as ground_truth_builder


class _FakeResponse:
    def __init__(self, json_data=None, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json


def _full_text_hit(cik, company_name="Test Co", file_date="2024-01-01", form="DEFM14A"):
    return {
        "_source": {
            "ciks": [str(cik)],
            "display_names": [f"{company_name} (CIK {cik})"],
            "file_date": file_date,
            "form": form,
        },
    }


def test_search_full_text_parses_hits(mocker):
    mocker.patch(
        "eval.ground_truth_builder.requests.get",
        return_value=_FakeResponse(json_data={"hits": {"hits": [_full_text_hit(1), _full_text_hit(2)]}}),
    )

    results = ground_truth_builder._search_full_text("selected companies analysis", ("DEFM14A",), max_results=20)

    assert len(results) == 2
    assert results[0]["cik"] == 1
    assert results[0]["form"] == "DEFM14A"
    assert results[0]["file_date"] == "2024-01-01"


def test_search_full_text_stops_on_partial_page(mocker):
    mock_get = mocker.patch(
        "eval.ground_truth_builder.requests.get",
        return_value=_FakeResponse(json_data={"hits": {"hits": [_full_text_hit(1)]}}),
    )

    ground_truth_builder._search_full_text("x", ("DEFM14A",), max_results=20)

    assert mock_get.call_count == 1


def test_search_full_text_returns_empty_on_failure(mocker):
    mocker.patch("eval.ground_truth_builder.requests.get", side_effect=Exception("network down"))

    results = ground_truth_builder._search_full_text("x", ("DEFM14A",), max_results=20)

    assert results == []


def test_search_full_text_skips_hits_without_cik(mocker):
    hit = _full_text_hit(1)
    hit["_source"]["ciks"] = []
    mocker.patch(
        "eval.ground_truth_builder.requests.get",
        return_value=_FakeResponse(json_data={"hits": {"hits": [hit]}}),
    )

    results = ground_truth_builder._search_full_text("x", ("DEFM14A",), max_results=20)

    assert results == []


def test_search_full_text_passes_sic_filter(mocker):
    mock_get = mocker.patch(
        "eval.ground_truth_builder.requests.get",
        return_value=_FakeResponse(json_data={"hits": {"hits": []}}),
    )

    ground_truth_builder._search_full_text("x", ("DEFM14A",), max_results=20, sic_codes=["3559", "3569"])

    assert mock_get.call_args.kwargs["params"]["sics"] == "3559,3569"


def test_discover_fairness_opinion_candidates_filters_by_sic(mocker):
    mocker.patch(
        "eval.ground_truth_builder._search_full_text",
        return_value=[
            {"cik": 1, "company_name": "Match Co", "file_date": "2024-01-01", "form": "DEFM14A"},
            {"cik": 2, "company_name": "NoMatch Co", "file_date": "2024-01-01", "form": "DEFM14A"},
        ],
    )
    mocker.patch(
        "eval.ground_truth_builder.sic_universe_builder.fetch_company_profile",
        side_effect=lambda cik: (
            {"tickers": ["AAA"], "sic": "3559", "name": "Match Co"} if cik == 1
            else {"tickers": ["BBB"], "sic": "9999", "name": "NoMatch Co"}
        ),
    )

    candidates = ground_truth_builder.discover_fairness_opinion_candidates(["3559"])

    assert len(candidates) == 1
    assert candidates[0]["cik"] == 1
    assert candidates[0]["ticker"] == "AAA"


def test_discover_fairness_opinion_candidates_dedupes_across_phrases(mocker):
    mocker.patch(
        "eval.ground_truth_builder._search_full_text",
        return_value=[{"cik": 1, "company_name": "Match Co", "file_date": "2024-01-01", "form": "DEFM14A"}],
    )
    mock_profile = mocker.patch(
        "eval.ground_truth_builder.sic_universe_builder.fetch_company_profile",
        return_value={"tickers": ["AAA"], "sic": "3559", "name": "Match Co"},
    )

    candidates = ground_truth_builder.discover_fairness_opinion_candidates(["3559"])

    assert len(candidates) == 1
    # one profile lookup per unique CIK, not once per keyword phrase
    assert mock_profile.call_count == 1


def test_discover_fairness_opinion_candidates_handles_no_ticker(mocker):
    mocker.patch(
        "eval.ground_truth_builder._search_full_text",
        return_value=[{"cik": 1, "company_name": "Delisted Co", "file_date": "2024-01-01", "form": "DEFM14A"}],
    )
    mocker.patch(
        "eval.ground_truth_builder.sic_universe_builder.fetch_company_profile",
        return_value={"tickers": [], "sic": "3559", "name": "Delisted Co"},
    )

    candidates = ground_truth_builder.discover_fairness_opinion_candidates(["3559"])

    assert candidates[0]["ticker"] is None
    assert candidates[0]["cik"] == 1


def test_fetch_fairness_opinion_text_falls_back_to_s4(mocker):
    class _Filings:
        def __init__(self, filing):
            self._filing = filing

        def latest(self):
            return self._filing

    class _Filing:
        def text(self):
            return "s4 document text"

    class _FakeCompany:
        def get_filings(self, form=None):
            return _Filings(None if form == "DEFM14A" else _Filing())

    mocker.patch("eval.ground_truth_builder.edgar.Company", return_value=_FakeCompany())

    text, form_used = ground_truth_builder._fetch_fairness_opinion_text("ACME")

    assert text == "s4 document text"
    assert form_used == "S-4"


def test_fetch_fairness_opinion_text_returns_none_when_no_filing(mocker):
    class _Filings:
        def latest(self):
            return None

    class _FakeCompany:
        def get_filings(self, form=None):
            return _Filings()

    mocker.patch("eval.ground_truth_builder.edgar.Company", return_value=_FakeCompany())

    text, form_used = ground_truth_builder._fetch_fairness_opinion_text("ACME")

    assert text is None
    assert form_used is None


def test_candidate_identifier_prefers_cik_over_ticker():
    candidate = {"cik": 123, "ticker": "AAA"}

    assert ground_truth_builder._candidate_identifier(candidate) == 123


def test_candidate_identifier_falls_back_to_ticker_string():
    assert ground_truth_builder._candidate_identifier("AAA") == "AAA"


def test_candidate_label_uses_ticker_when_present():
    assert ground_truth_builder._candidate_label({"cik": 123, "ticker": "AAA"}) == "AAA"


def test_candidate_label_falls_back_to_cik():
    assert ground_truth_builder._candidate_label({"cik": 123, "ticker": None}) == "CIK123"


def test_build_ground_truth_for_delisted_cik_candidate(mocker, tmp_path):
    mocker.patch.object(ground_truth_builder, "CACHE_DIR", tmp_path)
    mocker.patch("eval.ground_truth_builder.edgar.set_identity")
    mocker.patch("eval.ground_truth_builder.openai.OpenAI")
    mocker.patch(
        "eval.ground_truth_builder._fetch_fairness_opinion_text",
        return_value=("fairness opinion text mentioning selected companies analysis", "DEFM14A"),
    )
    mocker.patch(
        "eval.ground_truth_builder._extract_selected_companies",
        return_value=["Comp One Inc", "Comp Two Corp"],
    )
    mocker.patch(
        "eval.ground_truth_builder._map_name_to_ticker",
        side_effect=lambda label, name: {"Comp One Inc": "CMP1", "Comp Two Corp": "CMP2"}.get(name),
    )

    candidate = {"cik": 555, "ticker": None, "company_name": "Delisted Target"}
    config = {"llm": {"judge_model": "gpt-4.1-mini", "temperature": 0, "max_tokens": 500}}

    ground_truth = ground_truth_builder.build_ground_truth([candidate], config)

    assert ground_truth == {"CIK555": ["CMP1", "CMP2"]}
    cached = json.loads((tmp_path / "comp_analysis_555.json").read_text())
    assert cached["selected_company_tickers"] == ["CMP1", "CMP2"]
    assert cached["form_used"] == "DEFM14A"


def test_build_ground_truth_uses_cache_on_second_call(mocker, tmp_path):
    mocker.patch.object(ground_truth_builder, "CACHE_DIR", tmp_path)
    mocker.patch("eval.ground_truth_builder.edgar.set_identity")
    mocker.patch("eval.ground_truth_builder.openai.OpenAI")
    mock_fetch = mocker.patch(
        "eval.ground_truth_builder._fetch_fairness_opinion_text",
        return_value=("text", "DEFM14A"),
    )
    mocker.patch("eval.ground_truth_builder._extract_selected_companies", return_value=["Comp One Inc"])
    mocker.patch("eval.ground_truth_builder._map_name_to_ticker", return_value="CMP1")
    config = {"llm": {"judge_model": "gpt-4.1-mini", "temperature": 0, "max_tokens": 500}}

    ground_truth_builder.build_ground_truth(["ACME"], config)
    ground_truth_builder.build_ground_truth(["ACME"], config)

    mock_fetch.assert_called_once()


def test_build_ground_truth_skips_when_no_filing_found(mocker, tmp_path):
    mocker.patch.object(ground_truth_builder, "CACHE_DIR", tmp_path)
    mocker.patch("eval.ground_truth_builder.edgar.set_identity")
    mocker.patch("eval.ground_truth_builder.openai.OpenAI")
    mocker.patch("eval.ground_truth_builder._fetch_fairness_opinion_text", return_value=(None, None))
    config = {"llm": {"judge_model": "gpt-4.1-mini", "temperature": 0, "max_tokens": 500}}

    ground_truth = ground_truth_builder.build_ground_truth(["ACME"], config)

    assert ground_truth == {}


def test_build_manual_deal_review_prefills_audit_file(mocker, tmp_path):
    mocker.patch("eval.ground_truth_builder.edgar.set_identity")
    mocker.patch("eval.ground_truth_builder.openai.OpenAI")
    mocker.patch(
        "eval.ground_truth_builder._fetch_fairness_opinion_text",
        return_value=("fairness opinion text", "DEFM14A"),
    )
    mocker.patch(
        "eval.ground_truth_builder._extract_selected_companies",
        return_value=["Comp One Inc", "Comp Two Corp"],
    )
    mocker.patch(
        "eval.ground_truth_builder._map_name_to_ticker",
        side_effect=lambda label, name: {"Comp One Inc": "CMP1"}.get(name),
    )
    candidate = {
        "cik": 555,
        "ticker": None,
        "company_name": "Delisted Target",
        "sic": "3559",
        "file_date": "2024-01-01",
        "form": "DEFM14A",
    }
    config = {"llm": {"judge_model": "gpt-4.1-mini", "temperature": 0, "max_tokens": 500}}
    output_path = tmp_path / "manual_deals.review.json"

    reviews = ground_truth_builder.build_manual_deal_review([candidate], config, output_path)

    assert reviews == json.loads(output_path.read_text())
    assert reviews[0]["review_status"] == "needs_review"
    assert reviews[0]["manual_fields_to_confirm"] == [
        "filing_url",
        "advisor",
        "target_financials",
        "business_description",
        "selected_company_tickers",
        "selected_company_still_public_flags",
    ]
    assert reviews[0]["target"]["label"] == "CIK555"
    assert reviews[0]["target"]["ticker"] is None
    assert reviews[0]["target"]["source_candidate"]["file_date"] == "2024-01-01"
    assert reviews[0]["filing"]["form_used"] == "DEFM14A"
    assert reviews[0]["selected_companies"] == [
        {
            "company_name": "Comp One Inc",
            "suggested_ticker": "CMP1",
            "still_public": None,
            "include_in_ground_truth": None,
            "review_status": "needs_review",
        },
        {
            "company_name": "Comp Two Corp",
            "suggested_ticker": None,
            "still_public": None,
            "include_in_ground_truth": None,
            "review_status": "needs_review",
        },
    ]


CIRCOR_URL = "https://www.sec.gov/Archives/edgar/data/1091883/000114036123034666/ny20009611x2_defm14a.htm"


def test_parse_filing_url_extracts_cik_and_accession():
    cik, accession = ground_truth_builder._parse_filing_url(CIRCOR_URL)

    assert cik == 1091883
    assert accession == "0001140361-23-034666"


def test_parse_filing_url_rejects_non_filing_url():
    import pytest

    with pytest.raises(ValueError):
        ground_truth_builder._parse_filing_url("https://www.sec.gov/edgar/search/")


def test_html_to_text_strips_markup_and_entities():
    document = "<html><script>var x=1;</script><body><p>Selected&nbsp;Companies &amp; Analysis</p>\n<table><tr><td>Flowserve</td></tr></table></body></html>"

    text = ground_truth_builder._html_to_text(document)

    assert "var x=1" not in text
    assert "Selected Companies & Analysis" in text
    assert "Flowserve" in text
    assert "<" not in text


def _submissions_profile(name, sic, tickers, accession, filing_date, form, all_forms):
    return {
        "name": name,
        "sic": sic,
        "sicDescription": "Pumps & Pumping Equipment",
        "tickers": tickers,
        "filings": {
            "recent": {
                "accessionNumber": [accession],
                "filingDate": [filing_date],
                "form": [form] if not all_forms else all_forms,
            },
        },
    }


def test_build_manual_deal_review_from_urls_prefills_entry(mocker, tmp_path):
    mocker.patch("eval.ground_truth_builder.edgar.set_identity")
    mocker.patch("eval.ground_truth_builder.openai.OpenAI")
    mocker.patch("eval.ground_truth_builder.time.sleep")
    mocker.patch(
        "eval.ground_truth_builder._fetch_filing_document",
        return_value="selected companies analysis ... prospective financial information ...",
    )
    mocker.patch(
        "eval.ground_truth_builder._call_openai",
        side_effect=[
            '{"advisor": "Evercore Group L.L.C.", "selected_companies": ["Comp One Inc", "Foreign AG"]}',
            '{"fiscal_year": "FY2023E", "revenue_usd_mm": 850.0, "ebitda_usd_mm": 127.5, "source_note": "Projections table"}',
        ],
    )
    mocker.patch(
        "eval.ground_truth_builder._map_name_to_ticker",
        side_effect=lambda label, name: {"Comp One Inc": "CMP1"}.get(name),
    )
    mocker.patch("eval.ground_truth_builder._current_us_listings", return_value={"CMP1": 111})
    mocker.patch("eval.ground_truth_builder._fetch_target_description", return_value="Makes pumps.")

    target_profile = _submissions_profile(
        "CIRCOR INTERNATIONAL INC", "3561", [], "0001140361-23-034666", "2023-07-17", "DEFM14A", None,
    )
    comp_profile = _submissions_profile("Comp One Inc", "3561", ["CMP1"], "x", "2024-01-01", None, ["10-K", "8-K"])
    mocker.patch(
        "eval.ground_truth_builder.sic_universe_builder.fetch_company_profile",
        side_effect=lambda cik: {1091883: target_profile, 111: comp_profile}[cik],
    )

    config = {"llm": {"extraction_model": "gpt-4.1", "temperature": 0, "max_tokens": 500}}
    output_path = tmp_path / "manual_deals.review.json"

    entries = ground_truth_builder.build_manual_deal_review_from_urls([CIRCOR_URL], config, output_path)

    assert entries == json.loads(output_path.read_text())
    entry = entries[0]
    assert entry["review_status"] == "needs_review"
    assert entry["target"]["label"] == "CIK1091883"
    assert entry["target"]["sic"] == "3561"
    assert entry["filing"] == {
        "url": CIRCOR_URL,
        "accession": "0001140361-23-034666",
        "form_used": "DEFM14A",
        "filing_date": "2023-07-17",
    }
    assert entry["advisor"] == "Evercore Group L.L.C."
    assert entry["selected_companies"] == [
        {
            "company_name": "Comp One Inc",
            "suggested_ticker": "CMP1",
            "still_public": True,
            "us_filer": True,
            "include_in_ground_truth": None,
            "review_status": "needs_review",
        },
        {
            "company_name": "Foreign AG",
            "suggested_ticker": None,
            "still_public": None,
            "us_filer": None,
            "include_in_ground_truth": None,
            "review_status": "needs_review",
        },
    ]

    deal = entry["suggested_manual_deal"]
    assert deal["deal_id"] == "circor-international-inc-2023"
    assert deal["target_ticker"] == "CIK1091883"
    assert deal["target_cik"] == "0001091883"
    assert deal["target_sic"] == "3561"
    assert deal["business_description"] == "Makes pumps."
    assert deal["target_financials"]["revenue_usd_mm"] == 850.0
    assert deal["target_financials"]["ebitda_margin_estimate"] == 0.15
    assert "FY2023E" in deal["target_financials"]["source"]
    assert deal["review_status"] == "needs_review"
    assert deal["selected_companies"][0] == {
        "company_name": "Comp One Inc",
        "ticker": "CMP1",
        "still_public": True,
        "us_filer": True,
    }


def test_build_manual_deal_review_from_urls_merges_by_filing_url(mocker, tmp_path):
    mocker.patch("eval.ground_truth_builder.edgar.set_identity")
    mocker.patch("eval.ground_truth_builder.openai.OpenAI")
    mocker.patch("eval.ground_truth_builder._fetch_filing_document", return_value="text")
    mocker.patch(
        "eval.ground_truth_builder._call_openai",
        side_effect=[
            '{"advisor": "Bank B", "selected_companies": []}',
            '{"fiscal_year": null, "revenue_usd_mm": null, "ebitda_usd_mm": null, "source_note": null}',
        ],
    )
    mocker.patch("eval.ground_truth_builder._fetch_target_description", return_value=None)
    mocker.patch(
        "eval.ground_truth_builder.sic_universe_builder.fetch_company_profile",
        return_value=_submissions_profile("Target", "3561", [], "0001140361-23-034666", "2023-07-17", "DEFM14A", None),
    )

    candidate_entry = {"review_status": "needs_review", "target": {"label": "OTHER"}, "filing": {"form_used": "S-4"}}
    stale_url_entry = {"review_status": "needs_review", "advisor": "Old Bank", "filing": {"url": CIRCOR_URL}}
    output_path = tmp_path / "manual_deals.review.json"
    output_path.write_text(json.dumps([candidate_entry, stale_url_entry]))

    config = {"llm": {"extraction_model": "gpt-4.1", "temperature": 0, "max_tokens": 500}}
    entries = ground_truth_builder.build_manual_deal_review_from_urls([CIRCOR_URL], config, output_path)

    merged = json.loads(output_path.read_text())
    assert len(merged) == 2
    assert merged[0] == candidate_entry
    assert merged[1] == entries[0]
    assert merged[1]["advisor"] == "Bank B"
