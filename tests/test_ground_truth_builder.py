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
