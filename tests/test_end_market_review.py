import json

from src import end_market_reviewer, reporter


def _row(ticker, tier, core_blockers=(), fit_flag=None, outlier_flag=False, fit_notes=""):
    return {
        "ticker": ticker,
        "tier": tier,
        "core_blockers": list(core_blockers),
        "fit_flag": fit_flag,
        "outlier_flag": outlier_flag,
        "fit_notes": fit_notes,
    }


def _target_features():
    return {"sub_sector_description": "Precision motion components for automotive and industrial OEMs."}


def _llm_features(*tickers):
    return {t: {"sub_sector_description": f"{t} sub-sector description"} for t in tickers}


PENALTIES = {"subsector_similarity_threshold": 0.48}


def test_veto_demotes_false_core(mocker, make_config):
    rows = [_row("FET", "core"), _row("STRT", "core")]
    mocker.patch.object(end_market_reviewer, "review_end_markets", return_value={
        "FET": {"aligned": False, "reason": "serves oil and gas end markets, not automotive/industrial OEMs"},
        "STRT": {"aligned": True, "reason": "automotive OEM security components"},
    })

    reporter._apply_end_market_review(
        rows, _target_features(), _llm_features("FET", "STRT"),
        {"FET": 0.55, "STRT": 0.60}, PENALTIES, make_config(),
    )

    fet = next(r for r in rows if r["ticker"] == "FET")
    strt = next(r for r in rows if r["ticker"] == "STRT")
    assert fet["tier"] == "secondary"
    assert "end_market_review" in fet["core_blockers"]
    assert "end-market LLM review blocked Core" in fet["fit_notes"]
    assert strt["tier"] == "core"
    assert strt["fit_notes"] == ""


def test_rescue_promotes_marginal_similarity_shortfall(mocker, make_config):
    rows = [_row("GENC", "secondary", core_blockers=["end_market"], fit_notes="end-market similarity below threshold")]
    mocker.patch.object(end_market_reviewer, "review_end_markets", return_value={
        "GENC": {"aligned": True, "reason": "heavy machinery for adjacent industrial end markets"},
    })

    reporter._apply_end_market_review(
        rows, _target_features(), _llm_features("GENC"),
        {"GENC": 0.41}, PENALTIES, make_config(),
    )

    assert rows[0]["tier"] == "core"
    assert rows[0]["core_blockers"] == []
    assert "overrode a marginal similarity shortfall (0.41 vs. 0.48)" in rows[0]["fit_notes"]


def test_rescue_not_attempted_below_band_or_with_other_blockers(mocker, make_config):
    rows = [
        _row("FARB", "secondary", core_blockers=["end_market"]),           # similarity too far below
        _row("MULT", "secondary", core_blockers=["end_market", "size"]),   # extra blocker
    ]
    review = mocker.patch.object(end_market_reviewer, "review_end_markets", return_value={})

    reporter._apply_end_market_review(
        rows, _target_features(), _llm_features("FARB", "MULT"),
        {"FARB": 0.30, "MULT": 0.45}, PENALTIES, make_config(),
    )

    assert rows[0]["tier"] == "secondary"
    assert rows[1]["tier"] == "secondary"
    review.assert_not_called()  # nothing borderline -> no LLM call


def test_api_failure_changes_nothing(mocker, make_config):
    rows = [_row("FET", "core")]
    mocker.patch.object(end_market_reviewer, "review_end_markets", return_value=None)

    reporter._apply_end_market_review(
        rows, _target_features(), _llm_features("FET"), {"FET": 0.55}, PENALTIES, make_config(),
    )

    assert rows[0]["tier"] == "core"
    assert rows[0]["fit_notes"] == ""


def test_no_target_description_skips_review(mocker, make_config):
    rows = [_row("FET", "core")]
    review = mocker.patch.object(end_market_reviewer, "review_end_markets")

    reporter._apply_end_market_review(
        rows, {"sub_sector_description": None}, _llm_features("FET"), {"FET": 0.55}, PENALTIES, make_config(),
    )

    review.assert_not_called()
    assert rows[0]["tier"] == "core"


def test_weak_or_outlier_rows_are_never_touched(mocker, make_config):
    rows = [
        _row("WEAK", "review_exclude", fit_flag="weak"),
        _row("OUTL", "review_exclude", outlier_flag=True),
    ]
    review = mocker.patch.object(end_market_reviewer, "review_end_markets", return_value={})

    reporter._apply_end_market_review(
        rows, _target_features(), _llm_features("WEAK", "OUTL"),
        {"WEAK": 0.60, "OUTL": 0.60}, PENALTIES, make_config(),
    )

    assert all(r["tier"] == "review_exclude" for r in rows)
    review.assert_not_called()


def test_review_end_markets_parses_structured_verdicts(mocker, make_config):
    response = json.dumps({
        "verdicts": [
            {"ticker": "FET", "aligned": False, "reason": "oil and gas end markets"},
            {"ticker": "HALLUC", "aligned": True, "reason": "not requested"},
        ]
    })

    def parse_response(**kwargs):
        schema = kwargs["text_format"]
        return mocker.MagicMock(output_parsed=schema.model_validate_json(response))

    client = mocker.MagicMock()
    client.responses.parse.side_effect = parse_response
    mocker.patch("openai.OpenAI", return_value=client)

    verdicts = end_market_reviewer.review_end_markets(
        "target end markets", {"FET": "oil and gas equipment"}, make_config(),
    )

    assert verdicts == {"FET": {"aligned": False, "reason": "oil and gas end markets"}}


def test_review_end_markets_returns_none_on_api_failure(make_config):
    # conftest blocks real client construction -> constructor raises.
    verdicts = end_market_reviewer.review_end_markets(
        "target end markets", {"FET": "oil and gas equipment"}, make_config(),
    )
    assert verdicts is None
