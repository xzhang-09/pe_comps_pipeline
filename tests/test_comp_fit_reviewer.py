import src.comp_fit_reviewer as comp_fit_reviewer
from src.llm_schemas import CompFitReview


def test_review_prompt_contains_score_band_calibration_guidance():
    assert "Score-band calibration" in comp_fit_reviewer.SYSTEM_PROMPT
    assert "60-69" in comp_fit_reviewer.SYSTEM_PROMPT
    assert "material caveats" in comp_fit_reviewer.SYSTEM_PROMPT
    assert "Do not call a set good" in comp_fit_reviewer.SYSTEM_PROMPT


def _target():
    return {
        "name": "Example Manufacturing Co.",
        "description": "Industrial parts manufacturer serving automotive OEMs.",
        "primary_sic_codes": ["3714"],
        "revenue_usd_mm": 150,
        "ebitda_margin_estimate": 0.18,
    }


def _top_comp(ticker="AAA"):
    return {
        "rank": 1,
        "ticker": ticker,
        "company_name": f"{ticker} Inc.",
        "ev_ebitda_actual": 12.0,
        "residual_abs": 0.5,
        "ebitda_margin": 0.20,
        "gross_margin": 0.35,
        "revenue_ttm_usd_mm": 200.0,
        "revenue_cagr_3yr": 0.05,
        "net_debt_ebitda": 1.5,
        "business_model": "manufacturing",
        "customer_type": "B2B",
        "capital_intensity": "asset_heavy",
        "sub_sector_description": "Automotive OEM parts manufacturer.",
        "judge_score": 4,
        "low_confidence_flag": False,
    }


def test_review_comp_fit_parses_and_caches_llm_response(mocker, tmp_path, make_config):
    mocker.patch.object(comp_fit_reviewer, "REVIEW_PATH", tmp_path / "comp_fit_review.json")
    response = CompFitReview.model_validate({
        "overall_score": 82,
        "review_confidence": "medium",
        "summary": "The selected comps are directionally reasonable.",
        "strengths": ["Mostly manufacturing and B2B."],
        "weaknesses": ["Some end-market drift."],
        "top_fits": [{"ticker": "AAA", "score": 88, "reason": "Strong business-model fit."}],
        "questionable_fits": [],
        "near_miss_upgrades": [],
    })
    client = mocker.MagicMock()
    mocker.patch("src.comp_fit_reviewer.openai.OpenAI", return_value=client)
    mocker.patch("src.comp_fit_reviewer._call_openai_structured", return_value=response)

    result = comp_fit_reviewer.review_comp_fit(_target(), [_top_comp()], [], make_config())

    assert result["status"] == "available"
    assert result["overall_score"] == 82
    assert comp_fit_reviewer.REVIEW_PATH.exists()

    cached_result = comp_fit_reviewer.review_comp_fit(_target(), [_top_comp()], [], make_config())

    assert cached_result["status"] == "available"
    assert comp_fit_reviewer._call_openai_structured.call_count == 1


def test_review_comp_fit_returns_unavailable_on_structured_response_failure(mocker, tmp_path, make_config):
    mocker.patch.object(comp_fit_reviewer, "REVIEW_PATH", tmp_path / "comp_fit_review.json")
    mocker.patch("src.comp_fit_reviewer.openai.OpenAI")
    mocker.patch("src.comp_fit_reviewer._call_openai_structured", side_effect=ValueError("invalid structured response"))

    result = comp_fit_reviewer.review_comp_fit(_target(), [_top_comp()], [], make_config())

    assert result["status"] == "unavailable"
    assert "API call failed" in result["reason"]


def test_review_comp_fit_refreshes_when_signature_changes(mocker, tmp_path, make_config):
    mocker.patch.object(comp_fit_reviewer, "REVIEW_PATH", tmp_path / "comp_fit_review.json")
    responses = [
        CompFitReview.model_validate({
            "overall_score": 70,
            "review_confidence": "low",
            "summary": "First",
            "strengths": [],
            "weaknesses": [],
            "top_fits": [],
            "questionable_fits": [],
            "near_miss_upgrades": [],
        }),
        CompFitReview.model_validate({
            "overall_score": 80,
            "review_confidence": "medium",
            "summary": "Second",
            "strengths": [],
            "weaknesses": [],
            "top_fits": [],
            "questionable_fits": [],
            "near_miss_upgrades": [],
        }),
    ]
    mocker.patch("src.comp_fit_reviewer.openai.OpenAI")
    call = mocker.patch("src.comp_fit_reviewer._call_openai_structured", side_effect=responses)

    first = comp_fit_reviewer.review_comp_fit(_target(), [_top_comp("AAA")], [], make_config())
    second = comp_fit_reviewer.review_comp_fit(_target(), [_top_comp("BBB")], [], make_config())

    assert first["overall_score"] == 70
    assert second["overall_score"] == 80
    assert call.call_count == 2
