import json

import src.llm_analyzer as llm_analyzer


def _company(ticker, description="A" * 150, name=None):
    return {
        "ticker": ticker,
        "company_name": name or f"{ticker} Inc.",
        "business_description": description,
    }


def _mock_client(mocker, output_texts):
    """Fake OpenAI client whose responses.create() returns the given
    output_text values in order (one per call)."""
    responses = [mocker.MagicMock(output_text=text) for text in output_texts]
    client = mocker.MagicMock()
    client.responses.create.side_effect = responses
    mocker.patch("src.llm_analyzer.openai.OpenAI", return_value=client)
    return client


VALID_EXTRACTION = json.dumps({
    "business_model": "manufacturing",
    "revenue_recurrence": "high",
    "customer_type": "B2B",
    "capital_intensity": "asset_heavy",
    "primary_value_driver": "scale",
    "sub_sector_description": "specialty industrial fastener distributor",
    "confidence": 4,
})


def test_valid_json_parsed_correctly(mocker, sample_config):
    judge = json.dumps({"score": 5, "reason": "all fields well supported"})
    _mock_client(mocker, [VALID_EXTRACTION, judge])

    results = llm_analyzer.analyze_batch([_company("AAA")], sample_config)
    result = results["AAA"]

    assert result["business_model"] == "manufacturing"
    assert result["judge_score"] == 5


def test_markdown_fenced_json_parsed(mocker, sample_config):
    fenced_extraction = f"```json\n{VALID_EXTRACTION}\n```"
    judge = json.dumps({"score": 4, "reason": "mostly clear"})
    _mock_client(mocker, [fenced_extraction, judge])

    results = llm_analyzer.analyze_batch([_company("BBB")], sample_config)
    result = results["BBB"]

    assert result["business_model"] is not None
    assert result["extraction_failed"] is False


def test_invalid_json_sets_extraction_failed(mocker, sample_config):
    _mock_client(mocker, ["I cannot determine the business model from this text."])

    results = llm_analyzer.analyze_batch([_company("CCC")], sample_config)
    result = results["CCC"]

    assert result["extraction_failed"] is True


def test_low_judge_score_sets_flag(mocker, sample_config):
    judge = json.dumps({"score": 2, "reason": "unclear description"})
    _mock_client(mocker, [VALID_EXTRACTION, judge])

    results = llm_analyzer.analyze_batch([_company("DDD")], sample_config)

    assert results["DDD"]["low_confidence_flag"] is True


def test_high_judge_score_does_not_set_flag(mocker, sample_config):
    judge = json.dumps({"score": 4, "reason": "clearly supported"})
    _mock_client(mocker, [VALID_EXTRACTION, judge])

    results = llm_analyzer.analyze_batch([_company("EEE")], sample_config)

    assert results["EEE"]["low_confidence_flag"] is False


def test_checkpoint_skips_analyzed_ticker(mocker, sample_config):
    llm_analyzer.CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    llm_analyzer._save_checkpoint({"AAA": {"extraction_failed": False}})

    client = _mock_client(mocker, [])

    llm_analyzer.analyze_batch([_company("AAA")], sample_config)

    client.responses.create.assert_not_called()


def test_company_without_description_skipped(mocker, sample_config):
    client = _mock_client(mocker, [])

    results = llm_analyzer.analyze_batch(
        [_company("FFF", description=None)], sample_config,
    )

    client.responses.create.assert_not_called()
    assert results["FFF"]["extraction_failed"] is True
