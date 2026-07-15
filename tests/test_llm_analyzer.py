import json

import src.llm_analyzer as llm_analyzer

DEFAULT_DESCRIPTION = (
    "The Company designs, manufactures, and sells specialty industrial "
    "fasteners for automotive OEM customers under long-term supply contracts."
)


def _company(ticker, description=DEFAULT_DESCRIPTION, name=None):
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
    "evidence_quote": "designs, manufactures, and sells specialty industrial fasteners",
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
    assert results["EEE"]["evidence_verified"] is True


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


def test_hallucinated_evidence_quote_forces_low_confidence(mocker, sample_config):
    """A quote that doesn't actually appear in the source description should
    be caught even if the judge model rates the extraction highly — this is
    exactly the gap a same-pass confidence score can't catch."""
    extraction = json.dumps({
        "business_model": "manufacturing",
        "revenue_recurrence": "high",
        "customer_type": "B2B",
        "capital_intensity": "asset_heavy",
        "primary_value_driver": "scale",
        "sub_sector_description": "specialty industrial fastener distributor",
        "evidence_quote": "this sentence never appears in the description at all",
        "confidence": 5,
    })
    judge = json.dumps({"score": 5, "reason": "looks well supported"})
    _mock_client(mocker, [extraction, judge])

    results = llm_analyzer.analyze_batch([_company("GGG")], sample_config)
    result = results["GGG"]

    assert result["evidence_verified"] is False
    assert result["low_confidence_flag"] is True


def test_missing_evidence_quote_forces_low_confidence(mocker, sample_config):
    extraction = json.dumps({
        "business_model": "manufacturing",
        "revenue_recurrence": "high",
        "customer_type": "B2B",
        "capital_intensity": "asset_heavy",
        "primary_value_driver": "scale",
        "sub_sector_description": "specialty industrial fastener distributor",
        "evidence_quote": None,
        "confidence": 4,
    })
    judge = json.dumps({"score": 5, "reason": "looks well supported"})
    _mock_client(mocker, [extraction, judge])

    results = llm_analyzer.analyze_batch([_company("HHH")], sample_config)
    result = results["HHH"]

    assert result["evidence_verified"] is False
    assert result["low_confidence_flag"] is True


def test_null_business_model_does_not_require_evidence(mocker, sample_config):
    extraction = json.dumps({
        "business_model": None,
        "revenue_recurrence": None,
        "customer_type": None,
        "capital_intensity": None,
        "primary_value_driver": None,
        "sub_sector_description": None,
        "evidence_quote": None,
        "confidence": 1,
    })
    judge = json.dumps({"score": 4, "reason": "insufficient information"})
    _mock_client(mocker, [extraction, judge])

    results = llm_analyzer.analyze_batch([_company("III")], sample_config)
    result = results["III"]

    assert result["evidence_verified"] is True
    assert result["low_confidence_flag"] is False


def _config_with_target_description(sample_config, description=DEFAULT_DESCRIPTION):
    config = {**sample_config, "target_company": {**sample_config["target_company"], "description": description}}
    return config


def test_suggest_sic_codes_returns_parsed_suggestions(mocker, sample_config):
    response = json.dumps({
        "suggestions": [
            {
                "sic_code": "3714",
                "title": "Motor Vehicle Parts & Accessories",
                "bucket": "primary",
                "reason": "Description explicitly mentions automotive OEM fasteners.",
                "confidence": "high",
            },
            {
                "sic_code": "3490",
                "title": "Misc. Fabricated Metal Products",
                "bucket": "adjacent",
                "reason": "Broader fastener/metal-products manufacturing.",
                "confidence": "medium",
            },
        ],
    })
    _mock_client(mocker, [response])

    suggestions = llm_analyzer.suggest_sic_codes(_config_with_target_description(sample_config))

    assert len(suggestions) == 2
    assert suggestions[0]["sic_code"] == "3714"
    assert suggestions[0]["bucket"] == "primary"
    assert suggestions[1]["confidence"] == "medium"


def test_suggest_sic_codes_returns_empty_list_on_invalid_json(mocker, sample_config):
    _mock_client(mocker, ["not valid json"])

    suggestions = llm_analyzer.suggest_sic_codes(_config_with_target_description(sample_config))

    assert suggestions == []


def test_suggest_sic_codes_returns_empty_list_on_api_failure(mocker, sample_config):
    client = mocker.MagicMock()
    client.responses.create.side_effect = RuntimeError("API down")
    mocker.patch("src.llm_analyzer.openai.OpenAI", return_value=client)

    suggestions = llm_analyzer.suggest_sic_codes(_config_with_target_description(sample_config))

    assert suggestions == []


def test_evidence_quote_with_whitespace_differences_still_verified(mocker, sample_config):
    extraction = json.dumps({
        "business_model": "manufacturing",
        "revenue_recurrence": "high",
        "customer_type": "B2B",
        "capital_intensity": "asset_heavy",
        "primary_value_driver": "scale",
        "sub_sector_description": "specialty industrial fastener distributor",
        "evidence_quote": "designs,  manufactures,\nand sells specialty industrial fasteners",
        "confidence": 4,
    })
    judge = json.dumps({"score": 5, "reason": "clearly supported"})
    _mock_client(mocker, [extraction, judge])

    results = llm_analyzer.analyze_batch([_company("JJJ")], sample_config)
    result = results["JJJ"]

    assert result["evidence_verified"] is True
    assert result["low_confidence_flag"] is False


def test_embed_texts_returns_vectors_in_order(mocker):
    client = mocker.MagicMock()
    client.embeddings.create.return_value = mocker.MagicMock(data=[
        mocker.MagicMock(embedding=[1.0, 0.0]),
        mocker.MagicMock(embedding=[0.0, 1.0]),
    ])
    mocker.patch("src.llm_analyzer.openai.OpenAI", return_value=client)

    vectors = llm_analyzer.embed_texts(["a", "b"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    client.embeddings.create.assert_called_once_with(model=llm_analyzer.DEFAULT_EMBEDDING_MODEL, input=["a", "b"])


def test_embed_texts_uses_explicit_model_when_given(mocker):
    client = mocker.MagicMock()
    client.embeddings.create.return_value = mocker.MagicMock(data=[mocker.MagicMock(embedding=[1.0])])
    mocker.patch("src.llm_analyzer.openai.OpenAI", return_value=client)

    llm_analyzer.embed_texts(["a"], model="custom-embedding-model")

    client.embeddings.create.assert_called_once_with(model="custom-embedding-model", input=["a"])


def test_embed_texts_returns_none_on_api_failure(mocker):
    client = mocker.MagicMock()
    client.embeddings.create.side_effect = RuntimeError("API down")
    mocker.patch("src.llm_analyzer.openai.OpenAI", return_value=client)

    assert llm_analyzer.embed_texts(["a"]) is None


def test_embed_texts_returns_empty_list_for_empty_input(mocker):
    mock_client = mocker.patch("src.llm_analyzer.openai.OpenAI")

    assert llm_analyzer.embed_texts([]) == []
    mock_client.assert_not_called()
