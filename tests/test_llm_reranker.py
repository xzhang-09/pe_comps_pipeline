from src import llm_reranker
from src.llm_schemas import RerankResult


def test_llm_reranker_accepts_valid_permutation(mocker):
    mocker.patch(
        "src.llm_reranker._call_openai_structured",
        return_value=RerankResult.model_validate({
            "ordered_tickers": ["BBB", "AAA"],
            "moves": [{"ticker": "BBB", "direction": "up", "reason": "Better end-market fit."}],
        }),
    )

    result = llm_reranker.rerank(
        ranked=[{"ticker": "AAA"}, {"ticker": "BBB"}],
        target_profile={"business_model": "manufacturing"},
        llm_features={"AAA": {}, "BBB": {}},
        companies_by_ticker={"AAA": {}, "BBB": {}},
        model="gpt-4.1",
        temperature=0,
        max_tokens=500,
        rerank_window=2,
    )

    assert result == (["BBB", "AAA"], [{"ticker": "BBB", "direction": "up", "reason": "Better end-market fit."}])


def test_llm_reranker_rejects_invalid_permutation(mocker):
    mocker.patch(
        "src.llm_reranker._call_openai_structured",
        return_value=RerankResult.model_validate({"ordered_tickers": ["AAA", "AAA"], "moves": []}),
    )

    result = llm_reranker.rerank(
        ranked=[{"ticker": "AAA"}, {"ticker": "BBB"}],
        target_profile={},
        llm_features={"AAA": {}, "BBB": {}},
        companies_by_ticker={"AAA": {}, "BBB": {}},
        model="gpt-4.1",
        temperature=0,
        max_tokens=500,
        rerank_window=2,
    )

    assert result is None
