from src import embedding_store, embedding_universe_builder


def test_embedding_universe_builder_filters_top_n_and_similarity(mocker, make_config):
    mocker.patch("src.embedding_universe_builder._candidate_tickers", return_value=(["AAA", "BBB", "LOW"], False))
    mocker.patch("src.embedding_universe_builder._populate_store")
    mocker.patch("src.embedding_universe_builder.llm_analyzer.embed_texts", return_value=[[1.0, 0.0]])
    mocker.patch(
        "src.embedding_universe_builder.embedding_store.query",
        return_value=[("AAA", 0.92), ("LOW", 0.20), ("OUTSIDE", 0.99)],
    )
    config = make_config(universe={"discovery_mode": "sic+embedding", "embedding_top_n": 3, "embedding_min_similarity": 0.35})

    result = embedding_universe_builder.discover(config)

    assert [row["ticker"] for row in result] == ["AAA"]
    assert result[0]["candidate_source"] == "embedding"
    assert result[0]["embedding_similarity"] == 0.92
    embedding_universe_builder.embedding_store.query.assert_called_once_with(
        [1.0, 0.0], 3, allowed_tickers={"AAA", "BBB", "LOW"}, embedding_model="text-embedding-3-small",
    )


def test_embedding_universe_builder_populates_missing_descriptions(mocker, make_config):
    mocker.patch("src.embedding_universe_builder._candidate_tickers", return_value=(["AAA"], False))
    mocker.patch("src.embedding_universe_builder.embedding_store.load", return_value={})
    mocker.patch("src.embedding_universe_builder._description_for_ticker", return_value=("AAA makes industrial pumps.", {}))
    embed = mocker.patch("src.embedding_universe_builder.llm_analyzer.embed_texts", side_effect=[[[0.1, 0.2]], [[1.0, 0.0]]])
    upsert = mocker.patch("src.embedding_universe_builder.embedding_store.upsert_many")
    mocker.patch("src.embedding_universe_builder.embedding_store.query", return_value=[("AAA", 0.9)])

    embedding_universe_builder.discover(make_config(universe={"discovery_mode": "sic+embedding"}))

    upsert.assert_called_once()
    record = upsert.call_args.args[0][0]
    assert record["description"] == "AAA makes industrial pumps."
    assert record["embedding_model"] == "text-embedding-3-small"
    assert record["dimension"] == 2
    assert embed.call_count == 2


def test_populate_store_reembeds_changed_model_and_handles_partial_response(mocker):
    existing = embedding_store.make_record("AAA", "same", [0.1, 0.2], "old-model")
    mocker.patch("src.embedding_universe_builder.embedding_store.load", return_value={"AAA": existing})
    mocker.patch(
        "src.embedding_universe_builder._description_for_ticker",
        side_effect=[("same", {}), ("new description", {})],
    )
    mocker.patch("src.embedding_universe_builder.llm_analyzer.embed_texts", return_value=[[0.3, 0.4]])
    upsert = mocker.patch("src.embedding_universe_builder.embedding_store.upsert_many")
    trace = {}

    embedding_universe_builder._populate_store(["AAA", "BBB"], "new-model", trace)

    assert upsert.call_args.args[0][0]["ticker"] == "AAA"
    assert trace == {"AAA": "embedded", "BBB": "corpus_embedding_failed"}


def test_discovery_trace_attributes_description_threshold_and_top_n(mocker, make_config):
    mocker.patch("src.embedding_universe_builder._candidate_tickers", return_value=(["FAIL", "LOW", "CUT"], False))
    mocker.patch("src.embedding_universe_builder._populate_store", side_effect=lambda _t, _m, trace: trace.update({
        "FAIL": "description_fetch_failed", "LOW": "embedded", "CUT": "embedded",
    }))
    mocker.patch("src.embedding_universe_builder.llm_analyzer.embed_texts", return_value=[[1.0, 0.0]])
    mocker.patch("src.embedding_universe_builder.embedding_store.query", return_value=[("LOW", 0.2)])

    embedding_universe_builder.discover(make_config(universe={"discovery_mode": "sic+embedding"}))

    assert embedding_universe_builder.last_discovery_trace() == {
        "FAIL": "description_fetch_failed",
        "LOW": "below_similarity_threshold",
        "CUT": "outside_candidate_set_top_n",
    }


def test_candidate_tickers_skips_broad_expanded_sics(mocker):
    mocker.patch("src.embedding_universe_builder._expanded_sics", return_value=["1111", "2222"])
    mocker.patch(
        "src.embedding_universe_builder.sic_universe_builder.preflight_sic_codes",
        return_value={"1111": 10, "2222": 999},
    )
    discover = mocker.patch(
        "src.embedding_universe_builder.sic_universe_builder.discover_tickers_by_sic",
        return_value=["AAA"],
    )

    assert embedding_universe_builder._candidate_tickers(["1111"]) == (["AAA"], False)
    discover.assert_called_once_with(["1111"])


def test_candidate_tickers_prioritizes_seed_sics_and_stops_at_limit(mocker):
    mocker.patch("src.embedding_universe_builder._expanded_sics", return_value=["1111", "2222"])
    mocker.patch(
        "src.embedding_universe_builder.sic_universe_builder.preflight_sic_codes",
        return_value={"1111": 2, "2222": 2},
    )
    discover = mocker.patch(
        "src.embedding_universe_builder.sic_universe_builder.discover_tickers_by_sic",
        side_effect=[["SEED1", "SEED2"], ["EXP1"]],
    )

    assert embedding_universe_builder._candidate_tickers(["2222"], limit=2) == (["SEED1", "SEED2"], True)
    discover.assert_called_once_with(["2222"])


def test_unenumerated_stage_distinguishes_candidate_limit_from_taxonomy(mocker):
    embedding_universe_builder._last_candidate_limit_reached = True
    embedding_universe_builder._last_candidate_prefixes = {"35"}
    fetch_sic = mocker.patch(
        "src.embedding_universe_builder.sic_universe_builder.fetch_sic_for_ticker",
        side_effect={"SAME": "3569", "OTHER": "3714"}.get,
    )

    assert embedding_universe_builder.unenumerated_stage("SAME") == "truncated_by_embedding_candidate_limit"
    assert embedding_universe_builder.unenumerated_stage("OTHER") == "outside_expanded_taxonomy"
    embedding_universe_builder._last_candidate_limit_reached = False
    assert embedding_universe_builder.unenumerated_stage("SAME") == "outside_expanded_taxonomy"
    assert fetch_sic.call_count == 2
