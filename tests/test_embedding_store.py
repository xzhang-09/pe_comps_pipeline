from src import embedding_store


def test_embedding_store_upserts_and_queries_by_similarity(tmp_path):
    path = tmp_path / "embeddings.json"

    embedding_store.upsert("AAA", "alpha", [1.0, 0.0], path=path, embedding_model="model-a")
    embedding_store.upsert("BBB", "beta", [0.0, 1.0], path=path, embedding_model="model-a")
    embedding_store.upsert("AAA", "alpha updated", [0.8, 0.2], path=path, embedding_model="model-a")

    assert embedding_store.has("aaa", path=path) is True
    assert embedding_store.get("AAA", path=path)["description"] == "alpha updated"
    assert embedding_store.query([1.0, 0.0], top_n=2, path=path)[0][0] == "AAA"


def test_query_filters_candidates_before_top_n(tmp_path):
    path = tmp_path / "embeddings.json"
    records = [
        embedding_store.make_record("OUTSIDE", "outside", [1.0, 0.0], "model-a"),
        embedding_store.make_record("ALLOWED", "allowed", [0.8, 0.2], "model-a"),
    ]
    embedding_store.upsert_many(records, path)

    assert embedding_store.query(
        [1.0, 0.0], 1, path, allowed_tickers={"ALLOWED"}, embedding_model="model-a",
    ) == [("ALLOWED", embedding_store._cosine([1.0, 0.0], [0.8, 0.2]))]


def test_old_mixed_model_and_wrong_dimension_records_are_stale(tmp_path):
    path = tmp_path / "embeddings.json"
    old = {"ticker": "OLD", "description": "old", "embedding": [1.0, 0.0]}
    wrong_dimension = embedding_store.make_record("DIM", "dim", [1.0, 0.0], "model-a")
    wrong_dimension["dimension"] = 3
    embedding_store.upsert_many([old, wrong_dimension], path)

    assert embedding_store.has("OLD", path, description="old", embedding_model="model-a") is False
    assert embedding_store.has("OLD", path) is False
    assert embedding_store.query([1.0, 0.0], 5, path, embedding_model="model-b") == []
    assert embedding_store.query([1.0, 0.0], 5, path, embedding_model="model-a") == []


def test_description_change_marks_record_stale():
    record = embedding_store.make_record("AAA", "old description", [1.0], "model-a")

    assert embedding_store.is_fresh(record, "old description", "model-a") is True
    assert embedding_store.is_fresh(record, "new description", "model-a") is False
