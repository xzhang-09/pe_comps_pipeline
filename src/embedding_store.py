import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path

from src import json_store
from src.paths import project_path

STORE_PATH = project_path("data", "cache", "embedding_store.json")
SCHEMA_VERSION = 2


def load(path: Path = STORE_PATH) -> dict:
    data = json_store.load_json(path, default={})
    return data if isinstance(data, dict) else {}


def _save(data: dict, path: Path = STORE_PATH) -> None:
    json_store.write_json_atomic(path, data)


def description_hash(description: str) -> str:
    return hashlib.sha256(description.encode("utf-8")).hexdigest()


def is_fresh(record: object, description: str, embedding_model: str) -> bool:
    if not isinstance(record, dict):
        return False
    embedding = record.get("embedding")
    return (
        record.get("schema_version") == SCHEMA_VERSION
        and record.get("embedding_model") == embedding_model
        and record.get("description_hash") == description_hash(description)
        and isinstance(embedding, list)
        and record.get("dimension") == len(embedding)
    )


def has(
    ticker: str,
    path: Path = STORE_PATH,
    *,
    description: str | None = None,
    embedding_model: str | None = None,
) -> bool:
    record = load(path).get(ticker.upper())
    if description is None or embedding_model is None:
        return (
            isinstance(record, dict)
            and record.get("schema_version") == SCHEMA_VERSION
            and isinstance(record.get("embedding_model"), str)
            and isinstance(record.get("description_hash"), str)
        )
    return is_fresh(record, description, embedding_model)


def get(ticker: str, path: Path = STORE_PATH) -> dict | None:
    value = load(path).get(ticker.upper())
    return value if isinstance(value, dict) else None


def make_record(
    ticker: str,
    description: str,
    embedding: list[float],
    embedding_model: str,
    *,
    source_accession: str | None = None,
    source_date: str | None = None,
    description_source: str | None = None,
) -> dict:
    return {
        "ticker": ticker.upper(),
        "description": description,
        "embedding": embedding,
        "embedding_model": embedding_model,
        "dimension": len(embedding),
        "description_hash": description_hash(description),
        "source_accession": source_accession,
        "source_date": source_date,
        "description_source": description_source,
        "schema_version": SCHEMA_VERSION,
        "embedded_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def upsert_many(records: list[dict], path: Path = STORE_PATH, *, data: dict | None = None) -> None:
    store = load(path) if data is None else data
    for record in records:
        store[record["ticker"].upper()] = record
    _save(store, path)


def upsert(
    ticker: str,
    description: str,
    embedding: list[float],
    path: Path = STORE_PATH,
    *,
    embedding_model: str = "unknown",
) -> None:
    upsert_many([make_record(ticker, description, embedding, embedding_model)], path)


def _cosine(a: list[float], b: list[float]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return None
    return dot / (norm_a * norm_b)


def query(
    embedding: list[float],
    top_n: int,
    path: Path = STORE_PATH,
    *,
    allowed_tickers: set[str] | None = None,
    embedding_model: str | None = None,
) -> list[tuple[str, float]]:
    allowed = {ticker.upper() for ticker in allowed_tickers} if allowed_tickers is not None else None
    rows = []
    for ticker, record in load(path).items():
        if allowed is not None and ticker.upper() not in allowed:
            continue
        if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
            continue
        if embedding_model is not None and record.get("embedding_model") != embedding_model:
            continue
        candidate_embedding = record.get("embedding")
        if not isinstance(candidate_embedding, list) or record.get("dimension") != len(candidate_embedding):
            continue
        similarity = _cosine(embedding, candidate_embedding)
        if similarity is not None:
            rows.append((ticker, similarity))
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows[:top_n]
