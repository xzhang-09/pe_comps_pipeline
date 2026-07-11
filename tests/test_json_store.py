import json
import os
from unittest.mock import patch

import pytest

from src import json_store


class TestWriteJsonAtomic:
    def test_writes_readable_json(self, tmp_path):
        path = tmp_path / "record.json"
        json_store.write_json_atomic(path, {"ticker": "TEST", "value": 1.5})
        assert json.loads(path.read_text(encoding="utf-8")) == {"ticker": "TEST", "value": 1.5}

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "deeper" / "record.json"
        json_store.write_json_atomic(path, [1, 2, 3])
        assert json.loads(path.read_text(encoding="utf-8")) == [1, 2, 3]

    def test_overwrite_replaces_content(self, tmp_path):
        path = tmp_path / "record.json"
        json_store.write_json_atomic(path, {"v": 1})
        json_store.write_json_atomic(path, {"v": 2})
        assert json.loads(path.read_text(encoding="utf-8")) == {"v": 2}

    def test_failed_serialization_leaves_existing_file_intact(self, tmp_path):
        """The atomicity guarantee: a write that blows up mid-serialization
        must not clobber (or truncate) the previous good file."""
        path = tmp_path / "record.json"
        json_store.write_json_atomic(path, {"v": "good"})

        with pytest.raises(TypeError):
            json_store.write_json_atomic(path, {"v": object()})  # not JSON-serializable

        assert json.loads(path.read_text(encoding="utf-8")) == {"v": "good"}

    def test_failed_serialization_leaves_no_temp_files(self, tmp_path):
        path = tmp_path / "record.json"
        with pytest.raises(TypeError):
            json_store.write_json_atomic(path, object())
        assert [p.name for p in tmp_path.iterdir()] == []

    def test_interrupted_replace_leaves_old_file_complete(self, tmp_path):
        """Simulate a crash at the rename step: the destination must still
        hold the previous complete content, never a partial write."""
        path = tmp_path / "record.json"
        json_store.write_json_atomic(path, {"v": "old"})

        with patch.object(os, "replace", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                json_store.write_json_atomic(path, {"v": "new"})

        assert json.loads(path.read_text(encoding="utf-8")) == {"v": "old"}


class TestLoadJson:
    def test_missing_file_returns_default(self, tmp_path):
        assert json_store.load_json(tmp_path / "absent.json") is None
        assert json_store.load_json(tmp_path / "absent.json", default={}) == {}

    def test_valid_file_round_trips(self, tmp_path):
        path = tmp_path / "record.json"
        json_store.write_json_atomic(path, {"a": [1, 2]})
        assert json_store.load_json(path) == {"a": [1, 2]}

    def test_corrupt_file_returns_default_and_warns(self, tmp_path, caplog):
        """A truncated/corrupt cache file is a cache miss, not a crash —
        cache artifacts can be safely re-derived from source."""
        path = tmp_path / "record.json"
        path.write_text('{"ticker": "TEST", "reve', encoding="utf-8")  # truncated mid-write

        with caplog.at_level("WARNING"):
            result = json_store.load_json(path, default={})

        assert result == {}
        assert any("record.json" in message for message in caplog.messages)

    def test_empty_file_returns_default(self, tmp_path):
        path = tmp_path / "record.json"
        path.write_text("", encoding="utf-8")
        assert json_store.load_json(path, default={}) == {}
