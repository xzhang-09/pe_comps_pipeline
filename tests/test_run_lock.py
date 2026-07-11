import os

import pytest

from src.paths import project_path
from src.run_lock import LOCK_PATH, PipelineAlreadyRunningError, RunLock


@pytest.fixture
def lock_path(tmp_path):
    return tmp_path / "outputs" / ".run.lock"


class TestRunLock:
    def test_acquires_and_releases(self, lock_path):
        with RunLock(lock_path):
            assert lock_path.exists()
            assert lock_path.read_text(encoding="utf-8") == str(os.getpid())
        assert not lock_path.exists()

    def test_released_on_exception(self, lock_path):
        with pytest.raises(RuntimeError, match="boom"):
            with RunLock(lock_path):
                raise RuntimeError("boom")
        assert not lock_path.exists()

    def test_second_acquire_fails_while_held_by_live_process(self, lock_path):
        with RunLock(lock_path):
            with pytest.raises(PipelineAlreadyRunningError, match="already in progress"):
                with RunLock(lock_path):
                    pass
        # The failed acquire must not have released the first holder's lock...
        # (we're outside the with now, so it IS released)
        assert not lock_path.exists()

    def test_failed_acquire_does_not_delete_active_lock(self, lock_path):
        with RunLock(lock_path):
            with pytest.raises(PipelineAlreadyRunningError):
                with RunLock(lock_path):
                    pass
            assert lock_path.exists()  # still held by the outer lock

    def test_stale_lock_from_dead_process_is_reclaimed(self, lock_path, caplog):
        """A crashed run (SIGKILL, power loss) leaves its lock behind; the
        next run must reclaim it automatically rather than requiring the
        operator to delete a hidden file."""
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Spawn-and-reap a child so we hold a PID that is guaranteed dead.
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
        lock_path.write_text(str(pid), encoding="utf-8")

        with caplog.at_level("WARNING"):
            with RunLock(lock_path):
                assert lock_path.read_text(encoding="utf-8") == str(os.getpid())
        assert any("stale run lock" in m.lower() for m in caplog.messages)

    def test_unreadable_lock_is_reclaimed(self, lock_path):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("not-a-pid", encoding="utf-8")

        with RunLock(lock_path):
            assert lock_path.exists()
        assert not lock_path.exists()

    def test_default_path_is_under_outputs(self):
        assert LOCK_PATH == project_path("outputs", ".run.lock")
        assert LOCK_PATH.is_absolute()


class TestPipelineIntegration:
    def test_run_pipeline_refuses_to_start_while_lock_held(self, mocker, tmp_path, monkeypatch):
        """The end-to-end behavior the lock exists for: a second run against
        the same checkout aborts up front instead of clobbering outputs."""
        import yaml

        import src.pipeline as pipeline

        monkeypatch.chdir(tmp_path)
        config = {
            "target_company": {
                "name": "T", "description": "d", "primary_sic_codes": ["3714"], "adjacent_sic_codes": [],
            },
            "universe": {"max_candidates": 10},
            "llm": {
                "extraction_model": "gpt-4.1", "judge_model": "gpt-4.1-mini",
                "temperature": 0, "max_tokens": 500, "batch_size": 20, "judge_threshold": 3,
            },
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        enrich = mocker.patch("src.pipeline.enrich_universe")

        lock_file = tmp_path / "outputs" / ".run.lock"
        monkeypatch.setattr(pipeline.run_lock, "LOCK_PATH", lock_file)
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.write_text(str(os.getpid()), encoding="utf-8")  # a live "other run"

        with pytest.raises(PipelineAlreadyRunningError):
            pipeline.run_pipeline(str(config_path))

        enrich.assert_not_called()
        assert lock_file.exists()  # the refused run must not release the holder's lock
