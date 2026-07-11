"""Single-runner guard for pipeline executions.

The CLI and the Gradio UI share mutable state: `outputs/` report artifacts,
`outputs/failed_tickers.csv`, the fetch cache, and the LLM checkpoint. Two
concurrent runs overwrite each other's reports and interleave checkpoint
writes, so the pipeline takes an explicit cross-process lock for the full
run.

Implementation: a lock file created with O_CREAT|O_EXCL (atomic on every
platform this project targets) holding the owner's PID. A leftover lock whose
PID is no longer alive (crash, SIGKILL — anything that skipped the finally)
is reclaimed automatically, so an aborted run never requires manual cleanup.
"""
import os
from pathlib import Path

from src import get_logger
from src.paths import project_path

logger = get_logger(__name__)

LOCK_PATH = project_path("outputs", ".run.lock")


class PipelineAlreadyRunningError(RuntimeError):
    """Another pipeline run (CLI or UI) currently holds the run lock."""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # signal 0: existence check only, no signal delivered
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def _read_lock_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


class RunLock:
    """Context manager: hold the cross-process run lock for one pipeline run."""

    def __init__(self, path: str | Path | None = None):
        # Default resolved at enter-time so tests monkeypatching LOCK_PATH work.
        self._explicit_path = Path(path) if path is not None else None
        self._acquired = False

    @property
    def path(self) -> Path:
        return self._explicit_path if self._explicit_path is not None else LOCK_PATH

    def _try_create(self) -> bool:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return True

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._try_create():
            holder_pid = _read_lock_pid(self.path)
            if holder_pid is None or not _pid_alive(holder_pid):
                # Unreadable lock or dead holder: a crashed run's leftover.
                logger.warning(
                    f"Reclaiming stale run lock {self.path} "
                    f"(holder pid {holder_pid if holder_pid is not None else 'unreadable'} is gone)"
                )
                self.path.unlink(missing_ok=True)
                if not self._try_create():
                    # Someone else grabbed it between our unlink and create.
                    raise PipelineAlreadyRunningError(self._already_running_message())
            else:
                raise PipelineAlreadyRunningError(self._already_running_message(holder_pid))
        self._acquired = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._acquired:
            self.path.unlink(missing_ok=True)
            self._acquired = False

    def _already_running_message(self, holder_pid: int | None = None) -> str:
        holder = f" (pid {holder_pid})" if holder_pid is not None else ""
        return (
            f"Another pipeline run{holder} is already in progress — CLI and UI runs share "
            f"outputs/ and checkpoint files, so concurrent runs would overwrite each other. "
            f"Wait for it to finish, or delete {self.path} if you are sure no run is active."
        )
