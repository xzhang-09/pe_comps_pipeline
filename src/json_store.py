"""Atomic JSON persistence shared by every cache/checkpoint writer.

- A process killed mid-`json.dump` leaves a truncated file behind; the next
  run's bare `json.load` then raises and takes the whole pipeline down, and
  the fix (find and delete the half-written file) costs a day of re-burned
  API quota. write_json_atomic() writes to a temp file in the same directory
  and `os.replace`s it into place, so a reader only ever sees the old
  complete file or the new complete file — never a partial one.

- Even with atomic writes, a corrupt file can still appear from disk issues
  or manual edits.
  load_json() treats unparseable JSON as a cache miss — log a warning naming
  the file, return the caller's "absent" value — instead of crashing, since
  every store this module backs (fetch cache, SIC universe cache, LLM
  checkpoint, comp-fit review cache) can be safely re-derived by refetching.
"""
import json
import os
import tempfile
from pathlib import Path

from src import get_logger

logger = get_logger(__name__)


def write_json_atomic(path: str | Path, data, indent: int = 2) -> None:
    """Serialize `data` to `path` so readers never observe a partial file.

    The temp file lives in the target's own directory (os.replace is only
    atomic within one filesystem), and is cleaned up if serialization fails
    so a crash can't litter the cache dir with orphan temp files.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load_json(path: str | Path, default=None):
    """Read JSON from `path`, treating a missing OR corrupt file as absent.

    Corruption is logged (with the path, so the operator can inspect or
    delete the file) rather than raised: every consumer of this store can
    rebuild the entry from source, which is strictly cheaper than crashing
    an otherwise-healthy multi-hundred-company run.
    """
    path = Path(path)
    if not path.exists():
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        logger.warning(f"{path} — unreadable/corrupt JSON ({e}); treating as absent and re-deriving")
        return default
