import hashlib
import json
import subprocess

from src.config_schema import PipelineConfig


def git_commit() -> str:
    """Short HEAD SHA for report provenance; unavailable git is never fatal."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def config_hash(cfg: PipelineConfig) -> str:
    """Stable 12-char fingerprint for the config that produced a report."""
    canonical = json.dumps(cfg.model_dump(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
