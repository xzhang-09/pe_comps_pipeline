"""Project-root anchored filesystem paths."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
