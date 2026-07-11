"""Shared defaults that must stay consistent across entry points."""
import os

from src import get_logger

logger = get_logger(__name__)

DEFAULT_SEC_IDENTITY = "PE-Comps-Pipeline research@example.com"
DEFAULT_LLM_CONFIG = {
    "extraction_model": "gpt-4.1",
    "judge_model": "gpt-4.1-mini",
    "embedding_model": "text-embedding-3-small",
    "temperature": 0,
    "max_tokens": 500,
    "batch_size": 20,
    "judge_threshold": 3,
}


def sec_identity() -> str:
    identity = os.environ.get("SEC_IDENTITY")
    if identity:
        return identity
    logger.warning(
        "SEC_IDENTITY is not set; using placeholder SEC User-Agent. "
        "Set SEC_IDENTITY='Your Name your.email@example.com' for production runs."
    )
    return DEFAULT_SEC_IDENTITY
