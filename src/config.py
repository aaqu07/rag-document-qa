"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
    """Read a bounded integer from the environment."""

    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default

    value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


@dataclass(frozen=True)
class AppConfig:
    """Configurable app settings."""

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    max_files: int = _int_env("MAX_FILES", 3, minimum=1, maximum=3)
    chunk_words: int = _int_env("CHUNK_WORDS", 220, minimum=1)
    chunk_overlap_words: int = _int_env("CHUNK_OVERLAP_WORDS", 40, minimum=0)
    default_top_k: int = _int_env("TOP_K", 5, minimum=2, maximum=8)
