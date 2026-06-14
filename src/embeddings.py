"""Embedding helpers."""

from __future__ import annotations

from openai import OpenAI


def embed_texts(
    client: OpenAI,
    texts: list[str],
    *,
    model: str,
    batch_size: int = 64,
) -> list[list[float]]:
    """Create embeddings for a list of texts using batching."""

    embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend(item.embedding for item in response.data)

    return embeddings
