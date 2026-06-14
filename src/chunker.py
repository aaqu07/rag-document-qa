"""Chunking utilities for document retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from src.pdf_loader import PageText


@dataclass(frozen=True)
class DocumentChunk:
    """A retrievable piece of text with citation metadata."""

    id: str
    text: str
    document_name: str
    page_number: int
    chunk_index: int


def chunk_pages(
    pages: list[PageText],
    *,
    max_words: int = 220,
    overlap_words: int = 40,
) -> list[DocumentChunk]:
    """Split page text into overlapping word chunks.

    A simple word-based chunker is enough for this take-home app. The overlap helps
    reduce boundary issues when an answer spans two adjacent chunks.
    """

    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    chunks: list[DocumentChunk] = []
    chunk_counter = 0

    for page in pages:
        words = page.text.split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunk_id = (
                    f"{_safe_id(page.document_name)}-p{page.page_number}-c{chunk_counter}"
                )
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        text=chunk_text,
                        document_name=page.document_name,
                        page_number=page.page_number,
                        chunk_index=chunk_counter,
                    )
                )
                chunk_counter += 1

            if end == len(words):
                break
            start = end - overlap_words

    return chunks


def _safe_id(value: str) -> str:
    """Create a stable-ish collection id segment from a filename."""

    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-")
