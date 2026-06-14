"""PDF loading utilities.

The app extracts text page-by-page so every chunk can carry page-level citation metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass(frozen=True)
class PageText:
    """Text extracted from one PDF page."""

    document_name: str
    page_number: int
    text: str


def extract_pages_from_pdf(file_bytes: bytes, document_name: str) -> list[PageText]:
    """Extract readable text from a PDF, preserving page numbers.

    Args:
        file_bytes: Raw PDF bytes.
        document_name: Original uploaded file name, used for citations.

    Returns:
        A list of PageText objects. Empty pages are skipped.
    """

    pages: list[PageText] = []

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            for index, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append(
                        PageText(
                            document_name=document_name,
                            page_number=index,
                            text=_normalize_whitespace(text),
                        )
                    )
    except Exception as exc:  # pragma: no cover - Streamlit shows this to the user
        raise ValueError(f"Could not read PDF '{document_name}': {exc}") from exc

    return pages


def _normalize_whitespace(text: str) -> str:
    """Make extracted PDF text easier to chunk and retrieve."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)
