from src.chunker import chunk_pages
from src.pdf_loader import PageText
from src.rag_chain import build_source_rows
from src.vector_store import RetrievedChunk


def test_chunk_pages_preserves_metadata():
    page = PageText(document_name="sample.pdf", page_number=2, text=" ".join(["word"] * 500))

    chunks = chunk_pages([page], max_words=100, overlap_words=20)

    assert len(chunks) > 1
    assert chunks[0].document_name == "sample.pdf"
    assert chunks[0].page_number == 2
    assert chunks[0].text


def test_chunk_pages_rejects_invalid_overlap():
    page = PageText(document_name="sample.pdf", page_number=1, text="hello world")

    try:
        chunk_pages([page], max_words=10, overlap_words=10)
    except ValueError as exc:
        assert "overlap_words" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_build_source_rows_shows_citation_source_and_page():
    contexts = [
        RetrievedChunk(
            text="This is the cited text.",
            document_name="handbook.pdf",
            page_number=3,
            chunk_index=0,
            distance=0.1,
        )
    ]

    rows = build_source_rows(contexts)

    assert rows == [
        {
            "Citation": "[1]",
            "Source": "handbook.pdf, page 3",
            "Document": "handbook.pdf",
            "Page": 3,
            "Preview": "This is the cited text.",
        }
    ]
