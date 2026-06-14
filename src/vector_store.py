"""Local ChromaDB vector store wrapper."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import chromadb

from src.chunker import DocumentChunk


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk returned from vector search."""

    text: str
    document_name: str
    page_number: int
    chunk_index: int
    distance: float | None = None


class LocalChromaStore:
    """Small local vector store for uploaded documents.

    We use an ephemeral Chroma collection because uploaded documents are specific
    to the current Streamlit session. This keeps the demo simple and avoids stale
    document data between runs.
    """

    def __init__(self) -> None:
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.create_collection(
            name=f"document_qa_{uuid.uuid4().hex[:12]}",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add embedded chunks to Chroma."""

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return

        self._collection.add(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )

    def query(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return the top-k semantically similar chunks."""

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            retrieved.append(
                RetrievedChunk(
                    text=text,
                    document_name=str(metadata.get("document_name", "Unknown document")),
                    page_number=int(metadata.get("page_number", 0)),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    distance=float(distance) if distance is not None else None,
                )
            )

        return retrieved
