"""Streamlit RAG Document Q&A app.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import os

import streamlit as st
from openai import OpenAI

from src.chunker import chunk_pages
from src.config import AppConfig
from src.embeddings import embed_texts
from src.pdf_loader import extract_pages_from_pdf
from src.rag_chain import answer_question, build_source_rows
from src.vector_store import LocalChromaStore

CONFIG = AppConfig()


st.set_page_config(page_title="Document Q&A", page_icon="DOC", layout="wide")
st.title("RAG Document Q&A")
st.caption("Upload 1-3 PDFs, ask questions, and get answers with document/page citations.")


with st.sidebar:
    st.header("Settings")
    api_key_from_env = os.getenv("OPENAI_API_KEY", "")
    api_key = api_key_from_env or st.text_input(
        "OpenAI API key",
        type="password",
        help="Used only for this session. Prefer setting OPENAI_API_KEY in .env.",
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=8,
        value=CONFIG.default_top_k,
        help="More chunks can improve recall but may add noise.",
    )

    st.markdown("---")
    st.markdown(
        "**Models**\n\n"
        f"Embedding: `{CONFIG.embedding_model}`\n\n"
        f"Chat: `{CONFIG.chat_model}`"
    )


if not api_key:
    st.info("Add your OpenAI API key in `.env` or in the sidebar to start.")
    st.stop()

client = OpenAI(api_key=api_key)

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload between 1 and 3 PDF documents.",
)

if uploaded_files and len(uploaded_files) > CONFIG.max_files:
    st.error(f"Please upload at most {CONFIG.max_files} PDFs.")
    st.stop()


def _files_fingerprint(files: list) -> str:
    """Return a hash for the current uploaded file set."""

    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(file.getvalue())
    return digest.hexdigest()


def _build_index(files: list) -> tuple[LocalChromaStore, int, int]:
    """Extract, chunk, embed, and index uploaded files."""

    pages = []
    for file in files:
        pages.extend(extract_pages_from_pdf(file.getvalue(), file.name))

    if not pages:
        raise ValueError("No extractable text was found. The PDFs may be scanned images.")

    chunks = chunk_pages(
        pages,
        max_words=CONFIG.chunk_words,
        overlap_words=CONFIG.chunk_overlap_words,
    )

    if not chunks:
        raise ValueError("Could not create text chunks from the uploaded PDFs.")

    embeddings = embed_texts(
        client,
        [chunk.text for chunk in chunks],
        model=CONFIG.embedding_model,
    )

    store = LocalChromaStore()
    store.add_chunks(chunks, embeddings)
    return store, len(pages), len(chunks)


if uploaded_files:
    current_fingerprint = _files_fingerprint(uploaded_files)
    index_is_current = st.session_state.get("files_fingerprint") == current_fingerprint

    col1, col2 = st.columns([1, 3])
    with col1:
        build_clicked = st.button(
            "Build / refresh index",
            type="primary",
            use_container_width=True,
        )
    with col2:
        if index_is_current:
            st.success(
                f"Index ready: {st.session_state.get('page_count', 0)} pages, "
                f"{st.session_state.get('chunk_count', 0)} chunks."
            )
        else:
            st.warning("Build the index before asking questions.")

    if build_clicked:
        try:
            with st.spinner("Reading PDFs, creating embeddings, and building Chroma index..."):
                store, page_count, chunk_count = _build_index(uploaded_files)
            st.session_state["store"] = store
            st.session_state["files_fingerprint"] = current_fingerprint
            st.session_state["page_count"] = page_count
            st.session_state["chunk_count"] = chunk_count
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    st.markdown("---")

    if st.session_state.get("files_fingerprint") == current_fingerprint:
        question = st.chat_input("Ask a question about the uploaded documents")

        if question:
            st.chat_message("user").write(question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Retrieving relevant chunks and generating answer..."):
                        query_embedding = embed_texts(
                            client,
                            [question],
                            model=CONFIG.embedding_model,
                        )[0]
                        contexts = st.session_state["store"].query(query_embedding, top_k=top_k)
                        answer = answer_question(
                            client,
                            question=question,
                            contexts=contexts,
                            model=CONFIG.chat_model,
                        )

                    st.markdown(answer)

                    with st.expander("Sources reviewed", expanded=True):
                        st.dataframe(
                            build_source_rows(contexts),
                            hide_index=True,
                            use_container_width=True,
                        )
                except Exception as exc:
                    st.error(f"Could not answer the question: {exc}")
else:
    st.info("Upload 1-3 PDF files to begin.")
