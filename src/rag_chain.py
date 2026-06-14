"""RAG answer generation."""

from __future__ import annotations

from openai import OpenAI

from src.vector_store import RetrievedChunk

SYSTEM_PROMPT = """You are a careful document question-answering assistant.
Use only the provided context from the uploaded PDFs.
If the answer is not supported by the context, say: "I could not find that in the uploaded documents."
Cite factual claims using the source numbers in square brackets, for example [1] or [2].
Do not invent citations. Do not use outside knowledge.
"""


def answer_question(
    client: OpenAI,
    *,
    question: str,
    contexts: list[RetrievedChunk],
    model: str,
) -> str:
    """Generate an answer grounded in retrieved context."""

    if not contexts:
        return "I could not find relevant context in the uploaded documents."

    user_prompt = f"""Context:
{_format_context(contexts)}

Question:
{question}

Answer with citations:"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    message = response.choices[0].message.content
    return message.strip() if message else "No answer was generated."


def _format_context(contexts: list[RetrievedChunk]) -> str:
    """Format chunks with citation numbers visible to the model."""

    formatted_chunks: list[str] = []
    for index, chunk in enumerate(contexts, start=1):
        formatted_chunks.append(
            f"[{index}] Document: {chunk.document_name} | Page: {chunk.page_number}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(formatted_chunks)


def build_source_rows(contexts: list[RetrievedChunk]) -> list[dict[str, object]]:
    """Create source rows for the Streamlit UI."""

    rows: list[dict[str, object]] = []
    for index, chunk in enumerate(contexts, start=1):
        preview = chunk.text[:280] + ("..." if len(chunk.text) > 280 else "")
        rows.append(
            {
                "Citation": f"[{index}]",
                "Source": f"{chunk.document_name}, page {chunk.page_number}",
                "Document": chunk.document_name,
                "Page": chunk.page_number,
                "Preview": preview,
            }
        )
    return rows
