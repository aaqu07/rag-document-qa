# RAG Document Q&A App

A small AI-powered application that lets a user upload 1-3 PDF documents, ask natural-language questions, and receive answers with document/page citations.

This project is intentionally simple: the goal is clean, working code and clear reasoning rather than a production-scale RAG platform.

## Features

- Upload 1-3 PDF documents
- Extract text page-by-page using PyMuPDF
- Split pages into overlapping chunks
- Create embeddings using OpenAI
- Store and query chunks in a local ChromaDB vector store
- Generate grounded answers using an OpenAI chat model
- Show citations with document name and page number
- Run the app with one command after dependencies are installed

## Tech stack

- Streamlit for the UI
- PyMuPDF for PDF text extraction
- OpenAI embeddings for semantic search
- ChromaDB as a local in-memory vector store
- OpenAI chat model for grounded answer generation
- Pytest for lightweight unit tests

## Architecture

```text
PDF upload
  -> Page-wise text extraction
  -> Chunking with overlap
  -> OpenAI embeddings
  -> ChromaDB local vector search
  -> Top-k retrieved chunks
  -> LLM answer generation
  -> Answer + citations
```

## Why this approach?

### Streamlit

Streamlit keeps the UI lightweight and easy to run for a take-home assignment. It avoids adding a separate frontend/backend unless truly needed.

### PyMuPDF

PyMuPDF gives reliable page-wise text extraction from text-based PDFs. Page-wise extraction is important because citations need to point back to the source document and page.

### ChromaDB

ChromaDB is used as a simple local vector store. It supports storing document chunks with metadata such as file name, page number, and chunk id. That makes citation handling straightforward.

### OpenAI embeddings + chat model

Embeddings are used for semantic retrieval. The chat model receives only the retrieved context and is instructed to answer using that context only.

## Project structure

```text
.
|-- app.py
|-- src/
|   |-- chunker.py
|   |-- config.py
|   |-- embeddings.py
|   |-- pdf_loader.py
|   |-- rag_chain.py
|   `-- vector_store.py
|-- tests/
|   `-- test_chunking.py
|-- .env.example
|-- requirements.txt
|-- Dockerfile
|-- docker-compose.yml
`-- README.md
```

## Local setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment variables

Copy the example file:

```bash
cp .env.example .env
```

On Windows PowerShell, you can use:

```powershell
Copy-Item .env.example .env
```

Then add your OpenAI API key:

```text
OPENAI_API_KEY=your_api_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Optional Docker run

If Docker is available, you can also run the app in a container. This path is optional and expects `.env` to exist with `OPENAI_API_KEY` set:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

## How to use

1. Upload 1-3 PDF files.
2. Click **Build / refresh index**.
3. Ask a natural-language question about the uploaded documents.
4. Read the answer and expand **Sources reviewed** to see document/page citations.

## Citation strategy

Each PDF is extracted page-by-page. Every chunk stores metadata:

- document name
- page number
- chunk index

When a question is asked, the app retrieves the most relevant chunks and passes them to the model as numbered sources. The answer is instructed to cite those source numbers, and the UI displays the document name and page number for each source.

Example:

```text
The document states that submissions should include a README and dependency file [1].
```

The source table then maps `[1]` to the original PDF and page.

## Known limitations

- Works best with text-based PDFs, not scanned image PDFs.
- Citations are page-level, not exact paragraph-level.
- Complex tables, multi-column layouts, and images may not be extracted perfectly.
- The ChromaDB vector index is session-based and in-memory. It is rebuilt when documents change and is not persisted between app sessions.
- No authentication, user management, or persistent document library is included.
- The answer quality depends on PDF extraction quality and the retrieved chunks.

## Improvements with more time

- Add OCR support for scanned PDFs.
- Add section-heading extraction for section-level citations.
- Add persistent Chroma storage for saved document collections.
- Add automated RAG evaluation using a small golden question set.
- Add reranking to improve retrieval quality.
- Add support for PDF tables using a table extraction library.

## Running tests

```bash
pytest
```

## Security notes

- Do not commit real API keys.
- `.env` is ignored by `.gitignore`; use `.env.example` to show required configuration.
- Uploaded documents are processed locally in the running app session.
