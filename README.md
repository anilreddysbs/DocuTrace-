# DocuTrace

DocuTrace is a local multi-PDF knowledge assistant built with Python, Streamlit, Sentence Transformers, ChromaDB, BM25, and Google Gemini.

## Features

- Upload multiple PDFs in the sidebar
- Extract, clean, and chunk text page by page
- Use a reusable recursive chunking strategy based on the existing project experiments
- Generate 384-dimensional MiniLM embeddings
- Store vectors in a persistent local Chroma collection
- Support dense and hybrid retrieval with reciprocal rank fusion
- Show retrieved evidence and page citations
- Run without a Gemini API key, while clearly warning that answer generation requires one
- Include lightweight retrieval evaluation and app tests

## Project structure

- `app.py` — Streamlit user interface
- `src/docutrace/` — modular pipeline code
- `data/raw_pdfs/` — original PDFs
- `data/processed/` — extracted and processed experiment artifacts
- `data/vector_db/chroma/` — persistent vector database
- `tests/` — automated tests

## Installation

1. Activate the project virtual environment:
   `C:\Users\USER\Desktop\Gen_AI\DocuTrace_Rag\.venv\Scripts\activate`
2. Install dependencies:
   `python -m pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your API key if needed.

## Configuration

Create a `.env` file with:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
```

If the API key is not configured, the app still launches and allows indexing and retrieval. It will show a clear message that LLM generation requires the key.

## Running the app

From the project root:

```bash
streamlit run app.py
```

## Retrieval evaluation

From the project root:

```bash
python -m src.docutrace.evaluation.evaluate_retrieval
```

The evaluation module contains a small dataset using the synthetic DataOps PDFs and computes Precision@K, Recall@K, and MRR for the retrieval results.

## Notes

- The recursive chunking configuration is aligned with the existing notebook experiments: around 800 characters with 120-character overlap.
- Chroma uses local persistent storage under `data/vector_db/chroma`.
- The production application avoids hard-coded secrets and keeps the Gemini key in environment variables.
