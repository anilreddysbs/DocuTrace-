# DocuTrace — Multi-PDF RAG Assistant


> **Portfolio:** [anilreddysbs.vercel.app](https://anilreddysbs.vercel.app) · [GitHub profile](https://github.com/anilreddysbs)

DocuTrace is a Streamlit-based assistant for asking questions across multiple PDF documents. It extracts and chunks document text, creates local vector embeddings, retrieves relevant passages with dense and BM25-based search, and uses Google Gemini for answer generation when configured.

## Highlights

- Upload and index multiple PDF files
- Extract text page by page and preserve page-level references
- Split extracted content with a recursive chunking strategy
- Generate 384-dimensional embeddings with a MiniLM Sentence Transformer
- Persist vectors in a local ChromaDB collection
- Combine dense retrieval and BM25 retrieval using reciprocal rank fusion
- Display retrieved evidence with document and page references
- Run indexing and retrieval without a Gemini key; configure a key for LLM-generated answers
- Evaluate retrieval with Precision@K, Recall@K, and Mean Reciprocal Rank (MRR)

## Technology

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- ChromaDB
- BM25 / Reciprocal Rank Fusion
- Google GenAI SDK

## Quick start

### Prerequisites

- Python installed
- pip

### 1. Clone the repository

```bash
git clone https://github.com/anilreddysbs/DocuTrace-.git
cd DocuTrace-
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure Gemini (optional)

Copy `.env.example` to `.env` and set your key:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your-supported-gemini-model
```

Without a Gemini key, the app can still be launched for document indexing and retrieval; generated answers require a configured model key.

### 5. Launch the app

```bash
streamlit run app.py
```

Streamlit prints a local URL in the terminal. Open that URL in your browser.

## Retrieval evaluation

Run the evaluation module from the repository root:

```bash
python -m src.docutrace.evaluation.evaluate_retrieval
```

The evaluation uses the sample DataOps documents and reports Precision@K, Recall@K, and MRR.

Run the test suite with:

```bash
pytest
```

## Project structure

```text
app.py                  Streamlit application
src/docutrace/          Document processing and retrieval modules
notebooks/              Experiments and development notebooks
tests/                  Automated tests
data/raw_pdfs/          Uploaded or sample source PDFs (runtime data)
data/processed/         Extracted text and chunk artifacts (runtime data)
data/vector_db/chroma/  Persistent local ChromaDB data (runtime data)
```

The `data/` directories are runtime storage locations and may be created when the application is used.

## Configuration notes

- Keep API keys in environment variables; do not commit real credentials.
- The recursive chunking experiment currently uses approximately 800-character chunks with 120-character overlap.
- ChromaDB stores its local persistent collection under `data/vector_db/chroma`.
