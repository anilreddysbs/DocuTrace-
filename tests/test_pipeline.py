from __future__ import annotations

import sys
import types
from pathlib import Path

import fitz
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from docutrace.embeddings.embedder import Embedder
from docutrace.generation import llm_client as llm_module
from docutrace.generation.llm_client import GeminiLLMClient
from docutrace.ingestion.pdf_loader import extract_pdf_pages, stable_document_id
from docutrace.processing.cleaning import clean_text
from docutrace.processing.chunking import chunk_pages
from docutrace.retrieval.bm25_retriever import BM25Retriever
from docutrace.vector_store.chroma_store import ChromaChunkStore


@pytest.fixture
def synthetic_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Pipeline daily load checks upstream partition availability before retry.")
    page = doc.new_page()
    page.insert_text((50, 72), "Hive troubleshooting checks SHOW PARTITIONS and date format.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_pdf_extraction_and_metadata_preservation(synthetic_pdf_bytes: bytes) -> None:
    original_name = "sample_dataops.pdf"
    pages = extract_pdf_pages(synthetic_pdf_bytes, original_name)
    assert len(pages) == 2
    assert pages[0]["source"] == original_name
    assert pages[0]["page_number"] == 1
    assert "Pipeline" in pages[0]["text"]
    assert pages[0]["document_id"] == stable_document_id(synthetic_pdf_bytes, original_name)


def test_cleaning_normalizes_whitespace() -> None:
    raw = "Hello\r\n  world\n\n   again\n"
    cleaned = clean_text(raw)
    assert "Hello world" in cleaned
    assert "again" in cleaned


def test_chunk_generation_creates_stable_ids() -> None:
    pages = [{
        "source": "sample.pdf",
        "document_id": "abc123",
        "title": "sample",
        "page_number": 1,
        "text": "A " * 200,
    }]
    chunks = chunk_pages(pages, chunk_size=80, overlap=20)
    assert len(chunks) > 0
    assert chunks[0]["chunk_id"].startswith("abc123::p1::c001")
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["document_id"] == "abc123"


def test_embedding_dimension() -> None:
    embedder = Embedder()
    vector = embedder.embed_query("what is the downstream pipeline status?")
    assert vector.shape == (384,)


def test_chroma_insert_and_filter_and_delete(tmp_path: Path) -> None:
    store = ChromaChunkStore(persist_dir=tmp_path / "chroma", collection_name="docutrace_tests")
    embedder = Embedder()
    records = [
        {"chunk_id": "doc_a::p1::c001", "source": "doc_a.pdf", "document_id": "doc_a", "title": "doc_a", "page_number": 1, "chunk_index": 1, "strategy": "recursive_character", "text": "daily pipeline load runs after upstream ingestion."},
        {"chunk_id": "doc_a::p1::c002", "source": "doc_a.pdf", "document_id": "doc_a", "title": "doc_a", "page_number": 1, "chunk_index": 2, "strategy": "recursive_character", "text": "use partition checks to confirm data is present."},
        {"chunk_id": "doc_b::p1::c001", "source": "doc_b.pdf", "document_id": "doc_b", "title": "doc_b", "page_number": 1, "chunk_index": 1, "strategy": "recursive_character", "text": "hive partitions should be validated before target comparisons."},
    ]
    embeddings = embedder.embed_texts([record["text"] for record in records])
    store.upsert_chunks(records, embeddings)
    filtered = store.query(embeddings[0], top_k=5, where={"document_id": "doc_a"})
    assert len(filtered) >= 1
    assert store.delete_document("doc_a") == 2
    assert store.count_by_document("doc_a") == 0


def test_bm25_retrieval_ranks_expected_chunk() -> None:
    chunks = [
        {"chunk_id": "one", "text": "check hive partitions and business date before comparing source and target", "source": "x.pdf", "document_id": "d1", "page_number": 1, "chunk_index": 1},
        {"chunk_id": "two", "text": "run the pipeline only after upstream ingestion completes", "source": "y.pdf", "document_id": "d2", "page_number": 2, "chunk_index": 1},
    ]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("hive partitions business date", top_k=5)
    assert results[0]["chunk_id"] == "one"


def test_missing_api_key_behavior() -> None:
    client = GeminiLLMClient(api_key=None)
    assert client.available is False
    with pytest.raises(RuntimeError):
        client.generate_answer("test question", [{"text": "evidence"}])


def test_grounded_response_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        text = "This answer is grounded in the retrieved evidence and cites [S1]."

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.models = types.SimpleNamespace(generate_content=lambda **kwargs: FakeResponse())

    monkeypatch.setattr(llm_module.genai, "Client", FakeClient)
    client = llm_module.GeminiLLMClient(api_key="fake-key", model_name="gemini-2.0-flash")
    response = client.generate_answer(
        "What checks are required?",
        [{"source": "doc.pdf", "page_number": 1, "chunk_id": "d::p1::c1", "text": "Check partitions and date format."}],
    )
    assert "grounded" in response.lower()
    assert "[S1]" in response or "document" in response.lower()
