"""Service layer for the DocuTrace RAG pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from docutrace.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K
from docutrace.embeddings.embedder import Embedder
from docutrace.generation.llm_client import GeminiLLMClient
from docutrace.ingestion.pdf_loader import extract_pdf_pages, stable_document_id, validate_uploaded_pdf
from docutrace.processing.cleaning import clean_page_record
from docutrace.processing.chunking import chunk_pages
from docutrace.retrieval.bm25_retriever import BM25Retriever
from docutrace.retrieval.dense_retriever import DenseRetriever
from docutrace.retrieval.hybrid_retriever import HybridRetriever
from docutrace.vector_store.chroma_store import ChromaChunkStore

logger = logging.getLogger(__name__)


class RAGService:
    """Coordinate indexing, retrieval, and generation for a local multi-PDF RAG app."""

    def __init__(
        self,
        vector_store: ChromaChunkStore | None = None,
        embedder: Embedder | None = None,
        llm_client: GeminiLLMClient | None = None,
    ) -> None:
        self.vector_store = vector_store or ChromaChunkStore()
        self.embedder = embedder or Embedder()
        self.llm_client = llm_client or GeminiLLMClient()

    def index_uploaded_files(
        self,
        uploaded_files: list[Any],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> dict[str, Any]:
        """Index a batch of uploaded PDF files without reprocessing unchanged documents."""
        indexed = []
        skipped = []
        for uploaded_file in uploaded_files:
            try:
                file_bytes, original_name = validate_uploaded_pdf(uploaded_file)
                doc_id = stable_document_id(file_bytes, original_name)
                existing_ids = self.vector_store.list_documents()
                if doc_id in existing_ids:
                    skipped.append({"source": original_name, "document_id": doc_id, "status": "already_indexed"})
                    continue

                pages = extract_pdf_pages(file_bytes, original_name)
                cleaned_pages = [clean_page_record(page) for page in pages]
                if not any(page.get("text", "") for page in cleaned_pages):
                    raise ValueError(f"No extractable text found in '{original_name}'. OCR may be required.")

                chunk_records = chunk_pages(cleaned_pages, chunk_size=chunk_size, overlap=overlap)
                if not chunk_records:
                    raise ValueError(f"No chunks could be generated for '{original_name}'.")

                embeddings = self.embedder.embed_texts([record["text"] for record in chunk_records])
                self.vector_store.upsert_chunks(chunk_records, embeddings)
                indexed.append({"source": original_name, "document_id": doc_id, "pages": len(cleaned_pages), "chunks": len(chunk_records)})
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to index uploaded file %s", getattr(uploaded_file, 'name', 'unknown'))
                skipped.append({"source": getattr(uploaded_file, 'name', 'unknown'), "error": str(exc)})

        return {"indexed": indexed, "skipped": skipped}

    def list_indexed_documents(self) -> list[str]:
        """Return the set of currently indexed document IDs."""
        return self.vector_store.list_documents()

    def remove_document(self, document_id: str) -> bool:
        """Delete all chunks for a given document ID."""
        removed = self.vector_store.delete_document(document_id)
        return removed > 0

    def search(
        self,
        query: str,
        retrieval_mode: str = "hybrid",
        top_k: int = DEFAULT_TOP_K,
        document_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the vector store using dense or hybrid retrieval."""
        where = None
        if document_filter:
            where = {"document_id": {"$in": document_filter}} if len(document_filter) > 1 else {"document_id": document_filter[0]}

        if retrieval_mode == "semantic":
            dense_retriever = DenseRetriever(self.vector_store, self.embedder)
            return dense_retriever.retrieve(query, top_k=top_k, where=where)

        if retrieval_mode == "hybrid":
            dense_retriever = DenseRetriever(self.vector_store, self.embedder)
            all_chunks = self.vector_store.collection.get(include=["metadatas", "documents"])
            chunk_records = []
            for idx, chunk_id in enumerate(all_chunks.get("ids", [])):
                metadatas = all_chunks.get("metadatas", [])[idx]
                if metadatas is None:
                    continue
                if document_filter:
                    doc_filter_set = set(document_filter)
                    if metadatas.get("document_id") not in doc_filter_set:
                        continue
                chunk_records.append({
                    "chunk_id": chunk_id,
                    "text": all_chunks.get("documents", [])[idx],
                    "source": metadatas.get("source"),
                    "document_id": metadatas.get("document_id"),
                    "page_number": int(metadatas.get("page_number", 0)),
                    "chunk_index": int(metadatas.get("chunk_index", 0)),
                })
            bm25_retriever = BM25Retriever(chunk_records)
            hybrid = HybridRetriever(dense_retriever, bm25_retriever)
            return hybrid.retrieve(query, top_k=top_k, where=where)

        raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")

    def answer_question(
        self,
        question: str,
        retrieval_mode: str = "hybrid",
        top_k: int = DEFAULT_TOP_K,
        chat_history: list[dict[str, str]] | None = None,
        document_filter: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run retrieval and generate a grounded answer when Gemini is available."""
        evidence = self.search(question, retrieval_mode=retrieval_mode, top_k=top_k, document_filter=document_filter)

        answer = {
            "question": question,
            "retrieval_mode": retrieval_mode,
            "evidence": evidence,
            "answer": "",
            "citations": [],
            "api_key_available": self.llm_client.available,
        }

        if not evidence:
            answer["answer"] = "No relevant passages were found for this question in the indexed documents."
            return answer

        if not self.llm_client.available:
            answer["answer"] = (
                "LLM generation is unavailable because no Gemini API key is configured. "
                "Retrieved evidence is shown below for manual review."
            )
            return answer

        try:
            generated = self.llm_client.generate_answer(question, evidence, chat_history or [])
            answer["answer"] = generated
            answer["citations"] = [
                {
                    "label": f"[S{index}]",
                    "source": item.get("source"),
                    "page_number": item.get("page_number"),
                    "document_id": item.get("document_id"),
                }
                for index, item in enumerate(evidence, start=1)
            ]
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("LLM answer generation failed")
            answer["answer"] = f"Answer generation failed: {exc}"

        return answer

    def get_document_metadata(self, document_id: str) -> dict[str, Any]:
        """Return metadata for a stored document ID."""
        records = self.vector_store.collection.get(where={"document_id": document_id}, include=["metadatas"])
        if not records.get("metadatas"):
            return {"document_id": document_id, "chunks": 0, "sources": []}

        sources = sorted({item.get("source") for item in records["metadatas"] if item.get("source")})
        return {"document_id": document_id, "chunks": len(records.get("metadatas", [])), "sources": sources}

    def export_index(self, output_path: str | Path) -> None:
        """Export the full vector DB metadata to a JSON file for diagnostics."""
        payload = self.vector_store.collection.get(include=["documents", "metadatas", "embeddings"])
        data = {
            "ids": payload.get("ids", []),
            "documents": payload.get("documents", []),
            "metadatas": payload.get("metadatas", []),
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
