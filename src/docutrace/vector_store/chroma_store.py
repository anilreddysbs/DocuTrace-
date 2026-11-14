"""Chroma vector storage wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from docutrace.config import EMBEDDING_DIMENSION, VECTOR_DB_DIR

logger = logging.getLogger(__name__)


class ChromaChunkStore:
    """Persistent local Chroma collection for document chunks."""

    def __init__(
        self,
        persist_dir: str | Path = VECTOR_DB_DIR,
        collection_name: str = "docutrace_chunks",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunk_records: list[dict[str, Any]], embeddings: np.ndarray) -> None:
        """Insert or replace chunks and metadata into Chroma."""
        if len(chunk_records) != len(embeddings):
            raise ValueError("Chunk records and embedding rows must have the same length.")
        ids = [record["chunk_id"] for record in chunk_records]
        texts = [record["text"] for record in chunk_records]
        metadatas = [
            {
                "chunk_id": record["chunk_id"],
                "source": record["source"],
                "document_id": record["document_id"],
                "title": record.get("title", ""),
                "page_number": int(record["page_number"]),
                "chunk_index": int(record["chunk_index"]),
                "strategy": record.get("strategy", "recursive_character"),
            }
            for record in chunk_records
        ]
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=[vector.tolist() for vector in embeddings],
            metadatas=metadatas,
        )

    def query(self, query_embedding: np.ndarray, top_k: int = 5, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Query top similar chunks, returning dense similarity and metadata."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        output: list[dict[str, Any]] = []
        for idx, chunk_id in enumerate(results.get("ids", [[]])[0]):
            metadata = results["metadatas"][0][idx]
            distance = float(results["distances"][0][idx])
            output.append(
                {
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][idx],
                    "source": metadata.get("source"),
                    "document_id": metadata.get("document_id"),
                    "page_number": int(metadata.get("page_number", 0)),
                    "chunk_index": int(metadata.get("chunk_index", 0)),
                    "distance": distance,
                    "score": 1.0 - distance,
                    "metadata": metadata,
                }
            )
        return output

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks associated with a document ID and return deleted count."""
        existing = self.collection.get(where={"document_id": document_id}, include=[])
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def list_documents(self) -> list[str]:
        """List all unique document IDs stored in Chroma."""
        all_items = self.collection.get(include=["metadatas"])
        documents = []
        for metadata in all_items.get("metadatas", []):
            if metadata and "document_id" in metadata:
                documents.append(metadata["document_id"])
        return sorted(set(documents))

    def get_stats(self) -> dict[str, int]:
        """Return total chunk count and number of unique indexed documents."""
        collection = self.collection.get(include=["metadatas"])
        total = len(collection.get("ids", []))
        unique_docs = len(set(metadata["document_id"] for metadata in collection.get("metadatas", []) if metadata and "document_id" in metadata))
        return {"total_chunks": total, "total_documents": unique_docs}

    def count_by_document(self, document_id: str) -> int:
        """Get the number of chunks for one document."""
        return len(self.collection.get(where={"document_id": document_id}, include=[]).get("ids", []))
