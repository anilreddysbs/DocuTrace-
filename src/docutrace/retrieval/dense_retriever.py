"""Dense retrieval wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from docutrace.embeddings.embedder import Embedder
from docutrace.vector_store.chroma_store import ChromaChunkStore


class DenseRetriever:
    """Query Chroma with a dense vector search."""

    def __init__(self, vector_store: ChromaChunkStore | None = None, embedder: Embedder | None = None) -> None:
        self.vector_store = vector_store or ChromaChunkStore()
        self.embedder = embedder or Embedder()

    def retrieve(self, query: str, top_k: int = 5, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return dense search results for the query."""
        embedding = self.embedder.embed_query(query)
        results = self.vector_store.query(embedding, top_k=top_k, where=where)
        for item in results:
            item["retrieval_mode"] = "dense"
        return results
