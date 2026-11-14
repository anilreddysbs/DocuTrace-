"""BM25 retriever."""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Normalize a text string into lowercase token slices."""
    text = text.lower()
    return re.findall(r"[a-z0-9_]+", text)


class BM25Retriever:
    """Keyword retrieval over chunk text using BM25."""

    def __init__(self, chunk_records: list[dict[str, Any]] | None = None) -> None:
        self.chunk_records = list(chunk_records or [])
        self.index: BM25Okapi | None = None
        if self.chunk_records:
            self._build_index()

    def _build_index(self) -> None:
        corpus = [tokenize(record["text"]) for record in self.chunk_records]
        self.index = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the best BM25 matches for a query."""
        if not self.chunk_records or self.index is None:
            return []
        if not query or not query.strip():
            return []
        query_tokens = tokenize(query)
        scores = self.index.get_scores(query_tokens)
        ranked_indexes = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        results = []
        for idx in ranked_indexes:
            record = self.chunk_records[idx].copy()
            record["bm25_score"] = float(scores[idx])
            results.append(record)
        return results
