"""Hybrid retrieval combining dense and BM25 scores."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse two rankings using reciprocal rank fusion."""
    candidate_scores: dict[str, float] = defaultdict(float)
    candidate_payload: dict[str, dict[str, Any]] = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["chunk_id"]
        candidate_scores[chunk_id] += 1.0 / (k + rank)
        candidate_payload.setdefault(chunk_id, result.copy())
        candidate_payload[chunk_id]["dense_rank"] = rank
        candidate_payload[chunk_id]["dense_score"] = result.get("score", 0.0)

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["chunk_id"]
        candidate_scores[chunk_id] += 1.0 / (k + rank)
        candidate_payload.setdefault(chunk_id, result.copy())
        candidate_payload[chunk_id]["bm25_rank"] = rank
        candidate_payload[chunk_id]["bm25_score"] = result.get("bm25_score", 0.0)

    fused = []
    for chunk_id, score in candidate_scores.items():
        payload = candidate_payload.get(chunk_id, {})
        payload["chunk_id"] = chunk_id
        payload["rrf_score"] = float(score)
        payload["retrieval_mode"] = "hybrid"
        fused.append(payload)

    fused.sort(key=lambda item: item.get("rrf_score", 0.0), reverse=True)
    return fused


class HybridRetriever:
    """Dense + BM25 retrieval with reciprocal rank fusion."""

    def __init__(self, dense_retriever: Any, bm25_retriever: Any) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever

    def retrieve(self, query: str, top_k: int = 5, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return blended results from dense and keyword retrieval."""
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k, where=where)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k)
        fused = reciprocal_rank_fusion(dense_results, bm25_results)
        return fused[:top_k]
