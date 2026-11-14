"""Retrieval evaluation utilities for DocuTrace."""

from __future__ import annotations

from typing import Any


EVALUATION_DATASET = [
    {
        "query": "What should I check if a Hive query returns zero rows for an expected business date?",
        "relevant_documents": ["hive_troubleshooting_guide.pdf"],
        "relevant_pages": [1],
        "relevant_chunk_ids": [
            "hive_troubleshooting_guide::p1::c001",
            "hive_troubleshooting_guide::p1::c002",
        ],
    },
    {
        "query": "What do I do when the CUSTOMER_DAILY_LOAD pipeline fails because an upstream partition is missing?",
        "relevant_documents": ["data_pipeline_runbook.pdf"],
        "relevant_pages": [2],
        "relevant_chunk_ids": [
            "data_pipeline_runbook::p2::c001",
            "data_pipeline_runbook::p2::c002",
        ],
    },
    {
        "query": "Before comparing source and target data, which validation checks matter?",
        "relevant_documents": ["hive_troubleshooting_guide.pdf"],
        "relevant_pages": [2],
        "relevant_chunk_ids": [
            "hive_troubleshooting_guide::p2::c001",
            "hive_troubleshooting_guide::p2::c002",
        ],
    },
]


def precision_at_k(results: list[dict[str, Any]], relevant_ids: set[str], k: int) -> float:
    """Compute precision@K for a set of retrieved results."""
    if not results:
        return 0.0
    top_hits = results[:k]
    retrieved_ids = {item.get("chunk_id") for item in top_hits if item.get("chunk_id")}
    hits = len(retrieved_ids & relevant_ids)
    return hits / max(len(top_hits), 1)


def recall_at_k(results: list[dict[str, Any]], relevant_ids: set[str], k: int) -> float:
    """Compute recall@K for a set of retrieved results."""
    if not relevant_ids:
        return 0.0
    top_hits = results[:k]
    retrieved_ids = {item.get("chunk_id") for item in top_hits if item.get("chunk_id")}
    hits = len(retrieved_ids & relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(results: list[dict[str, Any]], relevant_ids: set[str]) -> float:
    """Compute MRR from the retrieved ranking."""
    if not relevant_ids:
        return 0.0
    for rank, item in enumerate(results, start=1):
        if item.get("chunk_id") in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(results: list[dict[str, Any]], query_record: dict[str, Any], k: int = 3) -> dict[str, float]:
    """Evaluate a single query against a gold set of relevant chunk IDs."""
    relevant_ids = set(query_record["relevant_chunk_ids"])
    return {
        "precision_at_k": precision_at_k(results, relevant_ids, k),
        "recall_at_k": recall_at_k(results, relevant_ids, k),
        "mrr": reciprocal_rank(results, relevant_ids),
    }


def summarize_evaluation(results_by_query: list[dict[str, Any]]) -> dict[str, float]:
    """Average metrics over a full evaluation batch."""
    metrics = [
        metric
        for result in results_by_query
        for metric in [result["precision_at_k"], result["recall_at_k"], result["mrr"]]
    ]
    if not metrics:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}

    return {
        "precision_at_k": sum(result["precision_at_k"] for result in results_by_query) / len(results_by_query),
        "recall_at_k": sum(result["recall_at_k"] for result in results_by_query) / len(results_by_query),
        "mrr": sum(result["mrr"] for result in results_by_query) / len(results_by_query),
    }
