"""Chunking helpers."""

from __future__ import annotations

import math
from typing import Any


def _split_on_separator(text: str, separator: str) -> list[str]:
    if not text:
        return []
    if separator == "":
        return [text]
    return [segment.strip() for segment in text.split(separator) if segment.strip()]


def recursive_character_chunks(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
    separators: tuple[str, ...] = ("\n\n", "\n", " ", ""),
) -> list[str]:
    """Split text recursively by section boundaries while preserving overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    if not text:
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    for separator in separators:
        parts = _split_on_separator(text, separator)
        if len(parts) <= 1:
            continue

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for part in parts:
            candidate = (separator.join(current + [part])).strip()
            if current and len(candidate) > chunk_size:
                chunks.append(separator.join(current).strip())
                if overlap > 0:
                    overlap_prefix = "".join(current[-1:])
                    if overlap_prefix:
                        current = [overlap_prefix]
                        current_len = len(overlap_prefix)
                    else:
                        current = []
                        current_len = 0
                else:
                    current = []
                    current_len = 0

            current.append(part)
            current_len = len(separator.join(current).strip())

        if current:
            chunks.append(separator.join(current).strip())
        if chunks:
            if overlap > 0 and len(chunks) > 1:
                merged: list[str] = []
                for idx, chunk in enumerate(chunks):
                    if idx == 0:
                        merged.append(chunk)
                        continue
                    previous = merged[-1]
                    overlap_text = previous[-overlap:] if overlap < len(previous) else previous
                    if chunk.startswith(overlap_text):
                        chunk = chunk[len(overlap_text):].lstrip()
                    if chunk:
                        merged.append(chunk)
                return [item for item in merged if item]
            return [item for item in chunks if item]

    step = max(chunk_size - overlap, 1)
    fallback: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start:start + chunk_size].strip()
        if piece:
            fallback.append(piece)
    return fallback


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 800,
    overlap: int = 120,
    strategy: str = "recursive_character",
) -> list[dict[str, Any]]:
    """Chunk extracted pages while preserving source and document metadata."""
    chunk_records: list[dict[str, Any]] = []
    for page in pages:
        text = (page.get("text") or "").strip()
        if not text:
            continue
        page_chunks = recursive_character_chunks(text, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk_text in enumerate(page_chunks, start=1):
            chunk_id = (
                f"{page['document_id']}::p{page['page_number']}::c{chunk_index:03d}"
            )
            chunk_records.append(
                {
                    "chunk_id": chunk_id,
                    "source": page["source"],
                    "document_id": page["document_id"],
                    "title": page.get("title", ""),
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                    "strategy": strategy,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                }
            )
    return chunk_records


def compute_average_chunk_size(chunk_records: list[dict[str, Any]]) -> float:
    """Return the mean chunk length for a collection of chunk records."""
    if not chunk_records:
        return 0.0
    total = sum(record.get("char_count", 0) for record in chunk_records)
    return total / len(chunk_records)
