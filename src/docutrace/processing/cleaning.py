"""Text cleaning helpers."""

from __future__ import annotations

import re
from typing import Any


def clean_text(raw_text: str) -> str:
    """Normalize page text while preserving readable prose and metadata."""
    if raw_text is None:
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")
    text = "".join(ch if ch.isprintable() or ch in "\n\t" else " " for ch in text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_page_record(page: dict[str, Any]) -> dict[str, Any]:
    """Return a cleaned page payload consistent with extracted metadata."""
    cleaned = dict(page)
    cleaned["text"] = clean_text(page.get("text", ""))
    return cleaned
