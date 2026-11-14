from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import fitz

logger = logging.getLogger(__name__)


def stable_document_id(file_bytes: bytes, original_name: str) -> str:
    """Create a stable, content-based document identifier."""
    digest = hashlib.sha256(file_bytes).hexdigest()[:16]
    safe_name = Path(original_name).stem.lower().replace(" ", "_")
    return f"{safe_name}_{digest}"


def validate_uploaded_pdf(file_obj: Any, max_size_mb: int = 20) -> tuple[bytes, str]:
    """Validate uploaded PDFs and return raw bytes plus a stable filename."""
    if file_obj is None:
        raise ValueError("No file uploaded.")

    original_name = getattr(file_obj, "name", "uploaded.pdf")
    if not original_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are allowed.")

    file_bytes = file_obj.read()
    max_bytes = max_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise ValueError(f"File exceeds {max_size_mb}MB limit.")
    if len(file_bytes) == 0:
        raise ValueError("Uploaded PDF is empty.")

    return file_bytes, original_name


def extract_pdf_pages(file_bytes: bytes, original_name: str) -> list[dict[str, Any]]:
    """Extract text from each page while preserving metadata and OCR warnings."""
    document_id = stable_document_id(file_bytes, original_name)
    extracted_pages: list[dict[str, Any]] = []

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf_doc:
            for page_number in range(pdf_doc.page_count):
                page = pdf_doc[page_number]
                text = (page.get_text("text") or "").strip()
                scanned = not bool(text)
                extracted_pages.append(
                    {
                        "source": original_name,
                        "document_id": document_id,
                        "page_number": page_number + 1,
                        "title": Path(original_name).stem,
                        "text": text,
                        "is_scanned": scanned,
                        "ocr_required": scanned,
                        "page_width": float(page.rect.width),
                        "page_height": float(page.rect.height),
                    }
                )
    except Exception as exc:  # pragma: no cover - defensive protection
        logger.exception("PDF extraction failed for %s", original_name)
        raise ValueError(f"Unable to read PDF '{original_name}': {exc}") from exc

    return extracted_pages
