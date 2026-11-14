from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db" / "chroma"
EVALUATION_DIR = DATA_DIR / "evaluation"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 5
DEFAULT_RETRIEVAL_MODE = "hybrid"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "RAW_PDFS_DIR",
    "PROCESSED_DIR",
    "VECTOR_DB_DIR",
    "EVALUATION_DIR",
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_DIMENSION",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_TOP_K",
    "DEFAULT_RETRIEVAL_MODE",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
]
