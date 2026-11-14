"""Embedding wrapper for sentence-transformers."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from docutrace.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class Embedder:
    """Singleton-like sentence-transformer wrapper with CPU default."""

    _instance: "Embedder | None" = None

    def __new__(cls, model_name: str = EMBEDDING_MODEL_NAME) -> "Embedder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.model = SentenceTransformer(model_name, device="cpu")
        return cls._instance

    def embed_texts(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        """Return embeddings for a batch of texts."""
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        vectors = self.model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected embedding dimension {EMBEDDING_DIMENSION}, got {array.shape[1]}"
            )
        return array

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Embed a single query string."""
        return self.embed_texts([query], normalize=normalize)[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the model."""
        return int(self.model.get_sentence_embedding_dimension())
