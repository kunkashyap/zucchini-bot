"""
Zucchini — Multilingual Embedding Module

Loads a multilingual sentence-transformer model and provides a ChromaDB-compatible
embedding function. The model maps text from multiple languages (including English,
Hindi, and Hinglish) into a shared vector space, enabling cross-lingual semantic search.

Why multilingual embeddings instead of translating everything to English?
    Translation adds latency, complexity, and can lose nuance — especially for
    Hinglish (code-mixed text that has no standard translation pipeline).
    Multilingual embeddings directly encode semantic meaning across languages
    into the same vector space, so "What is RAG?" and "RAG kya hai?" naturally
    end up near each other without any translation step.
"""

import logging
from typing import Union
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Module-level model cache — loaded once, reused across all calls
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the sentence-transformer model (cached after first call)."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s (this may take a moment)...", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully. Dimension: %d", _model.get_embedding_dimension())
    return _model


class MultilingualEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    ChromaDB-compatible embedding function using a multilingual
    sentence-transformer model.
    """

    def __init__(self):
        super().__init__()

    def name(self) -> str:
        return "multilingual-minilm-l12-v2"

    def __call__(self, input: Union[list[str], str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            input: A single string or list of strings to embed.

        Returns:
            List of embedding vectors (list of floats).
        """
        model = _get_model()
        if isinstance(input, str):
            input = [input]
        embeddings = model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


# Singleton embedding function instance
_embedding_fn: MultilingualEmbeddingFunction | None = None


def get_embedding_function() -> MultilingualEmbeddingFunction:
    """Return the singleton embedding function."""
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = MultilingualEmbeddingFunction()
    return _embedding_fn
