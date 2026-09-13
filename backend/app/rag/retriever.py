"""
Zucchini — Document Retriever

Queries the ChromaDB vector database using multilingual embeddings
to find the most relevant knowledge base chunks for a given query.
"""

import logging
import chromadb
from app.config import CHROMA_PATH, CHROMA_COLLECTION, TOP_K
from app.rag.embeddings import get_embedding_function

logger = logging.getLogger(__name__)


def retrieve(query: str, top_k: int | None = None) -> list[str]:
    """
    Retrieve the most relevant document chunks for a query.

    Uses the multilingual embedding function so that queries in English,
    Hindi, or Hinglish can all match relevant English-language knowledge
    base content.

    Args:
        query: The search query (can be in any supported language).
        top_k: Number of results to return. Defaults to config.TOP_K.

    Returns:
        List of relevant text chunks, ordered by similarity.
        Returns an empty list if no documents are found or if the
        collection doesn't exist.
    """
    if top_k is None:
        top_k = TOP_K

    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        embedding_fn = get_embedding_function()

        collection = client.get_collection(
            name=CHROMA_COLLECTION,
            embedding_function=embedding_fn,
        )

        if collection.count() == 0:
            logger.warning("Knowledge base is empty. No documents to retrieve.")
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
        )

        documents = results.get("documents", [[]])[0]
        logger.info("Retrieved %d chunks for query", len(documents))
        return documents

    except Exception as e:
        logger.error("Retrieval failed: %s", str(e))
        return []
