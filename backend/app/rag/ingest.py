"""
Zucchini — Document Ingestion

Reads text files from the knowledge base directory, chunks them,
generates multilingual embeddings, and stores them in ChromaDB.

Ingestion is idempotent: if the collection already contains documents,
it skips re-ingestion. Delete the chroma_db directory to force re-ingestion.
"""

import os
import logging
import chromadb
from app.config import KNOWLEDGE_DIR, CHROMA_PATH, CHROMA_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP
from app.rag.embeddings import get_embedding_function

logger = logging.getLogger(__name__)


def _read_documents(directory: str) -> list[dict]:
    """
    Read all .txt files from the knowledge directory.

    Returns:
        List of dicts with 'content' and 'source' keys.
    """
    documents = []
    if not os.path.exists(directory):
        logger.warning("Knowledge directory not found: %s", directory)
        return documents

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            documents.append({"content": content, "source": filename})
            logger.debug("Read document: %s (%d chars)", filename, len(content))

    return documents


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks.

    Uses a simple character-based approach. For a production system,
    you might use sentence-aware or paragraph-aware chunking.

    Args:
        text: The full text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Try to break at a sentence boundary (period, newline) for cleaner chunks
        if end < len(text):
            # Look for the last sentence-ending punctuation in the chunk
            for sep in ["\n\n", "\n", ". ", "। "]:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size * 0.5:  # Only break if past halfway
                    chunk = chunk[: last_sep + len(sep)]
                    end = start + len(chunk)
                    break
        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if c]  # Remove empty chunks


def ingest_documents() -> None:
    """
    Load knowledge base documents, chunk them, embed, and store in ChromaDB.

    This function is idempotent — it skips ingestion if documents already exist.
    """
    # Initialize ChromaDB persistent client
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = get_embedding_function()

    # Get or create collection
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_fn,
    )

    # Check if already ingested
    existing_count = collection.count()
    if existing_count > 0:
        logger.info(
            "Knowledge base already ingested (%d chunks). Skipping.",
            existing_count,
        )
        return

    # Read documents
    documents = _read_documents(KNOWLEDGE_DIR)
    if not documents:
        logger.warning("No documents found in %s. Knowledge base is empty.", KNOWLEDGE_DIR)
        return

    # Chunk and prepare for insertion
    all_chunks = []
    all_metadatas = []
    all_ids = []

    for doc in documents:
        chunks = _chunk_text(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['source']}__chunk_{i}"
            all_chunks.append(chunk)
            all_metadatas.append({"source": doc["source"], "chunk_index": i})
            all_ids.append(chunk_id)

    # Insert into ChromaDB (embeddings generated automatically by the collection)
    logger.info("Ingesting %d chunks from %d documents...", len(all_chunks), len(documents))
    collection.add(
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids,
    )
    logger.info("Knowledge base ingestion complete. %d chunks stored.", len(all_chunks))
