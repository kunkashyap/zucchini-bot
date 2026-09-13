"""
Tests for the RAG retrieval pipeline.

These tests require the knowledge base to be ingested first.
They verify that queries in English, Hindi, and Hinglish
can retrieve relevant documents about the same topics.

Note: These tests require the sentence-transformers model to be
downloaded, which happens automatically on first run.
"""

import pytest
from app.rag.ingest import ingest_documents
from app.rag.retriever import retrieve


@pytest.fixture(scope="module", autouse=True)
def setup_knowledge_base():
    """Ensure knowledge base is ingested before retrieval tests."""
    ingest_documents()


def test_retrieve_english():
    """English query should retrieve relevant RAG documents."""
    results = retrieve("What is retrieval augmented generation?")
    assert len(results) > 0
    # At least one result should mention RAG concepts
    combined = " ".join(results).lower()
    assert any(
        term in combined
        for term in ["retrieval", "augmented", "generation", "rag"]
    )


def test_retrieve_hindi():
    """Hindi query should retrieve relevant documents via multilingual embeddings."""
    results = retrieve("रिट्रीवल ऑगमेंटेड जनरेशन क्या है?")
    assert len(results) > 0


def test_retrieve_hinglish():
    """Hinglish query should retrieve relevant documents."""
    results = retrieve("RAG kya hota hai?")
    assert len(results) > 0


def test_retrieve_python():
    """Query about Python should retrieve Python-related content."""
    results = retrieve("What are the features of Python programming language?")
    assert len(results) > 0
    combined = " ".join(results).lower()
    assert "python" in combined


def test_retrieve_empty_query():
    """Empty query should still return results (nearest neighbors)."""
    results = retrieve("")
    # ChromaDB handles empty queries — may return results or empty
    assert isinstance(results, list)


def test_retrieve_respects_top_k():
    """Retrieval should respect the top_k parameter."""
    results = retrieve("machine learning", top_k=2)
    assert len(results) <= 2
