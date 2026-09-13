"""
Tests for the language detection service.

These tests verify that the three-tier detection strategy correctly
identifies English, Hindi, and Hinglish inputs.
"""

from app.services.language import detect_language


def test_detect_english():
    """Standard English text should be detected as english."""
    assert detect_language("What is machine learning?") == "english"


def test_detect_english_technical():
    """Technical English text should be detected as english."""
    assert detect_language("How does backpropagation work in neural networks?") == "english"


def test_detect_hindi():
    """Devanagari text should be detected as hindi."""
    assert detect_language("मशीन लर्निंग क्या है?") == "hindi"


def test_detect_hindi_rag():
    """Hindi question about RAG should be detected as hindi."""
    assert detect_language("RAG क्या है?") == "hindi"


def test_detect_hinglish():
    """Romanized Hindi mixed with English should be detected as hinglish."""
    assert detect_language("Machine learning kya hoti hai?") == "hinglish"


def test_detect_hinglish_rag():
    """Hinglish query about RAG."""
    assert detect_language("RAG kya hota hai aur ye useful kyun hai?") == "hinglish"


def test_detect_hinglish_simple():
    """Simple Hinglish with multiple markers."""
    assert detect_language("Yeh kaise kaam karta hai?") == "hinglish"


def test_empty_input():
    """Empty input should default to english."""
    assert detect_language("") == "english"


def test_whitespace_only():
    """Whitespace-only input should default to english."""
    assert detect_language("   ") == "english"
