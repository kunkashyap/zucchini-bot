"""
Zucchini — Language Detection Service

Detects whether user input is English, Hindi, or Hinglish (code-mixed).

IMPORTANT LIMITATION:
    Hinglish detection is heuristic-based. There is no standard language code
    for Hinglish, and conventional language detection libraries cannot reliably
    identify code-mixed Hindi-English text. This module uses a combination of
    script detection and keyword matching as a practical approximation.

Detection strategy (three tiers):
    1. Script check — if text contains significant Devanagari characters, classify as Hindi.
    2. Hinglish heuristic — if text is mostly Latin but contains common romanized
       Hindi words (kya, hai, kaise, etc.), classify as Hinglish.
    3. Fallback — use the langdetect library. If it detects Hindi, return Hindi.
       Otherwise default to English.
"""

import re
import logging
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)

# Common romanized Hindi words used in Hinglish conversation.
# This list is intentionally conservative to reduce false positives.
HINGLISH_MARKERS = {
    "kya", "hai", "kaise", "hota", "hoti", "nahi", "aur", "yeh", "woh",
    "karo", "mein", "toh", "bhi", "agar", "lekin", "kyun", "matlab",
    "accha", "acha", "theek", "thik", "tha", "thi", "hain", "kar",
    "raha", "rahi", "wala", "wali", "abhi", "sab", "kuch", "bahut",
    "bol", "baat", "samajh", "pata", "dekh", "soch", "liye", "uska",
    "iska", "unka", "kaisa", "kaisi", "kitna", "kitni", "jaise",
}

# Devanagari Unicode range
DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")


def _devanagari_ratio(text: str) -> float:
    """Return the fraction of characters that are Devanagari."""
    if not text:
        return 0.0
    devanagari_count = len(DEVANAGARI_PATTERN.findall(text))
    # Only count actual characters (not spaces/punctuation)
    alpha_chars = sum(1 for c in text if c.isalpha() or "\u0900" <= c <= "\u097F")
    if alpha_chars == 0:
        return 0.0
    return devanagari_count / alpha_chars


def _count_hinglish_markers(text: str) -> int:
    """Count how many Hinglish marker words appear in the text."""
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return len(words & HINGLISH_MARKERS)


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.

    Args:
        text: User's input string.

    Returns:
        One of: 'english', 'hindi', 'hinglish'
    """
    if not text or not text.strip():
        logger.debug("Empty input, defaulting to english")
        return "english"

    text = text.strip()

    # --- Tier 1: Devanagari script check ---
    dev_ratio = _devanagari_ratio(text)
    if dev_ratio > 0.3:
        # Significant Devanagari content
        # Check if it's mixed with Latin (could be Hinglish in Devanagari + English)
        latin_words = re.findall(r"[a-zA-Z]{2,}", text)
        # If there are multiple substantial Latin words mixed with Devanagari, it's Hinglish.
        # But a single short acronym (RAG, ML, AI) in Devanagari text is still Hindi.
        substantial_latin = [w for w in latin_words if len(w) > 4]
        if len(latin_words) > 1 and dev_ratio < 0.8:
            logger.info("Detected language: hinglish (mixed scripts)")
            return "hinglish"
        if substantial_latin and dev_ratio < 0.8:
            logger.info("Detected language: hinglish (mixed scripts)")
            return "hinglish"
        logger.info("Detected language: hindi (Devanagari content)")
        return "hindi"

    # --- Tier 2: Hinglish heuristic (Latin script with Hindi words) ---
    marker_count = _count_hinglish_markers(text)
    if marker_count >= 2:
        logger.info(
            "Detected language: hinglish (%d marker words found)", marker_count
        )
        return "hinglish"

    # --- Tier 3: Fallback to langdetect ---
    try:
        detected = detect(text)
        if detected == "hi":
            logger.info("Detected language: hindi (via langdetect)")
            return "hindi"
    except LangDetectException:
        logger.debug("langdetect failed, defaulting to english")

    logger.info("Detected language: english")
    return "english"
