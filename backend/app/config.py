"""
Zucchini — Central Configuration

All configurable values are loaded from environment variables with sensible defaults.
This is the single source of truth for configuration across the backend.
"""

import os
from dotenv import load_dotenv

# Load .env file from the backend directory
load_dotenv()


# --- LLM ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

# --- Embeddings ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# --- Vector Database ---
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION: str = "zucchini_knowledge"

# --- Retrieval ---
TOP_K: int = int(os.getenv("TOP_K", "4"))

# --- Conversation ---
MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "10"))

# --- Chunking ---
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# --- Paths ---
KNOWLEDGE_DIR: str = os.getenv("KNOWLEDGE_DIR", "./data/knowledge")

# --- Server ---
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
