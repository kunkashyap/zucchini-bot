"""
Zucchini — FastAPI Application Entry Point

Initializes the FastAPI app, configures CORS, includes API routes,
and runs knowledge base ingestion on startup.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.rag.ingest import ingest_documents
from app.services.llm import check_ollama_health
from app.config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, OPENAI_API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler.
    Runs knowledge base ingestion on server startup.
    """
    logger.info("=" * 50)
    logger.info("Zucchini backend starting...")
    logger.info("=" * 50)

    # Check LLM Provider status
    if LLM_PROVIDER.lower() == "ollama":
        if check_ollama_health():
            logger.info("Ollama connected at %s (Model: %s)", OLLAMA_BASE_URL, OLLAMA_MODEL)
        else:
            logger.warning(
                "Ollama service is NOT reachable at %s! "
                "Please make sure Ollama is running (`ollama serve`).", OLLAMA_BASE_URL
            )
    elif LLM_PROVIDER.lower() == "openai":
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is not set!")
        else:
            logger.info("OPENAI_API_KEY found")

    # Ingest knowledge base
    try:
        ingest_documents()
    except Exception as e:
        logger.error("Knowledge base ingestion failed: %s", str(e))
        logger.warning("The chatbot will work but without knowledge base retrieval.")

    logger.info("Zucchini backend ready")
    logger.info("=" * 50)
    yield


# Create the FastAPI application
app = FastAPI(
    title="Zucchini",
    description="Multilingual Intelligence Assistant — A RAG-powered chatbot supporting English, Hindi, and Hinglish.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(chat_router)
