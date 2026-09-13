"""
Zucchini — FastAPI Application Entry Point

Initializes the FastAPI app, configures CORS, includes API routes,
and runs knowledge base ingestion on startup.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.rag.ingest import ingest_documents
from app.config import OPENAI_API_KEY

from contextlib import asynccontextmanager

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

    # Check for API key
    if not OPENAI_API_KEY:
        logger.warning(
            "OPENAI_API_KEY is not set! "
            "The LLM will not work until you add it to your .env file."
        )
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
