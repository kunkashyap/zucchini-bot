"""
Zucchini — API Routes

Defines the FastAPI endpoints:
    - GET  /api/health  — Health check
    - POST /api/chat    — Process a chat message through the LangGraph workflow
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.graph.workflow import run_workflow

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""
    message: str = Field(..., description="The user's message", max_length=2000)
    history: list[ChatMessage] = Field(
        default=[],
        description="Prior conversation messages for context",
    )


class ChatResponse(BaseModel):
    """Response returned to the frontend."""
    response: str = Field(..., description="The assistant's reply")
    language: str = Field(..., description="Detected language of the query")
    metadata: dict = Field(default={}, description="Optional metadata")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a user message through the Zucchini workflow.

    1. Validates the input
    2. Runs the LangGraph pipeline (detect → rewrite → retrieve → generate → validate)
    3. Returns the response with detected language
    """
    # Validate non-empty message
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    logger.info("Chat request received (%d chars)", len(message))

    # Convert Pydantic models to plain dicts for the workflow
    history = [{"role": msg.role, "content": msg.content} for msg in request.history]

    try:
        # Run the LangGraph workflow
        result = run_workflow(query=message, conversation_history=history)

        response_text = result.get("response", "")
        language = result.get("language", "english")
        retrieved_count = len(result.get("retrieved_documents", []))

        if not response_text:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate a response. Please try again.",
            )

        return ChatResponse(
            response=response_text,
            language=language,
            metadata={"retrieval_count": retrieved_count},
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error("Chat endpoint error: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again later.",
        )
