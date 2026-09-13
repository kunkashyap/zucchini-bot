"""
Tests for the API endpoints.

The health endpoint can be tested without any external dependencies.
The chat endpoint requires OPENAI_API_KEY and is skipped if not available.
"""

import os
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
def transport():
    return ASGITransport(app=app)


@pytest.mark.asyncio
async def test_health(transport):
    """Health endpoint should always return ok."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_chat_empty_message(transport):
    """Empty message should return 400."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={"message": "", "history": []},
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_chat_missing_message(transport):
    """Missing message field should return 422 (validation error)."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={"history": []},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping live LLM test",
)
async def test_chat_valid_request(transport):
    """Valid chat request should return a response with language."""
    async with AsyncClient(
        transport=transport, base_url="http://test", timeout=60.0
    ) as client:
        response = await client.post(
            "/api/chat",
            json={"message": "What is RAG?", "history": []},
        )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "language" in data
    assert len(data["response"]) > 0
