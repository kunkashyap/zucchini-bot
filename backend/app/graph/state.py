"""
Zucchini — LangGraph State Definition

Defines the typed state that flows through the LangGraph workflow.
Every node reads from and writes to this shared state object.
"""

from typing import TypedDict


class GraphState(TypedDict):
    """
    State object for the Zucchini LangGraph workflow.

    Each field is documented below. Nodes update specific fields
    and pass the rest through unchanged.
    """

    # The user's raw input message
    query: str

    # Detected language: 'english', 'hindi', or 'hinglish'
    language: str

    # Query rewritten to be standalone and retrieval-friendly.
    # For simple/standalone queries, this equals the original query.
    rewritten_query: str

    # Conversation history as a list of {role: str, content: str} dicts.
    # 'role' is either 'user' or 'assistant'.
    conversation_history: list[dict]

    # Relevant text chunks retrieved from the vector database.
    # Empty list if no relevant documents were found.
    retrieved_documents: list[str]

    # The LLM-generated response to return to the user.
    response: str

    # Validation result: 'valid', 'invalid', or 'pending'
    validation_status: str

    # Number of times response generation has been retried.
    # Used to prevent infinite retry loops (max 2).
    retry_count: int

    # Error information, empty string if no error occurred.
    error: str
