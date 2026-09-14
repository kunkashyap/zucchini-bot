"""
Zucchini — LangGraph Workflow

Compiles the stateful graph that processes each user message.

Graph structure:
    START → detect_language → rewrite_query → retrieve_documents
          → generate_response → validate_response → (conditional)
              ├─ valid → END
              └─ invalid → generate_response (retry, max 2 times)

The graph is compiled once and reused for all requests.
"""

import logging
from langgraph.graph import StateGraph, END, START
from app.graph.state import GraphState
from app.graph.nodes import (
    detect_language,
    rewrite_query,
    retrieve_documents,
    generate_response,
    validate_response,
)

logger = logging.getLogger(__name__)

# Module-level compiled graph (built once)
_compiled_graph = None


def _should_retry(state: GraphState) -> str:
    """
    Conditional edge after validation.
    Returns the next node name or END.
    """
    if state.get("validation_status") == "valid":
        return END
    # Invalid — retry generation
    return "generate_response"


def _build_graph() -> StateGraph:
    """
    Build and compile the LangGraph workflow.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("detect_language", detect_language)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("generate_response", generate_response)
    graph.add_node("validate_response", validate_response)

    # Add edges (linear pipeline)
    graph.add_edge(START, "detect_language")
    graph.add_edge("detect_language", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_documents")
    graph.add_edge("retrieve_documents", "generate_response")
    graph.add_edge("generate_response", "validate_response")

    # Conditional edge: retry or finish
    graph.add_conditional_edges(
        "validate_response",
        _should_retry,
        {
            END: END,
            "generate_response": "generate_response",
        },
    )

    return graph.compile()


def get_workflow():
    """Return the compiled graph (built once, reused)."""
    global _compiled_graph
    if _compiled_graph is None:
        logger.info("Compiling LangGraph workflow...")
        _compiled_graph = _build_graph()
        logger.info("LangGraph workflow compiled successfully")
    return _compiled_graph


def run_workflow(query: str, conversation_history: list[dict] | None = None) -> dict:
    """
    Execute the full workflow for a user message.

    Args:
        query: The user's input message.
        conversation_history: Optional list of prior messages [{role, content}, ...].

    Returns:
        The final graph state dict containing 'response', 'language', etc.

    Raises:
        Exception: If the workflow fails catastrophically.
    """
    if conversation_history is None:
        conversation_history = []
    workflow = get_workflow()

    # Initialize the state
    initial_state: GraphState = {
        "query": query,
        "language": "",
        "rewritten_query": "",
        "conversation_history": conversation_history,
        "retrieved_documents": [],
        "response": "",
        "validation_status": "pending",
        "retry_count": 0,
        "error": "",
    }

    try:
        logger.info("Running workflow for query: '%s'", query[:80])
        final_state = workflow.invoke(initial_state)
        logger.info(
            "Workflow complete. Language: %s, Response length: %d",
            final_state.get("language", "unknown"),
            len(final_state.get("response", "")),
        )
        return final_state
    except Exception as e:
        logger.error("Workflow execution failed: %s", str(e))
        raise
