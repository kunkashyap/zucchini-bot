"""
Zucchini — LLM Service

Clean abstraction over the OpenAI-compatible LLM.
Loads model configuration from config, provides a reusable generate() helper.
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import OPENAI_API_KEY, MODEL_NAME

logger = logging.getLogger(__name__)

# Module-level LLM instance (created once, reused)
_llm_instance: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    """
    Return a singleton ChatOpenAI instance.
    The model is created once and reused across all calls.
    """
    global _llm_instance
    if _llm_instance is None:
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or set it as an environment variable."
            )
        _llm_instance = ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=0.7,
            max_tokens=1024,
        )
        logger.info("LLM initialized: model=%s", MODEL_NAME)
    return _llm_instance


def generate(prompt: str, system_message: str = "") -> str:
    """
    Send a prompt to the LLM and return the response text.

    Args:
        prompt: The user/human message.
        system_message: Optional system instructions.

    Returns:
        The LLM's response as a string.

    Raises:
        ValueError: If the LLM call fails.
    """
    llm = get_llm()
    messages = []
    if system_message:
        messages.append(SystemMessage(content=system_message))
    messages.append(HumanMessage(content=prompt))

    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error("LLM generation failed: %s", str(e))
        raise ValueError(f"LLM call failed: {str(e)}") from e
