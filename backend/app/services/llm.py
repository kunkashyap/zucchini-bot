"""
Zucchini — LLM Service

Provides an abstraction over ChatOllama (local LFM2 model) or ChatOpenAI.
Loads model configuration from config, provides a reusable generate() helper.
"""

import logging
import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    MODEL_NAME,
)

logger = logging.getLogger(__name__)

# Module-level LLM instance (created once, reused)
_llm_instance: BaseChatModel | None = None


def check_ollama_health() -> bool:
    """Check if the local Ollama instance is reachable."""
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception as e:
        logger.warning("Ollama health check failed: %s", str(e))
        return False


def get_llm() -> BaseChatModel:
    """
    Return a singleton LLM instance (ChatOllama by default).
    The model is created once and reused across all calls.
    """
    global _llm_instance
    if _llm_instance is None:
        provider = LLM_PROVIDER.lower()
        if provider == "ollama":
            if not check_ollama_health():
                logger.error("Ollama service unreachable at %s", OLLAMA_BASE_URL)
                raise ConnectionError(
                    f"Local AI model is unavailable. Please make sure Ollama is running at {OLLAMA_BASE_URL}."
                )

            logger.info("Initializing ChatOllama: model=%s, base_url=%s", OLLAMA_MODEL, OLLAMA_BASE_URL)
            _llm_instance = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.2,
            )
        elif provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Add it to your .env file."
                )
            logger.info("Initializing ChatOpenAI: model=%s", MODEL_NAME)
            _llm_instance = ChatOpenAI(
                model=MODEL_NAME,
                api_key=OPENAI_API_KEY,
                temperature=0.7,
                max_tokens=1024,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    return _llm_instance


def generate(prompt: str, system_message: str = "") -> str:
    """
    Send a prompt to the LLM and return the response text.

    Args:
        prompt: The user/human message.
        system_message: Optional system instructions.

    Returns:
        The LLM's response as a string.
    """
    try:
        llm = get_llm()
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))

        response = llm.invoke(messages)
        return str(response.content)
    except ConnectionError as ce:
        logger.error("Connection error during LLM generation: %s", str(ce))
        raise ce
    except Exception as e:
        logger.error("LLM generation failed: %s", str(e))
        raise ValueError(f"LLM call failed: {str(e)}") from e
