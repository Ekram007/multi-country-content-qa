"""LLM provider abstraction and initialization."""

import logging
from functools import cache
from typing import Any, Dict

from langchain_core.language_models import BaseChatModel

from src.core.settings import settings

logger = logging.getLogger(__name__)

# Global LLM instance
_llm_instance: BaseChatModel | None = None


def _create_openai_llm() -> BaseChatModel:
    """Create OpenAI LLM instance."""
    from langchain_openai import ChatOpenAI
    
    if not settings.openai_api_key:
        raise ValueError("OpenAI API key is required but not provided")
    
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.0,
        timeout=30.0,
        max_retries=2,
    )


def _create_google_llm() -> BaseChatModel:
    """Create Google Gemini LLM instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    if not settings.google_api_key:
        raise ValueError("Google API key is required but not provided")
    
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0.0,
        timeout=30.0,
        max_retries=2,
    )


def _create_openrouter_llm() -> BaseChatModel:
    """Create OpenRouter LLM instance."""
    from langchain_openai import ChatOpenAI
    
    if not settings.openrouter_api_key:
        raise ValueError("OpenRouter API key is required but not provided")
    
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        timeout=30.0,
        max_retries=2,
    )


# LLM factory mapping
_LLM_FACTORY: Dict[str, Any] = {
    "openai": _create_openai_llm,
    "google": _create_google_llm,
    "openrouter": _create_openrouter_llm,
}


def get_llm() -> BaseChatModel:
    """Get or create LLM instance (singleton pattern).
    
    Returns:
        BaseChatModel: Configured LLM instance
        
    Raises:
        ValueError: If provider is unsupported or API key is missing
    """
    global _llm_instance
    
    if _llm_instance is not None:
        return _llm_instance

    provider = settings.llm_provider.lower()
    
    if provider not in _LLM_FACTORY:
        available = ", ".join(_LLM_FACTORY.keys())
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Available providers: {available}"
        )

    try:
        factory = _LLM_FACTORY[provider]
        _llm_instance = factory()
        logger.info(
            f"Initialized LLM: provider={provider}, model={settings.llm_model}"
        )
        return _llm_instance
    
    except Exception as e:
        logger.error(f"Failed to initialize LLM provider '{provider}': {e}")
        raise


def reset_llm() -> None:
    """Reset the LLM instance (useful for testing)."""
    global _llm_instance
    _llm_instance = None


@cache
def get_supported_providers() -> list[str]:
    """Get list of supported LLM providers."""
    return list(_LLM_FACTORY.keys())