import logging

from langchain_core.language_models import BaseChatModel
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

_llm: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    global _llm
    if _llm is not None:
        return _llm

    settings = get_settings()
    provider = settings.llm_provider.lower()
    model = settings.llm_model

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            temperature=0.0,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.google_api_key,
            temperature=0.0,
        )
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            model=model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'openai', 'google', or 'openrouter'.")

    logger.info(f"Initialized LLM: provider={provider}, model={model}")
    return _llm
