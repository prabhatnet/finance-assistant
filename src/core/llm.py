"""LLM factory - Creates and configures language model instances."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from src.core.config import Settings, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_llm(
    settings: Settings | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """Create a configured LLM instance based on application settings.

    Args:
        settings: Application settings. Uses global settings if not provided.
        temperature: Override temperature for specific agent needs.

    Returns:
        Configured BaseChatModel instance.

    Raises:
        ValueError: If the configured LLM provider is not supported.
    """
    if settings is None:
        settings = get_settings()

    provider = settings.llm.provider.lower()
    model = settings.llm.model
    temp = temperature if temperature is not None else settings.llm.temperature
    max_tokens = settings.llm.max_tokens

    logger.info("Creating LLM: provider=%s, model=%s", provider, model)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temp,
            max_tokens=max_tokens,
            api_key=settings.openai_api_key or None,
            request_timeout=settings.llm.request_timeout,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temp,
            max_output_tokens=max_tokens,
            google_api_key=settings.google_api_key or None,
        )

    if provider == "anthropic":
        from langchain_community.chat_models import ChatAnthropic

        return ChatAnthropic(
            model=model,
            temperature=temp,
            max_tokens=max_tokens,
            anthropic_api_key=settings.anthropic_api_key or None,
        )

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. "
        "Supported providers: openai, google, anthropic"
    )
