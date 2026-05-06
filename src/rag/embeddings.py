"""Embedding model factory for vector representations of text."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from src.core.config import Settings, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_embeddings(settings: Settings | None = None) -> Embeddings:
    """Create an embedding model instance based on configuration.

    Supports HuggingFace sentence-transformers (local) and OpenAI embeddings.

    Args:
        settings: Application settings. Uses global settings if not provided.

    Returns:
        Configured Embeddings instance.
    """
    if settings is None:
        settings = get_settings()

    model_name = settings.embeddings.model
    logger.info("Creating embedding model: %s", model_name)

    if model_name.startswith("text-embedding"):
        # OpenAI embeddings
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model_name,
            api_key=settings.openai_api_key or None,
        )

    # Default: HuggingFace sentence-transformers (runs locally)
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
