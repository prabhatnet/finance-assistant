"""RAG Retriever - Retrieves relevant documents from the vector store."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from src.rag.vector_store import VectorStoreManager
from src.core.config import Settings, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGRetriever:
    """Retrieves relevant documents from the knowledge base via vector similarity search.

    Supports category-based filtering and score thresholds for targeted retrieval.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        settings: Settings | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.settings = settings or get_settings()
        self.top_k = self.settings.rag.top_k
        self.score_threshold = self.settings.rag.score_threshold

    def retrieve(
        self,
        query: str,
        category: str | None = None,
        top_k: int | None = None,
    ) -> list[Document]:
        """Retrieve relevant documents for a given query.

        Args:
            query: The search query.
            category: Optional category filter (e.g., "tax", "investing_basics").
            top_k: Override for the number of results.

        Returns:
            List of relevant Document objects with metadata.
        """
        k = top_k or self.top_k
        filter_dict: dict[str, Any] | None = None

        if category:
            filter_dict = {"category": category}
            logger.info("Retrieving with category filter: %s", category)

        try:
            docs = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter_dict=filter_dict,
            )
            logger.info("Retrieved %d documents for query", len(docs))
            return docs
        except RuntimeError:
            logger.warning("Vector store not initialized, returning empty results")
            return []

    async def aretrieve(
        self,
        query: str,
        category: str | None = None,
        top_k: int | None = None,
    ) -> list[Document]:
        """Async wrapper for retrieve (vector search is CPU-bound, so runs synchronously).

        Args:
            query: The search query.
            category: Optional category filter.
            top_k: Override for the number of results.

        Returns:
            List of relevant Document objects.
        """
        return self.retrieve(query, category=category, top_k=top_k)
