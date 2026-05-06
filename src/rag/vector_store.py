"""Vector store management - FAISS and ChromaDB initialization and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.core.config import Settings, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Manages the lifecycle of the vector store (create, load, persist, search).

    Supports FAISS (default) and ChromaDB backends.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embeddings = embeddings
        self.store_type = self.settings.vector_store.type
        self.persist_dir = Path(self.settings.vector_store.persist_directory)
        self._store: Any = None

    def initialize(self, documents: list[Document] | None = None) -> None:
        """Initialize or load the vector store.

        If documents are provided, creates a new store. Otherwise, loads
        from the persist directory if it exists.

        Args:
            documents: Optional list of documents to index.
        """
        if documents:
            self._create_from_documents(documents)
        else:
            self._load_existing()

    def _create_from_documents(self, documents: list[Document]) -> None:
        """Create a new vector store from documents.

        Args:
            documents: Documents to index.
        """
        logger.info(
            "Creating %s vector store with %d documents",
            self.store_type,
            len(documents),
        )

        if self.store_type == "faiss":
            from langchain_community.vectorstores import FAISS

            self._store = FAISS.from_documents(documents, self.embeddings)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(self.persist_dir))

        elif self.store_type == "chroma":
            from langchain_community.vectorstores import Chroma

            self._store = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=str(self.persist_dir),
                collection_name=self.settings.vector_store.collection_name,
            )

        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")

        logger.info("Vector store created and persisted to %s", self.persist_dir)

    def _load_existing(self) -> None:
        """Load an existing vector store from disk."""
        if not self.persist_dir.exists():
            logger.warning("No existing vector store found at %s", self.persist_dir)
            return

        logger.info("Loading %s vector store from %s", self.store_type, self.persist_dir)

        if self.store_type == "faiss":
            from langchain_community.vectorstores import FAISS

            self._store = FAISS.load_local(
                str(self.persist_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        elif self.store_type == "chroma":
            from langchain_community.vectorstores import Chroma

            self._store = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings,
                collection_name=self.settings.vector_store.collection_name,
            )

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Search the vector store for similar documents.

        Args:
            query: Search query string.
            k: Number of results to return.
            filter_dict: Optional metadata filter.

        Returns:
            List of matching Document objects.

        Raises:
            RuntimeError: If the vector store is not initialized.
        """
        if self._store is None:
            raise RuntimeError(
                "Vector store not initialized. Call initialize() first."
            )

        if filter_dict and self.store_type == "chroma":
            return self._store.similarity_search(query, k=k, filter=filter_dict)

        return self._store.similarity_search(query, k=k)

    @property
    def is_initialized(self) -> bool:
        """Check if the vector store is loaded and ready."""
        return self._store is not None
