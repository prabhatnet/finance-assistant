"""Knowledge Base Indexer - Loads, chunks, and indexes financial education documents."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import Settings, get_settings
from src.rag.embeddings import create_embeddings
from src.rag.vector_store import VectorStoreManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeBaseIndexer:
    """Loads financial education documents, splits them into chunks,
    and indexes them into the vector store for RAG retrieval.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.kb_path = Path(self.settings.rag.knowledge_base_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.vector_store.chunk_size,
            chunk_overlap=self.settings.vector_store.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_documents(self) -> list[Document]:
        """Load all documents from the knowledge base directory.

        Supports .txt and .md files. Each file's metadata includes
        the source filename and category (derived from parent directory).

        Returns:
            List of loaded Document objects.
        """
        documents: list[Document] = []

        if not self.kb_path.exists():
            logger.warning("Knowledge base path does not exist: %s", self.kb_path)
            return documents

        for file_path in self.kb_path.rglob("*"):
            if file_path.suffix not in (".txt", ".md"):
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
                category = file_path.parent.name if file_path.parent != self.kb_path else "general"

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path.name,
                        "category": category,
                        "file_path": str(file_path),
                    },
                )
                documents.append(doc)
            except Exception:
                logger.exception("Failed to load document: %s", file_path)

        logger.info("Loaded %d documents from knowledge base", len(documents))
        return documents

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into smaller chunks for indexing.

        Args:
            documents: List of full documents.

        Returns:
            List of chunked Document objects with preserved metadata.
        """
        chunks = self.text_splitter.split_documents(documents)
        logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
        return chunks

    def index(self) -> VectorStoreManager:
        """Run the full indexing pipeline: load -> chunk -> embed -> store.

        Returns:
            Initialized VectorStoreManager with indexed documents.
        """
        logger.info("Starting knowledge base indexing pipeline")

        documents = self.load_documents()
        if not documents:
            logger.warning("No documents found to index")

        chunks = self.chunk_documents(documents) if documents else []

        embeddings = create_embeddings(self.settings)
        vector_store = VectorStoreManager(embeddings, self.settings)

        if chunks:
            vector_store.initialize(documents=chunks)
        else:
            vector_store.initialize()

        logger.info("Knowledge base indexing complete")
        return vector_store
