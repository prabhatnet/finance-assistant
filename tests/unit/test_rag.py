"""Unit tests for the RAG pipeline components."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.rag.retriever import RAGRetriever
from src.data.cache import DataCache


class TestRAGRetriever:
    """Tests for the RAG Retriever."""

    @pytest.fixture
    def mock_vector_store(self):
        store = MagicMock()
        store.is_initialized = True
        store.similarity_search = MagicMock(return_value=[
            MagicMock(
                page_content="Compound interest is earned on principal and interest.",
                metadata={"source": "compound_interest.md", "category": "investing_basics"},
            ),
        ])
        return store

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_settings):
        return RAGRetriever(vector_store=mock_vector_store, settings=mock_settings)

    def test_retrieve_returns_documents(self, retriever):
        docs = retriever.retrieve("What is compound interest?")
        assert len(docs) == 1
        assert "compound interest" in docs[0].page_content.lower()

    def test_retrieve_with_category_filter(self, retriever, mock_vector_store):
        retriever.retrieve("tax question", category="tax")
        mock_vector_store.similarity_search.assert_called_with(
            query="tax question", k=3, filter_dict={"category": "tax"}
        )

    def test_retrieve_empty_when_store_not_initialized(self, mock_settings):
        store = MagicMock()
        store.similarity_search = MagicMock(side_effect=RuntimeError("Not initialized"))
        retriever = RAGRetriever(vector_store=store, settings=mock_settings)
        docs = retriever.retrieve("test query")
        assert docs == []


class TestDataCache:
    """Tests for the TTL cache."""

    def test_set_and_get(self):
        cache = DataCache(ttl_seconds=60)
        cache.set("key1", {"price": 100})
        assert cache.get("key1") == {"price": 100}

    def test_get_missing_key(self):
        cache = DataCache(ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_expired_entry_returns_none(self):
        cache = DataCache(ttl_seconds=0)  # Immediate expiry
        cache.set("key1", "value")
        # Entry is immediately expired with ttl=0
        import time
        time.sleep(0.01)
        assert cache.get("key1") is None

    def test_invalidate(self):
        cache = DataCache(ttl_seconds=60)
        cache.set("key1", "value")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_clear(self):
        cache = DataCache(ttl_seconds=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.size == 0
