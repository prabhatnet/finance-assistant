"""Pytest configuration and shared fixtures for the test suite."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.core.config import Settings, LLMSettings, EmbeddingSettings, VectorStoreSettings, RAGSettings, MarketDataSettings


@pytest.fixture
def mock_settings():
    """Create mock application settings for testing."""
    return Settings(
        app_name="AI Finance Assistant Test",
        app_version="0.1.0",
        app_env="testing",
        debug=True,
        log_level="DEBUG",
        openai_api_key="test-key",
        llm=LLMSettings(provider="openai", model="gpt-4o-mini", temperature=0.1),
        embeddings=EmbeddingSettings(model="all-MiniLM-L6-v2"),
        vector_store=VectorStoreSettings(
            type="faiss",
            persist_directory="./test_data/vector_store",
        ),
        rag=RAGSettings(top_k=3, knowledge_base_path="./test_data/knowledge_base"),
        market_data=MarketDataSettings(provider="yfinance"),
    )


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing agents without API calls."""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(content="This is a test response about finance.")
    )
    return llm


@pytest.fixture
def sample_state():
    """Create a sample AgentState for testing."""
    return {
        "query": "What is compound interest?",
        "chat_history": [],
        "route": "",
        "response": "",
        "agent_name": "",
        "sources": [],
        "portfolio_data": {},
        "market_data": {},
        "symbols": [],
        "news_articles": [],
        "user_profile": {},
        "error": None,
        "iteration_count": 0,
        "metadata": {},
    }


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    return {
        "holdings": [
            {"symbol": "AAPL", "shares": 50, "value": 8750.00},
            {"symbol": "MSFT", "shares": 30, "value": 12000.00},
            {"symbol": "GOOGL", "shares": 10, "value": 17500.00},
            {"symbol": "VOO", "shares": 100, "value": 45000.00},
        ],
        "total_value": 83250.00,
    }


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return {
        "AAPL": {
            "symbol": "AAPL",
            "price": 175.00,
            "change": 2.50,
            "change_percent": 1.45,
            "volume": 52000000,
            "name": "Apple Inc.",
        },
        "MSFT": {
            "symbol": "MSFT",
            "price": 400.00,
            "change": -1.20,
            "change_percent": -0.30,
            "volume": 25000000,
            "name": "Microsoft Corporation",
        },
    }
