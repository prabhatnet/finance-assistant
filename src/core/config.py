"""Application configuration management using Pydantic Settings and YAML.

Secret / API key loading strategy
----------------------------------
Locally          : pydantic-settings reads from the .env file at the project root.
HuggingFace Spaces: HF injects every secret as a plain OS environment variable;
                   no .env file is present, so pydantic-settings falls back to
                   reading those env vars directly.  No code change is needed.

Priority order (highest → lowest):
  1. Real OS environment variables  (HF Spaces secrets land here)
  2. .env file on disk              (local development)
  3. Pydantic field defaults
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

# Absolute path to the project root (two levels up from this file: src/core/config.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Detect HuggingFace Spaces: HF sets the SPACE_ID env var on every Space runtime
_ON_HF_SPACES: bool = "SPACE_ID" in os.environ


def _load_yaml_config() -> dict[str, Any]:
    """Load configuration from config.yaml file."""
    config_path = _PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml_config = _load_yaml_config()


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    provider: str = Field(default="openai", description="LLM provider name")
    model: str = Field(default="gpt-4o-mini", description="Model identifier")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    request_timeout: int = Field(default=30, gt=0)


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model: str = Field(default="all-MiniLM-L6-v2")
    dimension: int = Field(default=384)


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""

    type: str = Field(default="faiss")
    persist_directory: str = Field(default=str(_PROJECT_ROOT / "data" / "vector_store"))
    collection_name: str = Field(default="finance_knowledge")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)


class RAGSettings(BaseSettings):
    """RAG pipeline configuration."""

    top_k: int = Field(default=5)
    score_threshold: float = Field(default=0.7)
    knowledge_base_path: str = Field(default=str(_PROJECT_ROOT / "src" / "data" / "knowledge_base"))


class MarketDataSettings(BaseSettings):
    """Market data API configuration."""

    provider: str = Field(default="yfinance")
    cache_ttl_seconds: int = Field(default=300)
    alpha_vantage_api_key: str = Field(default="")
    default_symbols: list[str] = Field(
        default=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "SPY", "QQQ", "VOO"]
    )


class Settings(BaseSettings):
    """Root application settings, composing all sub-configs."""

    app_name: str = Field(default="AI Finance Assistant")
    app_version: str = Field(default="0.1.0")
    app_env: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # API keys (loaded from environment)
    openai_api_key: str = Field(default="")
    google_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")

    # Sub-configurations
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    market_data: MarketDataSettings = Field(default_factory=MarketDataSettings)

    model_config = {
        # Absolute path so the file is found regardless of working directory.
        # On HuggingFace Spaces this file won't exist; pydantic-settings silently
        # skips it and reads secrets directly from OS environment variables instead.
        "env_file": str(_PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @classmethod
    def from_yaml(cls) -> Settings:
        """Create Settings instance merging YAML config with env vars."""
        yaml_cfg = _yaml_config

        # Map YAML values to flat settings
        llm_cfg = yaml_cfg.get("llm", {})
        embed_cfg = yaml_cfg.get("embeddings", {})
        vs_cfg = yaml_cfg.get("vector_store", {})
        rag_cfg = yaml_cfg.get("rag", {})
        mkt_cfg = yaml_cfg.get("market_data", {})

        return cls(
            app_name=yaml_cfg.get("app", {}).get("name", "AI Finance Assistant"),
            app_version=yaml_cfg.get("app", {}).get("version", "0.1.0"),
            debug=yaml_cfg.get("app", {}).get("debug", False),
            log_level=yaml_cfg.get("logging", {}).get("level", "INFO"),
            llm=LLMSettings(**llm_cfg) if llm_cfg else LLMSettings(),
            embeddings=EmbeddingSettings(**embed_cfg) if embed_cfg else EmbeddingSettings(),
            vector_store=VectorStoreSettings(**vs_cfg) if vs_cfg else VectorStoreSettings(),
            rag=RAGSettings(**rag_cfg) if rag_cfg else RAGSettings(),
            market_data=MarketDataSettings(**mkt_cfg) if mkt_cfg else MarketDataSettings(),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings.from_yaml()
