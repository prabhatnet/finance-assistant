"""State definitions for LangGraph workflow and agent communication."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state passed between agents in the LangGraph workflow.

    This TypedDict defines the schema for the state that flows through
    the multi-agent graph. Each agent reads from and writes to this state.
    """

    # --- User Input ---
    query: str                              # The current user query
    chat_history: list[dict[str, str]]      # Conversation history [{role, content}]

    # --- Routing ---
    agent_name: str                         # Name of the agent that handled the query
    route: str                              # Determined route (agent type to use)

    # --- Agent Response ---
    response: str                           # The generated response text
    sources: list[dict[str, Any]]           # RAG source attributions

    # --- Domain-Specific Data ---
    portfolio_data: dict[str, Any]          # User's portfolio holdings
    market_data: dict[str, Any]             # Fetched market quotes
    symbols: list[str]                      # Stock symbols extracted from query
    news_articles: list[dict[str, Any]]     # Fetched news articles
    user_profile: dict[str, Any]            # User's financial profile for goal planning

    # --- Workflow Control ---
    error: str | None                       # Error message if processing failed
    iteration_count: int                    # Current workflow iteration counter
    metadata: dict[str, Any]               # Additional metadata
