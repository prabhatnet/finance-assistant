"""Unit tests for the LangGraph workflow and router."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.workflow.router import QueryRouter, route_to_agent, VALID_ROUTES


class TestQueryRouter:
    """Tests for the Query Router."""

    @pytest.fixture
    def router(self, mock_llm):
        return QueryRouter(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_route_finance_qa(self, mock_llm):
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="finance_qa"))
        router = QueryRouter(llm=mock_llm)
        route = await router.route("What is a stock?")
        assert route == "finance_qa"

    @pytest.mark.asyncio
    async def test_route_market(self, mock_llm):
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="market"))
        router = QueryRouter(llm=mock_llm)
        route = await router.route("What's the price of AAPL?")
        assert route == "market"

    @pytest.mark.asyncio
    async def test_route_invalid_defaults_to_finance_qa(self, mock_llm):
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="invalid_agent"))
        router = QueryRouter(llm=mock_llm)
        route = await router.route("random query")
        assert route == "finance_qa"

    @pytest.mark.asyncio
    async def test_route_handles_exception(self, mock_llm):
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        router = QueryRouter(llm=mock_llm)
        route = await router.route("test")
        assert route == "finance_qa"


class TestRouteToAgent:
    """Tests for the route_to_agent conditional edge function."""

    def test_valid_routes(self):
        for route in VALID_ROUTES:
            state = {"route": route}
            assert route_to_agent(state) == route

    def test_invalid_route_defaults(self):
        state = {"route": "nonexistent"}
        assert route_to_agent(state) == "finance_qa"

    def test_missing_route_defaults(self):
        state = {}
        assert route_to_agent(state) == "finance_qa"
