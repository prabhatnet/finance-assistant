"""Unit tests for specialized financial agents."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.finance_qa_agent import FinanceQAAgent
from src.agents.portfolio_agent import PortfolioAnalysisAgent
from src.agents.market_agent import MarketAnalysisAgent
from src.agents.goal_planning_agent import GoalPlanningAgent
from src.agents.news_agent import NewsSynthesizerAgent
from src.agents.tax_agent import TaxEducationAgent


class TestFinanceQAAgent:
    """Tests for the Finance Q&A Agent."""

    @pytest.fixture
    def agent(self, mock_llm):
        return FinanceQAAgent(llm=mock_llm, retriever=None)

    @pytest.mark.asyncio
    async def test_process_returns_response(self, agent, sample_state):
        result = await agent.process(sample_state)
        assert "response" in result
        assert result["agent_name"] == "Finance Q&A Agent"

    @pytest.mark.asyncio
    async def test_process_includes_disclaimer(self, agent, sample_state):
        result = await agent.process(sample_state)
        assert "educational purposes" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_process_with_rag_context(self, mock_llm):
        mock_retriever = MagicMock()
        mock_retriever.aretrieve = AsyncMock(return_value=[
            MagicMock(page_content="Test content", metadata={"source": "test.md"})
        ])
        agent = FinanceQAAgent(llm=mock_llm, retriever=mock_retriever)
        state = {"query": "What is a stock?", "chat_history": []}
        result = await agent.process(state)
        assert result["sources"]


class TestPortfolioAgent:
    """Tests for the Portfolio Analysis Agent."""

    @pytest.fixture
    def agent(self, mock_llm):
        return PortfolioAnalysisAgent(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_process_with_portfolio(self, agent, sample_state, sample_portfolio):
        state = {**sample_state, "portfolio_data": sample_portfolio}
        result = await agent.process(state)
        assert result["agent_name"] == "Portfolio Analysis Agent"

    @pytest.mark.asyncio
    async def test_process_without_portfolio(self, agent, sample_state):
        result = await agent.process(sample_state)
        assert "response" in result


class TestMarketAnalysisAgent:
    """Tests for the Market Analysis Agent."""

    @pytest.fixture
    def agent(self, mock_llm):
        return MarketAnalysisAgent(llm=mock_llm, market_data_provider=None)

    @pytest.mark.asyncio
    async def test_process_without_market_data(self, agent, sample_state):
        result = await agent.process(sample_state)
        assert result["agent_name"] == "Market Analysis Agent"


class TestGoalPlanningAgent:
    """Tests for the Goal Planning Agent."""

    @pytest.fixture
    def agent(self, mock_llm):
        return GoalPlanningAgent(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_process_with_profile(self, agent, sample_state):
        state = {
            **sample_state,
            "query": "How should I save for retirement?",
            "user_profile": {
                "age": 30,
                "annual_income": 75000,
                "risk_tolerance": "moderate",
                "time_horizon_years": 30,
            },
        }
        result = await agent.process(state)
        assert result["agent_name"] == "Goal Planning Agent"


class TestNewsSynthesizerAgent:
    """Tests for the News Synthesizer Agent."""

    @pytest.fixture
    def agent(self, mock_llm):
        return NewsSynthesizerAgent(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_process_with_articles(self, agent, sample_state):
        state = {
            **sample_state,
            "news_articles": [
                {"title": "Fed raises rates", "source": "Reuters", "summary": "..."}
            ],
        }
        result = await agent.process(state)
        assert result["agent_name"] == "News Synthesizer Agent"


class TestTaxEducationAgent:
    """Tests for the Tax Education Agent."""

    @pytest.fixture
    def agent(self, mock_llm):
        return TaxEducationAgent(llm=mock_llm, retriever=None)

    @pytest.mark.asyncio
    async def test_process_tax_query(self, agent, sample_state):
        state = {**sample_state, "query": "How do Roth IRAs work?"}
        result = await agent.process(state)
        assert result["agent_name"] == "Tax Education Agent"
        assert "tax professional" in result["response"].lower()
