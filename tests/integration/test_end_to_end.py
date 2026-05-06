"""Integration tests for end-to-end workflow execution."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestEndToEndWorkflow:
    """Integration tests for the full query-to-response pipeline."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_finance_qa_flow(self, sample_state, mock_llm):
        """Test the complete flow from query to response for finance Q&A."""
        from src.workflow.nodes import initialize_nodes
        from src.agents.finance_qa_agent import FinanceQAAgent
        from src.workflow.router import QueryRouter

        # Setup
        agent = FinanceQAAgent(llm=mock_llm, retriever=None)
        router = QueryRouter(llm=mock_llm)
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="finance_qa")
        )

        initialize_nodes(
            agents={"finance_qa": agent},
            router=router,
        )

        from src.workflow.nodes import route_query_node, finance_qa_node

        # Execute routing
        routed_state = await route_query_node(sample_state)
        assert routed_state["route"] == "finance_qa"

        # Execute agent
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="Compound interest is interest on interest.")
        )
        result = await finance_qa_node(routed_state)
        assert result["response"]
        assert result["agent_name"] == "Finance Q&A Agent"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workflow_handles_missing_agent(self, sample_state):
        """Test graceful handling when an agent is not initialized."""
        from src.workflow.nodes import initialize_nodes, portfolio_analysis_node

        initialize_nodes(agents={}, router=None)
        result = await portfolio_analysis_node(sample_state)
        assert result.get("error")
