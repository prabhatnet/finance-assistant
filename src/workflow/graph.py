"""LangGraph workflow graph definition - Orchestrates multi-agent interactions."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.core.state import AgentState
from src.workflow.nodes import (
    finance_qa_node,
    goal_planning_node,
    market_analysis_node,
    news_synthesis_node,
    portfolio_analysis_node,
    route_query_node,
    tax_education_node,
)
from src.workflow.router import route_to_agent
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_workflow_graph() -> StateGraph:
    """Create and compile the LangGraph workflow for the finance assistant.

    The graph follows this flow:
    1. User query enters the router node
    2. Router determines which specialized agent to invoke
    3. The appropriate agent processes the query
    4. Response is returned to the user

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    # Define the state graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", route_query_node)
    workflow.add_node("finance_qa", finance_qa_node)
    workflow.add_node("portfolio", portfolio_analysis_node)
    workflow.add_node("market", market_analysis_node)
    workflow.add_node("goal_planning", goal_planning_node)
    workflow.add_node("news", news_synthesis_node)
    workflow.add_node("tax", tax_education_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Add conditional edges from router to appropriate agent
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "finance_qa": "finance_qa",
            "portfolio": "portfolio",
            "market": "market",
            "goal_planning": "goal_planning",
            "news": "news",
            "tax": "tax",
        },
    )

    # All agents lead to END
    workflow.add_edge("finance_qa", END)
    workflow.add_edge("portfolio", END)
    workflow.add_edge("market", END)
    workflow.add_edge("goal_planning", END)
    workflow.add_edge("news", END)
    workflow.add_edge("tax", END)

    # Compile the graph
    compiled = workflow.compile()
    logger.info("Workflow graph compiled successfully")

    return compiled
