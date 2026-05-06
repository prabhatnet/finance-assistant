"""Workflow nodes - Functions that execute within the LangGraph state machine."""

from __future__ import annotations

from src.core.state import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy-initialized shared resources (set during app startup)
_agents: dict = {}
_router = None


def initialize_nodes(agents: dict, router: object) -> None:
    """Initialize workflow nodes with agent instances and router.

    Called during application startup to inject dependencies.

    Args:
        agents: Dictionary mapping route names to agent instances.
        router: QueryRouter instance for routing decisions.
    """
    global _agents, _router
    _agents = agents
    _router = router
    logger.info("Workflow nodes initialized with %d agents", len(agents))


async def route_query_node(state: AgentState) -> AgentState:
    """Router node - Classifies the query and sets the route in state.

    Args:
        state: Current workflow state with user query.

    Returns:
        Updated state with 'route' field set.
    """
    if _router is None:
        logger.error("Router not initialized")
        return {**state, "route": "finance_qa", "error": "Router not initialized"}

    query = state.get("query", "")
    route = await _router.route(query)
    return {**state, "route": route}


async def finance_qa_node(state: AgentState) -> AgentState:
    """Finance Q&A agent node.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with agent response.
    """
    agent = _agents.get("finance_qa")
    if agent is None:
        return {**state, "error": "Finance Q&A agent not available"}
    return await agent.process(state)


async def portfolio_analysis_node(state: AgentState) -> AgentState:
    """Portfolio Analysis agent node.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with agent response.
    """
    agent = _agents.get("portfolio")
    if agent is None:
        return {**state, "error": "Portfolio agent not available"}
    return await agent.process(state)


async def market_analysis_node(state: AgentState) -> AgentState:
    """Market Analysis agent node.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with agent response.
    """
    agent = _agents.get("market")
    if agent is None:
        return {**state, "error": "Market agent not available"}
    return await agent.process(state)


async def goal_planning_node(state: AgentState) -> AgentState:
    """Goal Planning agent node.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with agent response.
    """
    agent = _agents.get("goal_planning")
    if agent is None:
        return {**state, "error": "Goal Planning agent not available"}
    return await agent.process(state)


async def news_synthesis_node(state: AgentState) -> AgentState:
    """News Synthesizer agent node.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with agent response.
    """
    agent = _agents.get("news")
    if agent is None:
        return {**state, "error": "News agent not available"}
    return await agent.process(state)


async def tax_education_node(state: AgentState) -> AgentState:
    """Tax Education agent node.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with agent response.
    """
    agent = _agents.get("tax")
    if agent is None:
        return {**state, "error": "Tax agent not available"}
    return await agent.process(state)
