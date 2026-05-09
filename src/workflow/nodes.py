"""Workflow nodes - Functions that execute within the LangGraph state machine."""

from __future__ import annotations

import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.prompts import SYNTHESIZER_SYSTEM_PROMPT
from src.core.state import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy-initialized shared resources (set during app startup)
_agents: dict = {}
_router = None
_llm: BaseChatModel | None = None


def initialize_nodes(agents: dict, router: object, llm: BaseChatModel | None = None) -> None:
    """Initialize workflow nodes with agent instances, router, and LLM.

    Called during application startup to inject dependencies.

    Args:
        agents: Dictionary mapping route names to agent instances.
        router: QueryRouter instance for routing decisions.
        llm: LLM instance used by the multi-agent synthesizer.
    """
    global _agents, _router, _llm
    _agents = agents
    _router = router
    _llm = llm
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


async def planner_node(state: AgentState) -> AgentState:
    """Planner node - Decomposes a complex multi-domain query into an agent plan.

    Args:
        state: Current workflow state with user query.

    Returns:
        Updated state with 'plan' and 'is_multi_agent' set.
    """
    agent = _agents.get("planner")
    if agent is None:
        logger.error("Planner agent not initialized")
        return {**state, "error": "Planner agent not available"}
    return await agent.process(state)


async def multi_agent_coordinator_node(state: AgentState) -> AgentState:
    """Coordinator node - Runs planned agents in parallel, then synthesizes results.

    Reads the 'plan' field set by the planner, dispatches each sub-query to its
    target agent concurrently, collects the outputs, and calls the LLM synthesizer
    to produce a single coherent response.

    Args:
        state: Current workflow state with 'plan' populated by planner.

    Returns:
        Updated state with synthesized 'response' and 'agent_outputs'.
    """
    plan: list[dict[str, str]] = state.get("plan", [])
    if not plan:
        logger.error("multi_agent_coordinator called with empty plan")
        return {**state, "error": "No execution plan found"}

    logger.info(
        "Coordinator dispatching %d agents: %s",
        len(plan),
        [step["agent"] for step in plan],
    )

    async def _run_agent(agent_name: str, sub_query: str) -> tuple[str, str]:
        """Run a single agent with its focused sub-query."""
        agent = _agents.get(agent_name)
        if agent is None:
            logger.warning("Coordinator: agent '%s' not found, skipping", agent_name)
            return agent_name, ""
        sub_state: AgentState = {**state, "query": sub_query}  # type: ignore[misc]
        try:
            result = await agent.process(sub_state)
            return agent_name, result.get("response", "")
        except Exception:
            logger.exception("Coordinator: agent '%s' raised an exception", agent_name)
            return agent_name, ""

    # Execute all planned agents concurrently
    tasks = [_run_agent(step["agent"], step["sub_query"]) for step in plan]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    agent_outputs: dict[str, str] = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error("Coordinator: a task raised an unhandled exception: %s", result)
        else:
            name, output = result
            if output:
                agent_outputs[name] = output

    synthesized = await _synthesize(state.get("query", ""), agent_outputs)

    return {
        **state,
        "response": synthesized,
        "agent_outputs": agent_outputs,
        "agent_name": "Multi-Agent Coordinator",
    }


async def _synthesize(original_query: str, agent_outputs: dict[str, str]) -> str:
    """Merge multiple agent outputs into a single coherent response via LLM."""
    if not agent_outputs:
        return (
            "I was unable to gather sufficient information to answer your question. "
            "Please try rephrasing or ask each part separately."
        )

    if len(agent_outputs) == 1:
        # Only one agent contributed — no synthesis needed
        return next(iter(agent_outputs.values()))

    sections = "\n\n".join(
        f"[{name.upper().replace('_', ' ')} ANALYSIS]\n{output}"
        for name, output in agent_outputs.items()
    )
    synthesis_prompt = (
        f"Original question from user:\n{original_query}\n\n"
        f"Analyses from specialized agents:\n{sections}"
    )

    if _llm is None:
        logger.warning("Synthesizer: no LLM available, concatenating outputs")
        return "\n\n---\n\n".join(
            f"**{name.replace('_', ' ').title()} Analysis:**\n{output}"
            for name, output in agent_outputs.items()
        )

    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=synthesis_prompt),
    ]
    try:
        response = await _llm.ainvoke(messages)
        return str(response.content)
    except Exception:
        logger.exception("Synthesizer LLM call failed — falling back to concatenation")
        return "\n\n---\n\n".join(
            f"**{name.replace('_', ' ').title()} Analysis:**\n{output}"
            for name, output in agent_outputs.items()
        )
