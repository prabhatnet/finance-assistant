"""Query Router - Determines which agent should handle a user query."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.prompts import ROUTER_SYSTEM_PROMPT
from src.core.state import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_ROUTES = {"finance_qa", "portfolio", "market", "goal_planning", "news", "tax", "planner"}


class QueryRouter:
    """Routes user queries to the appropriate specialized agent using LLM classification.

    Uses the LLM to analyze the user's query and determine which
    agent is best suited to handle it.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    async def route(self, query: str) -> str:
        """Determine which agent should handle the query.

        Args:
            query: The user's input query.

        Returns:
            Agent route string (one of VALID_ROUTES).
        """
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            route = str(response.content).strip().lower()

            if route not in VALID_ROUTES:
                logger.warning(
                    "Router returned invalid route '%s', defaulting to finance_qa",
                    route,
                )
                return "finance_qa"

            logger.info("Query routed to: %s", route)
            return route
        except Exception:
            logger.exception("Routing failed, defaulting to finance_qa")
            return "finance_qa"


def route_to_agent(state: AgentState) -> str:
    """Conditional edge function for LangGraph routing.

    Reads the 'route' field from state and returns the next node name.

    Args:
        state: Current workflow state with route determined.

    Returns:
        Name of the next node to execute.
    """
    route = state.get("route", "finance_qa")
    if route not in VALID_ROUTES:
        return "finance_qa"
    return route
