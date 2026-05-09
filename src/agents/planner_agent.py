"""Planner Agent - Decomposes complex multi-domain queries into agent sub-tasks."""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent
from src.core.prompts import PLANNER_SYSTEM_PROMPT
from src.core.state import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_PLAN_AGENTS = {"market", "tax", "goal_planning", "portfolio", "finance_qa", "news"}


class PlannerAgent(BaseAgent):
    """Decomposes complex multi-domain financial queries into a structured plan.

    When a user query requires expertise from multiple specialized agents
    (e.g. market volatility + tax implications + retirement planning),
    the PlannerAgent creates a list of focused sub-queries, one per agent,
    that the multi-agent coordinator will execute in parallel.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        super().__init__(
            name="Planner Agent",
            description="Decomposes complex multi-domain queries into agent sub-tasks",
            llm=llm,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )

    async def process(self, state: AgentState) -> AgentState:
        """Analyze the query and produce an execution plan for multiple agents.

        Args:
            state: Current workflow state containing the user's query.

        Returns:
            Updated state with 'plan' (list of agent sub-tasks) and 'is_multi_agent' set.
        """
        query = state.get("query", "")
        logger.info("Planner Agent decomposing query (%.80s...)", query)

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]

        plan: list[dict[str, str]] = []
        try:
            response = await self.llm.ainvoke(messages)
            raw = str(response.content).strip()

            # Strip markdown code fences if the LLM wraps output in them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("Planner output must be a JSON array")

            # Validate and filter entries
            for entry in parsed:
                agent = entry.get("agent", "")
                sub_query = entry.get("sub_query", "")
                if agent in VALID_PLAN_AGENTS and sub_query:
                    plan.append({"agent": agent, "sub_query": sub_query})

            if not plan:
                raise ValueError("Planner produced an empty or fully-invalid plan")

            logger.info(
                "Planner created %d-step plan: %s",
                len(plan),
                [p["agent"] for p in plan],
            )
        except Exception:
            logger.exception("Planning failed — falling back to single goal_planning step")
            plan = [{"agent": "goal_planning", "sub_query": query}]

        return {
            **state,
            "plan": plan,
            "is_multi_agent": True,
            "agent_name": self.name,
        }
