"""Goal Planning Agent - Assists with financial goal setting and planning."""

from src.agents.base_agent import BaseAgent
from src.core.state import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)

GOAL_PLANNING_SYSTEM_PROMPT = """You are a financial goal planning assistant.
Your role is to help users set realistic financial goals and create actionable plans
to achieve them, considering their risk tolerance and time horizon.

Guidelines:
- Help users define SMART financial goals (Specific, Measurable, Achievable, Relevant, Time-bound)
- Calculate required savings rates and investment returns for goals
- Explain the relationship between time horizon, risk tolerance, and asset allocation
- Provide general asset allocation frameworks based on goals and timelines
- Use the rule of 72 and compound interest to illustrate growth potential
- Consider inflation in long-term projections
- Discuss emergency funds, debt management, and savings priorities
- NEVER guarantee specific returns or outcomes
- Always recommend consulting a financial advisor for personalized plans

Goal Categories:
- Retirement planning (401k, IRA optimization)
- Emergency fund building
- Home purchase savings
- Education funding (529 plans)
- General wealth building
- Debt payoff strategies
"""


class GoalPlanningAgent(BaseAgent):
    """Agent specialized in financial goal setting and planning.

    Helps users create structured financial plans with realistic
    timelines and savings strategies.
    """

    def __init__(self, llm: object) -> None:
        super().__init__(
            name="Goal Planning Agent",
            description="Assists with financial goal setting and planning",
            llm=llm,
            system_prompt=GOAL_PLANNING_SYSTEM_PROMPT,
        )

    async def process(self, state: AgentState) -> AgentState:
        """Process a goal planning query.

        Args:
            state: Current workflow state with the user's goal planning query.

        Returns:
            Updated state with goal planning response.
        """
        logger.info("Goal Planning Agent processing query")
        query = state.get("query", "")
        chat_history = state.get("chat_history", [])
        user_profile = state.get("user_profile", {})

        context = ""
        if user_profile:
            context = self._format_user_profile(user_profile)

        response = await self._invoke_llm(
            user_message=query,
            context=context,
            chat_history=chat_history,
        )

        response += self._format_disclaimer()

        return {
            **state,
            "response": response,
            "agent_name": self.name,
        }

    def _format_user_profile(self, profile: dict) -> str:
        """Format user profile data into context for the LLM.

        Args:
            profile: User's financial profile information.

        Returns:
            Formatted profile string.
        """
        lines = ["User Financial Profile:"]
        if age := profile.get("age"):
            lines.append(f"  Age: {age}")
        if income := profile.get("annual_income"):
            lines.append(f"  Annual Income: ${income:,.0f}")
        if risk := profile.get("risk_tolerance"):
            lines.append(f"  Risk Tolerance: {risk}")
        if horizon := profile.get("time_horizon_years"):
            lines.append(f"  Investment Time Horizon: {horizon} years")
        if savings := profile.get("current_savings"):
            lines.append(f"  Current Savings: ${savings:,.0f}")
        if goals := profile.get("goals"):
            lines.append(f"  Financial Goals: {', '.join(goals)}")
        return "\n".join(lines)
