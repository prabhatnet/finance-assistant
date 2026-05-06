"""Portfolio Analysis Agent - Reviews and analyzes user investment portfolios."""

from src.agents.base_agent import BaseAgent
from src.core.state import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)

PORTFOLIO_SYSTEM_PROMPT = """You are an expert portfolio analysis assistant.
Your role is to help users understand their investment portfolio's composition,
risk profile, and performance characteristics.

Guidelines:
- Analyze portfolio diversification across asset classes, sectors, and geographies
- Calculate and explain key metrics (allocation percentages, concentration risk)
- Identify potential imbalances or overexposure
- Suggest general rebalancing strategies (not specific buy/sell recommendations)
- Compare portfolio characteristics against common benchmarks (e.g., S&P 500)
- Explain risk-return tradeoffs in the context of the user's holdings
- NEVER recommend specific securities to buy or sell
- Always note that past performance doesn't guarantee future results

When analyzing a portfolio:
1. Summarize the overall allocation
2. Identify the top holdings and sector concentrations
3. Assess diversification quality
4. Highlight potential risks
5. Provide educational context for any recommendations
"""


class PortfolioAnalysisAgent(BaseAgent):
    """Agent specialized in portfolio analysis and assessment.

    Analyzes user-provided portfolios for diversification, risk exposure,
    and general health, providing educational insights.
    """

    def __init__(self, llm: object) -> None:
        super().__init__(
            name="Portfolio Analysis Agent",
            description="Reviews and analyzes user portfolios",
            llm=llm,
            system_prompt=PORTFOLIO_SYSTEM_PROMPT,
        )

    async def process(self, state: AgentState) -> AgentState:
        """Analyze a user's portfolio.

        Args:
            state: Current workflow state with query and optional portfolio data.

        Returns:
            Updated state with portfolio analysis response.
        """
        logger.info("Portfolio Analysis Agent processing query")
        query = state.get("query", "")
        chat_history = state.get("chat_history", [])
        portfolio_data = state.get("portfolio_data", {})

        # Build context from portfolio data if available
        context = ""
        if portfolio_data:
            context = self._format_portfolio_context(portfolio_data)

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

    def _format_portfolio_context(self, portfolio_data: dict) -> str:
        """Format portfolio data into a context string for the LLM.

        Args:
            portfolio_data: Dictionary containing holdings and allocations.

        Returns:
            Formatted portfolio summary string.
        """
        lines = ["Current Portfolio Holdings:"]
        holdings = portfolio_data.get("holdings", [])
        for holding in holdings:
            symbol = holding.get("symbol", "N/A")
            shares = holding.get("shares", 0)
            value = holding.get("value", 0)
            lines.append(f"  - {symbol}: {shares} shares (${value:,.2f})")

        total = portfolio_data.get("total_value", 0)
        if total:
            lines.append(f"\nTotal Portfolio Value: ${total:,.2f}")

        return "\n".join(lines)
