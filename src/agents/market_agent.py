"""Market Analysis Agent - Provides real-time market insights using live data."""

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent
from src.core.prompts import SYMBOL_EXTRACTION_PROMPT
from src.core.state import AgentState
from src.data.market_data import MarketDataProvider
from src.utils.logger import get_logger

logger = get_logger(__name__)

MARKET_SYSTEM_PROMPT = """You are a market analysis assistant providing real-time market insights.
Your role is to help users understand current market conditions, stock performance,
and market trends using live data.

Guidelines:
- Present market data clearly with key metrics highlighted
- Explain what the numbers mean in plain language
- Provide context for price movements (market trends, sector performance)
- Compare current data to historical benchmarks when relevant
- Identify notable trends or patterns without making predictions
- NEVER predict future stock prices or recommend buy/sell actions
- Always remind users that market data is informational, not advisory
- Note that data may be delayed and should be verified

When presenting market data:
1. Show current price, change, and percentage change
2. Include volume and market cap when relevant
3. Mention 52-week high/low for context
4. Provide sector and industry classification
5. Include relevant index performance for comparison
"""


class MarketAnalysisAgent(BaseAgent):
    """Agent specialized in real-time market data analysis.

    Fetches live market data via API integrations and presents
    it with educational context and analysis.
    """

    def __init__(
        self,
        llm: object,
        market_data_provider: MarketDataProvider | None = None,
    ) -> None:
        super().__init__(
            name="Market Analysis Agent",
            description="Provides real-time market insights",
            llm=llm,
            system_prompt=MARKET_SYSTEM_PROMPT,
        )
        self.market_data = market_data_provider

    async def process(self, state: AgentState) -> AgentState:
        """Process a market analysis query with live data.

        Args:
            state: Current workflow state with the user's market query.

        Returns:
            Updated state with market analysis response.
        """
        logger.info("Market Analysis Agent processing query")
        query = state.get("query", "")
        chat_history = state.get("chat_history", [])
        symbols = state.get("symbols", [])

        # Extract ticker symbols from the query if not already populated
        if not symbols:
            symbols = await self._extract_symbols(query)

        # Fetch real-time market data
        context = ""
        market_context = {}
        if self.market_data and symbols:
            try:
                for symbol in symbols:
                    data = await self.market_data.get_quote(symbol)
                    if data:
                        market_context[symbol] = data
                context = self._format_market_context(market_context)
            except Exception:
                logger.warning("Market data fetch failed, proceeding without live data")

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
            "market_data": market_context,
        }

    async def _extract_symbols(self, query: str) -> list[str]:
        """Extract stock ticker symbols from the query using the LLM."""
        try:
            messages = [
                HumanMessage(content=SYMBOL_EXTRACTION_PROMPT.format(query=query))
            ]
            response = await self.llm.ainvoke(messages)
            result = str(response.content).strip().upper()
            if result == "NONE" or not result:
                return []
            return [s.strip() for s in result.split(",") if s.strip()]
        except Exception:
            logger.warning("Symbol extraction failed for query: %s", query)
            return []

    def _format_market_context(self, market_data: dict) -> str:
        """Format market data into a context string for the LLM.

        Args:
            market_data: Dictionary mapping symbols to their quote data.

        Returns:
            Formatted market data string.
        """
        lines = ["Live Market Data:"]
        for symbol, data in market_data.items():
            price = data.get("price", "N/A")
            change = data.get("change", "N/A")
            change_pct = data.get("change_percent", "N/A")
            volume = data.get("volume", "N/A")
            lines.append(
                f"  {symbol}: ${price} | Change: {change} ({change_pct}%) | Vol: {volume}"
            )
        return "\n".join(lines)
