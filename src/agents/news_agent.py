"""News Synthesizer Agent - Summarizes and contextualizes financial news."""

from src.agents.base_agent import BaseAgent
from src.core.state import AgentState
from src.data.market_data import MarketDataProvider
from src.utils.logger import get_logger

logger = get_logger(__name__)

NEWS_SYSTEM_PROMPT = """You are a financial news synthesis assistant.
Your role is to summarize, contextualize, and explain financial news
in a way that is accessible to beginner investors.

Guidelines:
- Summarize news articles concisely, highlighting key takeaways
- Explain what the news means for different types of investors
- Provide historical context to help users understand significance
- Identify potential market impacts without making predictions
- Separate facts from opinions and speculation
- Note the source and timeliness of information
- NEVER advise users to trade based on news events
- Remind users that markets often react differently than expected

When synthesizing news:
1. Provide a brief headline summary
2. Explain the key facts and figures
3. Add context (historical comparisons, industry impact)
4. Discuss potential implications (without predictions)
5. Suggest what metrics to watch going forward
"""


class NewsSynthesizerAgent(BaseAgent):
    """Agent specialized in financial news synthesis and explanation.

    Processes financial news articles and presents them with
    educational context suitable for beginner investors.
    """

    # Default symbols to fetch news for when no specific symbols are mentioned
    DEFAULT_NEWS_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"]

    def __init__(self, llm: object, market_data_provider: MarketDataProvider | None = None) -> None:
        super().__init__(
            name="News Synthesizer Agent",
            description="Summarizes and contextualizes financial news",
            llm=llm,
            system_prompt=NEWS_SYSTEM_PROMPT,
        )
        self.market_data = market_data_provider

    async def process(self, state: AgentState) -> AgentState:
        """Process a news synthesis query.

        Args:
            state: Current workflow state with the user's news query.

        Returns:
            Updated state with synthesized news response.
        """
        logger.info("News Synthesizer Agent processing query")
        query = state.get("query", "")
        chat_history = state.get("chat_history", [])
        news_articles = state.get("news_articles", [])

        # Fetch real news if none pre-populated
        if not news_articles and self.market_data:
            symbols = state.get("symbols") or self.DEFAULT_NEWS_SYMBOLS
            for symbol in symbols[:3]:  # Limit to 3 symbols to avoid too many API calls
                try:
                    articles = await self.market_data.get_news(symbol)
                    news_articles.extend(articles)
                except Exception:
                    logger.warning("Failed to fetch news for %s", symbol)

        context = ""
        if news_articles:
            context = self._format_news_context(news_articles)

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

    def _format_news_context(self, articles: list[dict]) -> str:
        """Format news articles into context for the LLM.

        Args:
            articles: List of news article dictionaries.

        Returns:
            Formatted news context string.
        """
        lines = ["Recent Financial News:"]
        for i, article in enumerate(articles[:10], 1):
            title = article.get("title", "Untitled")
            source = article.get("source", "Unknown")
            summary = article.get("summary", "")
            published = article.get("published", "N/A")
            lines.append(f"\n  [{i}] {title}")
            lines.append(f"      Source: {source} | Published: {published}")
            if summary:
                lines.append(f"      Summary: {summary}")
        return "\n".join(lines)
