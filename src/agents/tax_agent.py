"""Tax Education Agent - Explains tax concepts and account types."""

from src.agents.base_agent import BaseAgent
from src.core.state import AgentState
from src.rag.retriever import RAGRetriever
from src.utils.logger import get_logger

logger = get_logger(__name__)

TAX_SYSTEM_PROMPT = """You are a tax education assistant specializing in investment-related tax concepts.
Your role is to explain tax rules, account types, and tax-efficient strategies
in clear language for beginner investors.

Guidelines:
- Explain tax concepts using simple language and concrete examples
- Cover capital gains (short-term vs long-term), dividends, and interest income
- Explain tax-advantaged accounts (Traditional IRA, Roth IRA, 401k, HSA, 529)
- Discuss tax-loss harvesting and wash sale rules at a high level
- Explain the difference between tax-deferred and tax-free growth
- Provide general tax-efficient investing strategies
- NEVER provide specific tax advice for individual situations
- ALWAYS recommend consulting a tax professional or CPA
- Note that tax laws change and information may not reflect current regulations

Tax Topics:
- Capital gains tax (short-term vs long-term rates)
- Dividend taxation (qualified vs ordinary)
- Tax-advantaged account types and contribution limits
- Required Minimum Distributions (RMDs)
- Tax-loss harvesting basics
- Estate and gift tax fundamentals
- Tax implications of different investment vehicles
"""


class TaxEducationAgent(BaseAgent):
    """Agent specialized in tax education related to investing.

    Provides clear explanations of tax concepts, account types,
    and tax-efficient strategies using RAG-backed knowledge.
    """

    def __init__(
        self,
        llm: object,
        retriever: RAGRetriever | None = None,
    ) -> None:
        super().__init__(
            name="Tax Education Agent",
            description="Explains tax concepts and account types",
            llm=llm,
            system_prompt=TAX_SYSTEM_PROMPT,
        )
        self.retriever = retriever

    async def process(self, state: AgentState) -> AgentState:
        """Process a tax education query.

        Args:
            state: Current workflow state with the user's tax query.

        Returns:
            Updated state with tax education response.
        """
        logger.info("Tax Education Agent processing query")
        query = state.get("query", "")
        chat_history = state.get("chat_history", [])

        context = ""
        sources = []
        if self.retriever:
            try:
                docs = await self.retriever.aretrieve(query, category="tax")
                context = "\n\n".join(doc.page_content for doc in docs)
                sources = [doc.metadata for doc in docs]
            except Exception:
                logger.warning("RAG retrieval failed for tax agent")

        response = await self._invoke_llm(
            user_message=query,
            context=context,
            chat_history=chat_history,
        )

        response += self._format_disclaimer()
        response += (
            "\n*Tax laws are complex and change frequently. "
            "Please consult a qualified tax professional for advice "
            "specific to your situation.*"
        )

        return {
            **state,
            "response": response,
            "agent_name": self.name,
            "sources": sources,
        }
