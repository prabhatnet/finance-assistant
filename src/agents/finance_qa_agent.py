"""Finance Q&A Agent - Handles general financial education queries."""

from src.agents.base_agent import BaseAgent
from src.core.state import AgentState
from src.rag.retriever import RAGRetriever
from src.utils.logger import get_logger

logger = get_logger(__name__)

FINANCE_QA_SYSTEM_PROMPT = """You are a knowledgeable and friendly financial education assistant.
Your role is to explain financial concepts in clear, jargon-free language suitable for beginners.

Guidelines:
- Break down complex concepts into simple, digestible explanations
- Use real-world analogies and examples to illustrate points
- Provide step-by-step explanations when appropriate
- Always mention relevant risks and considerations
- Encourage users to do further research and consult professionals
- NEVER provide specific investment advice or recommend specific securities
- Always include appropriate disclaimers

Topics you cover:
- Basic investing concepts (stocks, bonds, ETFs, mutual funds)
- Risk and return fundamentals
- Compound interest and time value of money
- Asset allocation and diversification
- Retirement accounts (401k, IRA, Roth IRA)
- Dollar-cost averaging
- Market basics and terminology
"""


class FinanceQAAgent(BaseAgent):
    """Agent specialized in answering general financial education questions.

    Uses RAG to retrieve relevant educational content from the knowledge base
    to provide accurate, well-sourced responses.
    """

    def __init__(
        self,
        llm: object,
        retriever: RAGRetriever | None = None,
    ) -> None:
        super().__init__(
            name="Finance Q&A Agent",
            description="Handles general financial education queries",
            llm=llm,
            system_prompt=FINANCE_QA_SYSTEM_PROMPT,
        )
        self.retriever = retriever

    async def process(self, state: AgentState) -> AgentState:
        """Process a financial education query.

        Args:
            state: Current workflow state containing the user's query.

        Returns:
            Updated state with the educational response.
        """
        logger.info("Finance Q&A Agent processing query")
        query = state.get("query", "")
        chat_history = state.get("chat_history", [])

        # Retrieve relevant context from knowledge base
        context = ""
        if self.retriever:
            try:
                docs = await self.retriever.aretrieve(query)
                context = "\n\n".join(doc.page_content for doc in docs)
            except Exception:
                logger.warning("RAG retrieval failed, proceeding without context")

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
            "sources": [doc.metadata for doc in docs] if self.retriever and context else [],
        }
