"""Base agent class providing shared functionality for all specialized agents."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.core.state import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all specialized financial agents.

    Provides common LLM interaction patterns, RAG retrieval,
    error handling, and response formatting.
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm: BaseChatModel,
        system_prompt: str,
    ) -> None:
        self.name = name
        self.description = description
        self.llm = llm
        self.system_prompt = system_prompt

    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        """Process the current state and return an updated state.

        Args:
            state: Current conversation/workflow state.

        Returns:
            Updated state with agent's response.
        """
        ...

    async def _invoke_llm(
        self,
        user_message: str,
        context: str = "",
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Invoke the LLM with the agent's system prompt, optional context, and chat history.

        Args:
            user_message: The user's query.
            context: Optional RAG-retrieved context.
            chat_history: Optional prior conversation messages.

        Returns:
            The LLM's response text.
        """
        messages: list[SystemMessage | HumanMessage | AIMessage] = [
            SystemMessage(content=self._build_system_prompt(context))
        ]

        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=user_message))

        try:
            response = await self.llm.ainvoke(messages)
            return str(response.content)
        except Exception:
            logger.exception("LLM invocation failed for agent %s", self.name)
            raise

    def _build_system_prompt(self, context: str = "") -> str:
        """Build the full system prompt with optional RAG context.

        Args:
            context: Retrieved context from the knowledge base.

        Returns:
            Formatted system prompt string.
        """
        prompt = self.system_prompt
        if context:
            prompt += (
                "\n\n--- Retrieved Knowledge Base Context ---\n"
                f"{context}\n"
                "--- End Context ---\n\n"
                "Use the above context to ground your response. "
                "Cite sources when available."
            )
        return prompt

    def _format_disclaimer(self) -> str:
        """Return a standard financial disclaimer."""
        return (
            "\n\n---\n"
            "*Disclaimer: This information is for educational purposes only "
            "and does not constitute financial advice. Please consult a "
            "qualified financial advisor before making investment decisions.*"
        )
