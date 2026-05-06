"""Chat page - Conversational interface for multi-agent interactions."""

import asyncio
from typing import Any

import streamlit as st

from src.utils.logger import get_logger

logger = get_logger(__name__)


def render_chat_page() -> None:
    """Render the conversational chat interface.

    Provides a chat-style interface where users can interact with
    the multi-agent system for financial education queries.
    """
    st.header("💬 Financial Assistant Chat")
    st.markdown(
        "Ask me anything about investing, markets, portfolio analysis, "
        "financial planning, tax concepts, or financial news."
    )

    # Display chat history
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.markdown(content)
            if message.get("agent_name"):
                st.caption(f"Handled by: {message['agent_name']}")

    # Chat input
    if user_input := st.chat_input("Ask a financial question..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process query through the workflow
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = _process_query(user_input)
                st.markdown(response.get("response", "I'm sorry, I couldn't process that query."))
                if agent_name := response.get("agent_name"):
                    st.caption(f"Handled by: {agent_name}")

        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response.get("response", ""),
            "agent_name": response.get("agent_name", ""),
        })


def _process_query(query: str) -> dict[str, Any]:
    """Process a user query through the LangGraph workflow.

    Args:
        query: The user's input query.

    Returns:
        Dictionary with response and metadata.
    """
    try:
        from src.workflow.graph import create_workflow_graph

        graph = create_workflow_graph()
        state = {
            "query": query,
            "chat_history": [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history[-10:]  # Keep last 10 messages for context
            ],
        }

        # Run the async workflow
        result = asyncio.run(graph.ainvoke(state))
        return result
    except Exception as e:
        logger.exception("Failed to process query")
        return {
            "response": (
                "I apologize, but I encountered an error processing your request. "
                "Please try again or rephrase your question."
            ),
            "error": str(e),
        }
