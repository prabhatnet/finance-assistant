"""Main Streamlit application entry point for the AI Finance Assistant."""

import asyncio

import streamlit as st

from src.core.config import get_settings
from src.web_app.pages.chat import render_chat_page
from src.web_app.pages.portfolio import render_portfolio_page
from src.web_app.pages.market import render_market_page
from src.web_app.components.sidebar import render_sidebar
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main entry point for the Streamlit application."""
    settings = get_settings()

    # Page configuration
    st.set_page_config(
        page_title=settings.app_name,
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {"holdings": [], "total_value": 0}

    # Render sidebar and get navigation
    selected_page = render_sidebar()

    # Main content area
    st.title(f"💰 {settings.app_name}")
    st.caption("Your intelligent financial education companion")

    # Route to selected page
    if selected_page == "Chat":
        render_chat_page()
    elif selected_page == "Portfolio":
        render_portfolio_page()
    elif selected_page == "Market":
        render_market_page()


if __name__ == "__main__":
    main()
