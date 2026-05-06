"""Sidebar component - Navigation and settings."""

import streamlit as st


def render_sidebar() -> str:
    """Render the application sidebar with navigation and settings.

    Returns:
        The name of the selected page.
    """
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/money-bag.png", width=80)
        st.title("Navigation")

        selected_page = st.radio(
            "Go to",
            options=["Chat", "Portfolio", "Market"],
            label_visibility="collapsed",
        )

        st.divider()

        # Quick tips section
        st.subheader("💡 Quick Tips")
        st.markdown(
            """
            - **Chat**: Ask any financial question
            - **Portfolio**: Analyze your holdings
            - **Market**: View live market data
            """
        )

        st.divider()

        # System status
        st.subheader("⚙️ System Status")
        st.caption(f"Version: 0.1.0")
        st.caption(f"Agents: 6 active")

        # Disclaimer
        st.divider()
        st.caption(
            "⚠️ *This is an educational tool, not financial advice. "
            "Always consult a qualified financial advisor.*"
        )

    return selected_page
