"""Portfolio page - Portfolio analysis dashboard with visualizations."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def render_portfolio_page() -> None:
    """Render the portfolio analysis dashboard.

    Provides tools for users to input their holdings and receive
    analysis, visualization, and diversification insights.
    """
    st.header("📊 Portfolio Analysis")
    st.markdown("Enter your holdings below to get diversification analysis and insights.")

    # Portfolio input section
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Add Holdings")
        with st.form("add_holding"):
            symbol = st.text_input("Stock Symbol", placeholder="e.g., AAPL").upper()
            shares = st.number_input("Number of Shares", min_value=0.0, step=1.0)
            cost_basis = st.number_input("Cost per Share ($)", min_value=0.0, step=0.01)
            submitted = st.form_submit_button("Add to Portfolio")

            if submitted and symbol and shares > 0:
                holding = {
                    "symbol": symbol,
                    "shares": shares,
                    "cost_basis": cost_basis,
                    "value": shares * cost_basis,
                }
                st.session_state.portfolio["holdings"].append(holding)
                st.session_state.portfolio["total_value"] = sum(
                    h["value"] for h in st.session_state.portfolio["holdings"]
                )
                st.success(f"Added {shares} shares of {symbol}")

    with col2:
        st.subheader("Current Holdings")
        holdings = st.session_state.portfolio["holdings"]
        if holdings:
            df = pd.DataFrame(holdings)
            st.dataframe(df, use_container_width=True)

            total = st.session_state.portfolio["total_value"]
            st.metric("Total Portfolio Value", f"${total:,.2f}")
        else:
            st.info("No holdings added yet. Use the form to add stocks.")

    # Visualization section
    if holdings:
        st.divider()
        st.subheader("Portfolio Visualization")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Allocation pie chart
            df = pd.DataFrame(holdings)
            fig = px.pie(
                df,
                values="value",
                names="symbol",
                title="Portfolio Allocation",
                hole=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)

        with viz_col2:
            # Holdings bar chart
            fig = px.bar(
                df,
                x="symbol",
                y="value",
                title="Holdings by Value",
                labels={"value": "Value ($)", "symbol": "Symbol"},
            )
            st.plotly_chart(fig, use_container_width=True)

        # Analysis button
        if st.button("🔍 Analyze My Portfolio", type="primary"):
            with st.spinner("Analyzing portfolio..."):
                _analyze_portfolio(holdings)


def _analyze_portfolio(holdings: list[dict]) -> None:
    """Run portfolio analysis through the Portfolio Analysis Agent.

    Args:
        holdings: List of holding dictionaries.
    """
    try:
        st.subheader("Analysis Results")
        total_value = sum(h["value"] for h in holdings)

        # Basic metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Total Holdings", len(holdings))
        with metrics_col2:
            st.metric("Total Value", f"${total_value:,.2f}")
        with metrics_col3:
            largest = max(holdings, key=lambda h: h["value"])
            concentration = (largest["value"] / total_value * 100) if total_value else 0
            st.metric("Largest Position", f"{largest['symbol']} ({concentration:.1f}%)")

        # Diversification warning
        if concentration > 30:
            st.warning(
                f"⚠️ High concentration: {largest['symbol']} represents "
                f"{concentration:.1f}% of your portfolio. Consider diversifying."
            )

        st.info(
            "💡 For a full AI-powered analysis, ensure your API keys are configured "
            "and the system will provide personalized insights."
        )
    except Exception:
        logger.exception("Portfolio analysis failed")
        st.error("Failed to analyze portfolio. Please try again.")
