"""Market page - Real-time market data dashboard."""

import asyncio

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def render_market_page() -> None:
    """Render the market overview dashboard with real-time data.

    Displays stock quotes, market indices, and interactive charts.
    """
    st.header("📈 Market Overview")
    st.markdown("View real-time market data and stock information.")

    settings = get_settings()

    # Symbol search
    col1, col2 = st.columns([3, 1])
    with col1:
        search_symbol = st.text_input(
            "Look up a stock",
            placeholder="Enter a ticker symbol (e.g., AAPL, MSFT, TSLA)",
        ).upper()
    with col2:
        st.write("")  # Spacing
        st.write("")
        search_clicked = st.button("🔍 Search", type="primary")

    # Display search results
    if search_clicked and search_symbol:
        _display_stock_info(search_symbol)

    # Default market overview
    st.divider()
    st.subheader("Market Indices & Popular Stocks")

    default_symbols = settings.market_data.default_symbols
    _display_market_overview(default_symbols)


def _display_stock_info(symbol: str) -> None:
    """Display detailed information for a single stock.

    Args:
        symbol: Stock ticker symbol.
    """
    try:
        from src.data.market_data import MarketDataProvider

        provider = MarketDataProvider()

        with st.spinner(f"Fetching data for {symbol}..."):
            quote = asyncio.run(provider.get_quote(symbol))
            info = asyncio.run(provider.get_company_info(symbol))
            history = asyncio.run(provider.get_historical(symbol, period="3mo"))

        if quote.get("error"):
            st.error(f"Could not fetch data for {symbol}. Please check the symbol and try again.")
            return

        # Display quote metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Price", f"${quote.get('price', 0):,.2f}")
        with col2:
            change = quote.get("change", 0)
            st.metric("Change", f"${change:,.2f}", delta=f"{quote.get('change_percent', 0)}%")
        with col3:
            st.metric("Volume", f"{quote.get('volume', 0):,}")
        with col4:
            st.metric("Market Cap", _format_market_cap(quote.get("market_cap", 0)))

        # Company info
        with st.expander("Company Information"):
            st.write(f"**{info.get('name', symbol)}**")
            st.write(f"Sector: {info.get('sector', 'N/A')}")
            st.write(f"Industry: {info.get('industry', 'N/A')}")
            st.write(f"52-Week High: ${info.get('52_week_high', 'N/A')}")
            st.write(f"52-Week Low: ${info.get('52_week_low', 'N/A')}")

        # Price chart
        if history:
            df = pd.DataFrame(history)
            fig = go.Figure(data=[
                go.Candlestick(
                    x=df["date"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name=symbol,
                )
            ])
            fig.update_layout(
                title=f"{symbol} - 3 Month Price History",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception:
        logger.exception("Failed to display stock info for %s", symbol)
        st.error(f"Error fetching data for {symbol}. Please try again later.")


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_quotes_cached(symbols: tuple[str, ...]) -> list[dict]:
    """Fetch quotes for multiple symbols, cached for 5 minutes."""
    from src.data.market_data import MarketDataProvider

    provider = MarketDataProvider()

    async def _fetch_all() -> list[dict]:
        import asyncio
        tasks = [provider.get_quote(s) for s in symbols]
        return await asyncio.gather(*tasks)

    return asyncio.run(_fetch_all())


def _display_market_overview(symbols: list[str]) -> None:
    """Display a quick overview of multiple stocks with live data.

    Args:
        symbols: List of stock ticker symbols.
    """
    with st.spinner("Fetching live market data..."):
        quotes = _fetch_quotes_cached(tuple(symbols))

    rows = []
    for symbol, q in zip(symbols, quotes):
        price = q.get("price", 0) or 0
        change = q.get("change", 0) or 0
        change_pct = q.get("change_percent", 0) or 0
        volume = q.get("volume", 0) or 0

        rows.append({
            "Symbol": symbol,
            "Price": f"${price:,.2f}" if price else "—",
            "Change": f"${change:+,.2f}" if price else "—",
            "Change %": f"{change_pct:+.2f}%" if price else "—",
            "Volume": f"{volume:,}" if volume else "—",
            "_change": change,  # for colour logic
        })

    df = pd.DataFrame(rows)

    # Colour the Change % column green/red
    def _colour_change(val: str) -> str:
        if val.startswith("+"):
            return "color: #22c55e"
        if val.startswith("-"):
            return "color: #ef4444"
        return ""

    display_df = df.drop(columns=["_change"])
    styled = display_df.style.applymap(_colour_change, subset=["Change", "Change %"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.caption("Data refreshes every 5 minutes. Prices from yFinance.")


def _format_market_cap(value: int) -> str:
    """Format market cap value into human-readable string.

    Args:
        value: Market cap in dollars.

    Returns:
        Formatted string (e.g., "$2.5T", "$150B").
    """
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.1f}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,}"
