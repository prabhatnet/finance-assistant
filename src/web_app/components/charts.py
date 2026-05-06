"""Chart components - Reusable Plotly chart builders for the web app."""

from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def create_allocation_pie(holdings: list[dict[str, Any]], title: str = "Allocation") -> go.Figure:
    """Create a donut chart showing portfolio allocation.

    Args:
        holdings: List of holding dicts with 'symbol' and 'value' keys.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    df = pd.DataFrame(holdings)
    fig = px.pie(
        df,
        values="value",
        names="symbol",
        title=title,
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def create_price_chart(
    history: list[dict[str, Any]],
    symbol: str,
    chart_type: str = "line",
) -> go.Figure:
    """Create a price history chart (line or candlestick).

    Args:
        history: List of daily price dicts with date, open, high, low, close.
        symbol: Stock ticker symbol for title.
        chart_type: "line" or "candlestick".

    Returns:
        Plotly Figure object.
    """
    df = pd.DataFrame(history)

    if chart_type == "candlestick":
        fig = go.Figure(data=[
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
            )
        ])
    else:
        fig = go.Figure(data=[
            go.Scatter(x=df["date"], y=df["close"], mode="lines", name="Close")
        ])

    fig.update_layout(
        title=f"{symbol} Price History",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
    )
    return fig


def create_comparison_bar(
    data: dict[str, float],
    title: str = "Comparison",
    y_label: str = "Value",
) -> go.Figure:
    """Create a bar chart for comparing values.

    Args:
        data: Dictionary mapping labels to values.
        title: Chart title.
        y_label: Y-axis label.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure(data=[
        go.Bar(
            x=list(data.keys()),
            y=list(data.values()),
            marker_color="steelblue",
        )
    ])
    fig.update_layout(title=title, yaxis_title=y_label)
    return fig
