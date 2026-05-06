"""Input validators for the finance assistant."""

from __future__ import annotations

import re


def validate_stock_symbol(symbol: str) -> bool:
    """Validate a stock ticker symbol format.

    Args:
        symbol: Stock ticker to validate.

    Returns:
        True if the symbol appears valid.
    """
    if not symbol or not isinstance(symbol, str):
        return False
    # Valid US stock symbols: 1-5 uppercase letters
    return bool(re.match(r"^[A-Z]{1,5}$", symbol.upper()))


def validate_portfolio_input(holdings: list[dict]) -> tuple[bool, str]:
    """Validate portfolio holdings input.

    Args:
        holdings: List of holding dictionaries.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not holdings:
        return False, "Portfolio must contain at least one holding"

    for i, holding in enumerate(holdings):
        if "symbol" not in holding:
            return False, f"Holding {i+1} missing 'symbol' field"
        if not validate_stock_symbol(holding["symbol"]):
            return False, f"Invalid symbol: {holding.get('symbol')}"
        if holding.get("shares", 0) <= 0:
            return False, f"Holding {i+1} must have positive shares"

    return True, ""


def sanitize_user_input(text: str, max_length: int = 2000) -> str:
    """Sanitize user input text to prevent injection and limit length.

    Args:
        text: Raw user input.
        max_length: Maximum allowed length.

    Returns:
        Sanitized text string.
    """
    if not text:
        return ""
    # Truncate to max length
    text = text[:max_length]
    # Remove control characters except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()
