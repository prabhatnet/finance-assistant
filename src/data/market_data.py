"""Market Data Provider - Fetches real-time stock data via yFinance and Alpha Vantage."""

from __future__ import annotations

from typing import Any

from src.core.config import Settings, get_settings
from src.data.cache import DataCache
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketDataProvider:
    """Fetches real-time market data from yFinance or Alpha Vantage APIs.

    Includes caching to respect rate limits and improve performance.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.provider = self.settings.market_data.provider
        self.cache = DataCache(ttl_seconds=self.settings.market_data.cache_ttl_seconds)

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get a real-time quote for a stock symbol.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL").

        Returns:
            Dictionary with price, change, volume, and other quote data.
        """
        cache_key = f"quote:{symbol.upper()}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for %s", symbol)
            return cached

        try:
            if self.provider == "yfinance":
                data = await self._fetch_yfinance(symbol)
            elif self.provider == "alpha_vantage":
                data = await self._fetch_alpha_vantage(symbol)
            else:
                raise ValueError(f"Unsupported market data provider: {self.provider}")

            self.cache.set(cache_key, data)
            return data
        except Exception:
            logger.exception("Failed to fetch quote for %s", symbol)
            return self._empty_quote(symbol)

    async def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
    ) -> list[dict[str, Any]]:
        """Get historical price data for a stock symbol.

        Args:
            symbol: Stock ticker symbol.
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, max).

        Returns:
            List of daily price dictionaries.
        """
        cache_key = f"hist:{symbol.upper()}:{period}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            records = []
            for date, row in hist.iterrows():
                records.append({
                    "date": str(date.date()),
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                })

            self.cache.set(cache_key, records)
            return records
        except Exception:
            logger.exception("Failed to fetch historical data for %s", symbol)
            return []

    async def get_company_info(self, symbol: str) -> dict[str, Any]:
        """Get company information for a stock symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Dictionary with company name, sector, industry, description, etc.
        """
        cache_key = f"info:{symbol.upper()}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            company_data = {
                "name": info.get("longName", symbol),
                "symbol": symbol.upper(),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "description": info.get("longBusinessSummary", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "dividend_yield": info.get("dividendYield", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            }

            self.cache.set(cache_key, company_data)
            return company_data
        except Exception:
            logger.exception("Failed to fetch company info for %s", symbol)
            return {"name": symbol, "symbol": symbol.upper()}

    async def get_news(self, symbol: str) -> list[dict[str, Any]]:
        """Get recent news for a stock symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            List of news article dictionaries.
        """
        cache_key = f"news:{symbol.upper()}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            news = ticker.news or []

            articles = []
            for item in news[:10]:
                articles.append({
                    "title": item.get("title", ""),
                    "source": item.get("publisher", "Unknown"),
                    "link": item.get("link", ""),
                    "published": item.get("providerPublishTime", ""),
                    "summary": item.get("title", ""),
                })

            self.cache.set(cache_key, articles)
            return articles
        except Exception:
            logger.exception("Failed to fetch news for %s", symbol)
            return []

    async def _fetch_yfinance(self, symbol: str) -> dict[str, Any]:
        """Fetch quote data using yFinance.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Quote data dictionary.
        """
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        info = ticker.info
        fast_info = ticker.fast_info

        return {
            "symbol": symbol.upper(),
            "price": round(float(fast_info.get("lastPrice", 0)), 2),
            "change": round(
                float(fast_info.get("lastPrice", 0)) - float(fast_info.get("previousClose", 0)),
                2,
            ),
            "change_percent": round(
                (
                    (float(fast_info.get("lastPrice", 0)) - float(fast_info.get("previousClose", 0)))
                    / float(fast_info.get("previousClose", 1))
                    * 100
                ),
                2,
            ),
            "volume": int(fast_info.get("lastVolume", 0)),
            "market_cap": int(fast_info.get("marketCap", 0)),
            "name": info.get("longName", symbol),
            "currency": fast_info.get("currency", "USD"),
        }

    async def _fetch_alpha_vantage(self, symbol: str) -> dict[str, Any]:
        """Fetch quote data using Alpha Vantage API.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Quote data dictionary.
        """
        import requests

        api_key = self.settings.market_data.alpha_vantage_api_key
        if not api_key:
            raise ValueError("Alpha Vantage API key not configured")

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": api_key,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        quote = data.get("Global Quote", {})
        return {
            "symbol": symbol.upper(),
            "price": round(float(quote.get("05. price", 0)), 2),
            "change": round(float(quote.get("09. change", 0)), 2),
            "change_percent": quote.get("10. change percent", "0%").rstrip("%"),
            "volume": int(quote.get("06. volume", 0)),
            "name": symbol.upper(),
        }

    def _empty_quote(self, symbol: str) -> dict[str, Any]:
        """Return an empty quote structure for error fallback.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Empty quote dictionary.
        """
        return {
            "symbol": symbol.upper(),
            "price": 0,
            "change": 0,
            "change_percent": 0,
            "volume": 0,
            "name": symbol.upper(),
            "error": "Data unavailable",
        }
