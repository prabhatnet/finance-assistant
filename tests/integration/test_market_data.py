"""Integration tests for market data providers."""

import pytest
from unittest.mock import patch, MagicMock

from src.data.market_data import MarketDataProvider
from src.data.cache import DataCache


class TestMarketDataIntegration:
    """Integration tests for market data fetching."""

    @pytest.fixture
    def provider(self, mock_settings):
        return MarketDataProvider(settings=mock_settings)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_quote_caches_result(self, provider):
        """Test that quotes are cached after first fetch."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_info = {"longName": "Apple Inc."}
            mock_fast_info = {
                "lastPrice": 175.50,
                "previousClose": 173.00,
                "lastVolume": 52000000,
                "marketCap": 2700000000000,
                "currency": "USD",
            }
            instance = MagicMock()
            instance.info = mock_info
            instance.fast_info = mock_fast_info
            mock_ticker.return_value = instance

            # First call - hits API
            result1 = await provider.get_quote("AAPL")
            assert result1["symbol"] == "AAPL"
            assert result1["price"] == 175.50

            # Second call - should hit cache
            result2 = await provider.get_quote("AAPL")
            assert result2["price"] == 175.50
            # yfinance.Ticker should only be called once
            assert mock_ticker.call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_quote_handles_failure(self, provider):
        """Test graceful handling of API failures."""
        with patch("yfinance.Ticker", side_effect=Exception("API Error")):
            result = await provider.get_quote("INVALID")
            assert result.get("error") == "Data unavailable"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_historical_returns_list(self, provider):
        """Test historical data retrieval."""
        import pandas as pd

        with patch("yfinance.Ticker") as mock_ticker:
            mock_hist = pd.DataFrame({
                "Open": [170.0, 172.0],
                "High": [175.0, 176.0],
                "Low": [169.0, 171.0],
                "Close": [173.0, 175.0],
                "Volume": [50000000, 48000000],
            }, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))

            instance = MagicMock()
            instance.history = MagicMock(return_value=mock_hist)
            mock_ticker.return_value = instance

            result = await provider.get_historical("AAPL", period="1mo")
            assert len(result) == 2
            assert result[0]["close"] == 173.0
