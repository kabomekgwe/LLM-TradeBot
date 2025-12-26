"""Provider tests with parametrized testing across all exchanges.

These tests verify that all exchange providers implement the BaseExchangeProvider
interface correctly and return data in the unified format.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integrations.trading.config import TradingConfig
from integrations.trading.providers.factory import create_provider
from integrations.trading.models.market_data import OHLCV, Ticker, OrderBook, Balance
from integrations.trading.models.positions import Position, Order


# Parametrized provider list - all supported providers
PROVIDERS_TO_TEST = [
    "binance_futures",
    "binance_spot",
    "kraken",
    "coinbase",
    "alpaca",
    "paper",
]


class TestProviderInterface:
    """Test that all providers implement the interface correctly."""

    @pytest.mark.parametrize("provider_name", PROVIDERS_TO_TEST)
    def test_provider_creation(self, provider_name):
        """Test that provider can be created from config."""
        config = TradingConfig(
            provider=provider_name,
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
        )

        provider = create_provider(config)

        assert provider is not None
        assert provider.get_provider_name() == provider_name

    @pytest.mark.parametrize("provider_name", PROVIDERS_TO_TEST)
    @pytest.mark.asyncio
    async def test_fetch_ohlcv_returns_unified_format(self, provider_name):
        """Test that fetch_ohlcv returns list of OHLCV objects."""
        # Skip actual API calls in tests - just verify structure
        # In real tests, would use mock or testnet
        pytest.skip("Requires live API connection or mocking")

    @pytest.mark.parametrize("provider_name", PROVIDERS_TO_TEST)
    @pytest.mark.asyncio
    async def test_fetch_ticker_returns_unified_format(self, provider_name):
        """Test that fetch_ticker returns Ticker object."""
        pytest.skip("Requires live API connection or mocking")


class TestDataModels:
    """Test unified data models."""

    def test_ohlcv_validation(self):
        """Test OHLCV validation catches invalid data."""
        # Valid OHLCV
        valid = OHLCV(
            timestamp=1640000000000,
            open=42000.0,
            high=42500.0,
            low=41800.0,
            close=42300.0,
            volume=100.5,
        )
        assert valid.close == 42300.0
        assert valid.is_bullish

        # Invalid OHLCV - high < low should raise
        with pytest.raises(ValueError, match="high.*< low"):
            OHLCV(
                timestamp=1640000000000,
                open=42000.0,
                high=41000.0,  # Lower than low!
                low=42000.0,
                close=42000.0,
                volume=100.0,
            )

    def test_ohlcv_from_ccxt(self):
        """Test OHLCV.from_ccxt conversion."""
        ccxt_data = [1640000000000, 42000.0, 42500.0, 41800.0, 42300.0, 100.5]

        candle = OHLCV.from_ccxt(ccxt_data)

        assert candle.timestamp == 1640000000000
        assert candle.open == 42000.0
        assert candle.high == 42500.0
        assert candle.low == 41800.0
        assert candle.close == 42300.0
        assert candle.volume == 100.5
        assert candle.is_bullish

    def test_ticker_properties(self):
        """Test Ticker calculated properties."""
        ticker = Ticker(
            symbol="BTC/USDT",
            bid=41999.5,
            ask=42000.5,
            last=42000.0,
            volume=1000.0,
            timestamp=1640000000000,
        )

        assert ticker.spread == 1.0  # ask - bid
        assert ticker.mid_price == 42000.0  # (bid + ask) / 2
        assert ticker.spread_pct < 0.01  # Very tight spread

    def test_orderbook_properties(self):
        """Test OrderBook calculated properties."""
        orderbook = OrderBook(
            symbol="BTC/USDT",
            bids=[(41999.0, 1.0), (41998.0, 2.0)],
            asks=[(42001.0, 1.5), (42002.0, 2.5)],
            timestamp=1640000000000,
        )

        assert orderbook.best_bid == (41999.0, 1.0)
        assert orderbook.best_ask == (42001.0, 1.5)
        assert orderbook.spread == 2.0  # 42001 - 41999
        assert orderbook.mid_price == 42000.0

    def test_position_pnl_calculation(self):
        """Test Position PnL calculations."""
        from integrations.trading.models.positions import PositionSide

        # Long position - price went up
        long_pos = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            size=1.0,
            entry_price=40000.0,
            current_price=42000.0,
            unrealized_pnl=2000.0,
        )

        assert long_pos.is_long
        assert long_pos.is_profitable
        assert long_pos.pnl_pct == 5.0  # (42000 - 40000) / 40000 * 100

        # Short position - price went down
        short_pos = Position(
            symbol="BTC/USDT",
            side=PositionSide.SHORT,
            size=1.0,
            entry_price=42000.0,
            current_price=40000.0,
            unrealized_pnl=2000.0,
        )

        assert short_pos.is_short
        assert short_pos.is_profitable
        assert short_pos.pnl_pct == pytest.approx(4.76, rel=0.01)  # (42000-40000)/42000*100

    def test_order_fill_percentage(self):
        """Test Order fill tracking."""
        from integrations.trading.models.positions import OrderSide, OrderType, OrderStatus

        order = Order(
            id="123",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=10.0,
            price=42000.0,
            status=OrderStatus.PARTIALLY_FILLED,
            filled=6.0,
        )

        assert order.fill_percentage == 60.0
        assert order.remaining == 4.0
        assert not order.is_complete


class TestProviderNormalizer:
    """Test symbol and format normalization."""

    def test_symbol_normalization(self):
        """Test symbol format conversion between exchanges."""
        from integrations.trading.providers.normalizer import ExchangeNormalizer

        # Binance format to Coinbase format
        result = ExchangeNormalizer.normalize_symbol(
            "BTC/USDT", "binance_futures", "coinbase"
        )
        assert result == "BTC-USD" or result == "BTC-USDT"  # Depends on quote handling

        # Binance format to Alpaca (stocks, no quote)
        result = ExchangeNormalizer.normalize_symbol(
            "AAPL/USD", "binance_spot", "alpaca"
        )
        assert result == "AAPL"

    def test_timeframe_normalization(self):
        """Test timeframe format conversion."""
        from integrations.trading.providers.normalizer import ExchangeNormalizer

        # Standard to Alpaca format
        result = ExchangeNormalizer.normalize_timeframe("5m", "alpaca")
        assert result == "5Min"

        # Standard to Binance (no change needed)
        result = ExchangeNormalizer.normalize_timeframe("5m", "binance_futures")
        assert result == "5m"


class TestRegimeDetection:
    """Test market regime detection."""

    def test_trending_regime_detection(self):
        """Test detection of trending market."""
        from integrations.trading.models.regime import RegimeDetector, MarketRegime

        # Create trending candles (consistent upward movement)
        candles = [
            OHLCV(
                timestamp=1640000000000 + i * 60000,
                open=40000.0 + i * 100,
                high=40100.0 + i * 100,
                low=39900.0 + i * 100,
                close=40050.0 + i * 100,
                volume=100.0,
            )
            for i in range(20)
        ]

        regime = RegimeDetector.detect_from_candles(candles, lookback=20)

        assert regime in (MarketRegime.TRENDING, MarketRegime.VOLATILE)

    def test_choppy_regime_detection(self):
        """Test detection of choppy/sideways market."""
        from integrations.trading.models.regime import RegimeDetector, MarketRegime

        # Create choppy candles (alternating up/down)
        candles = []
        for i in range(20):
            price = 40000.0 + (100.0 if i % 2 == 0 else -100.0)
            candles.append(OHLCV(
                timestamp=1640000000000 + i * 60000,
                open=price,
                high=price + 50,
                low=price - 50,
                close=price + 25,
                volume=100.0,
            ))

        regime = RegimeDetector.detect_from_candles(candles, lookback=20)

        assert regime in (MarketRegime.CHOPPY, MarketRegime.NEUTRAL)

    def test_regime_recommendations(self):
        """Test regime-based strategy recommendations."""
        from integrations.trading.models.regime import RegimeDetector, MarketRegime

        # Test trending regime recommendations
        strategy = RegimeDetector.get_recommended_strategy(MarketRegime.TRENDING)

        assert strategy["style"] == "trend_following"
        assert strategy["bull_weight"] > strategy["bear_weight"]  # Favor bull in trending

        # Test choppy regime recommendations
        strategy = RegimeDetector.get_recommended_strategy(MarketRegime.CHOPPY)

        assert strategy["style"] == "mean_reversion"
        assert strategy["decision_threshold"] > 0.5  # Higher threshold in choppy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
