"""
Shared pytest fixtures for LLM-TradeBot testing.
Provides factory fixtures for market data, mock exchange, and mock models.
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock
from datetime import datetime
from dataclasses import dataclass
import numpy as np


@dataclass
class OHLCVCandle:
    """OHLCV candle data structure matching trading.types.Candle."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@pytest.fixture
def ohlcv_factory():
    """
    Factory fixture for creating OHLCV market data.

    Usage:
        def test_example(ohlcv_factory):
            data = ohlcv_factory(trend="uptrend", num_candles=100)
    """
    def _create_ohlcv(
        symbol: str = "BTC/USDT",
        start_price: float = 30000,
        num_candles: int = 100,
        timeframe: str = "5m",  # '5m', '15m', '1h'
        trend: str = "sideways",  # 'uptrend', 'downtrend', 'sideways'
        volatility: float = 0.02,  # 2% volatility
        volume_range: tuple = (100, 500)
    ):
        """
        Create synthetic OHLCV data for testing.

        Args:
            symbol: Trading pair symbol
            start_price: Starting price for first candle
            num_candles: Number of candles to generate
            timeframe: Candle timeframe (affects timestamp spacing)
            trend: Market trend ('uptrend', 'downtrend', 'sideways')
            volatility: Price volatility as percentage (0.02 = 2%)
            volume_range: Min/max trading volume per candle

        Returns:
            List of OHLCVCandle objects
        """
        import random

        # Timeframe to milliseconds
        timeframe_ms = {
            '5m': 300000,
            '15m': 900000,
            '1h': 3600000
        }.get(timeframe, 300000)

        candles = []
        current_price = start_price
        base_time = int(datetime.now().timestamp() * 1000)

        for i in range(num_candles):
            timestamp = base_time - (num_candles - i) * timeframe_ms

            # Apply trend
            if trend == "uptrend":
                current_price *= 1.001  # 0.1% increase per candle
            elif trend == "downtrend":
                current_price *= 0.999  # 0.1% decrease per candle
            # sideways: no change to current_price

            # Generate OHLC with randomness
            open_price = current_price
            high = open_price * (1 + random.uniform(0, volatility))
            low = open_price * (1 - random.uniform(0, volatility))
            close = random.uniform(low, high)
            volume = random.uniform(*volume_range)

            candles.append(OHLCVCandle(
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume
            ))

            current_price = close

        return candles

    return _create_ohlcv


@pytest_asyncio.fixture
async def mock_exchange():
    """
    Mock CCXT exchange with realistic async API responses.

    Usage:
        async def test_example(mock_exchange):
            data = await mock_exchange.fetch_ohlcv('BTC/USDT', '5m')
    """
    exchange = AsyncMock()

    # Mock fetch_ohlcv (returns list of OHLCV arrays)
    exchange.fetch_ohlcv = AsyncMock(return_value=[
        [1609459200000, 29000, 29500, 28800, 29200, 1250.5],
        [1609462800000, 29200, 29600, 29100, 29400, 1180.2],
    ])

    # Mock fetch_ticker
    exchange.fetch_ticker = AsyncMock(return_value={
        'symbol': 'BTC/USDT',
        'last': 29400,
        'bid': 29398,
        'ask': 29402,
        'baseVolume': 1250.5,
        'quoteVolume': 36750000,
        'timestamp': 1609462800000
    })

    # Mock fetch_balance
    exchange.fetch_balance = AsyncMock(return_value={
        'USDT': {'free': 10000, 'used': 0, 'total': 10000},
        'BTC': {'free': 0.5, 'used': 0, 'total': 0.5}
    })

    # Mock fetch_order_book
    exchange.fetch_order_book = AsyncMock(return_value={
        'bids': [[29398, 1.5], [29397, 2.0]],
        'asks': [[29402, 1.2], [29403, 1.8]]
    })

    # Mock create_order
    exchange.create_order = AsyncMock(return_value={
        'id': 'test-order-123',
        'status': 'closed',
        'symbol': 'BTC/USDT',
        'type': 'limit',
        'side': 'buy',
        'price': 29400,
        'amount': 0.01,
        'filled': 0.01,
        'remaining': 0.0,
        'timestamp': 1609462800000
    })

    # Mock async context manager (for async with exchange: usage)
    exchange.__aenter__ = AsyncMock(return_value=exchange)
    exchange.__aexit__ = AsyncMock(return_value=None)

    return exchange


@pytest.fixture
def mock_lightgbm_model():
    """
    Mock LightGBM Booster for testing agents without training.

    Usage:
        def test_predict_agent(mock_lightgbm_model):
            prediction = mock_lightgbm_model.predict([[65, 0.5, 0.3, 0.2, 30100, 30000, 29900, 0.01]])
    """
    mock_model = Mock()

    # Mock predict (returns numpy array of probabilities)
    mock_model.predict = Mock(return_value=np.array([0.75]))  # 75% probability of price up

    # Mock best_iteration (for early stopping)
    mock_model.best_iteration = 100

    # Mock num_trees
    mock_model.num_trees = Mock(return_value=100)

    return mock_model


# Parametrized market scenarios for comprehensive testing
@pytest.fixture(params=[
    {"name": "strong_uptrend", "trend": "uptrend", "num_candles": 50, "expected_bullish": True},
    {"name": "strong_downtrend", "trend": "downtrend", "num_candles": 50, "expected_bullish": False},
    {"name": "sideways_market", "trend": "sideways", "num_candles": 100, "expected_bullish": None},
])
def market_scenario(request, ohlcv_factory):
    """
    Parametrized fixture providing different market scenarios.

    Usage:
        def test_agent_scenarios(market_scenario):
            data = market_scenario['data']
            expected = market_scenario['expected_bullish']
    """
    params = request.param
    return {
        'name': params['name'],
        'data': ohlcv_factory(
            trend=params['trend'],
            num_candles=params['num_candles']
        ),
        'expected_bullish': params['expected_bullish']
    }
