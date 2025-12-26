"""
Shared momentum calculation utilities for Bull/Bear agents.
Consolidates duplicate logic and provides reusable functions.
"""


def calculate_price_momentum(ohlcv, periods: int = 5):
    """
    Calculate simple price momentum over N periods.

    Args:
        ohlcv: List of OHLCV candles
        periods: Number of periods to calculate momentum over

    Returns:
        float: Momentum percentage (-1.0 to 1.0, where 1.0 = 100% gain)
    """
    if not ohlcv or len(ohlcv) < periods + 1:
        return 0.0

    current_price = float(ohlcv[-1].close)
    past_price = float(ohlcv[-(periods + 1)].close)

    if past_price == 0:
        return 0.0

    momentum = (current_price - past_price) / past_price
    return momentum


def detect_trend(ohlcv, short_period: int = 5, long_period: int = 20):
    """
    Detect price trend using short vs long period comparison.

    Args:
        ohlcv: List of OHLCV candles
        short_period: Short-term average period
        long_period: Long-term average period

    Returns:
        str: 'uptrend', 'downtrend', or 'sideways'
    """
    if not ohlcv or len(ohlcv) < long_period:
        return 'sideways'

    # Calculate simple moving averages
    recent_prices = [float(candle.close) for candle in ohlcv[-short_period:]]
    longer_prices = [float(candle.close) for candle in ohlcv[-long_period:]]

    short_avg = sum(recent_prices) / len(recent_prices)
    long_avg = sum(longer_prices) / len(longer_prices)

    # Determine trend
    if short_avg > long_avg * 1.02:  # 2% threshold
        return 'uptrend'
    elif short_avg < long_avg * 0.98:
        return 'downtrend'
    else:
        return 'sideways'
