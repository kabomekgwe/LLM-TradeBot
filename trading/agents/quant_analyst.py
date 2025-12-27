"""QuantAnalystAgent - Technical signal generation.

This is Agent #2 in the 8-agent system.
Generates technical signals via trend, oscillator, and sentiment sub-analysis.
"""

import numpy as np
import talib
from typing import Any

from .base_agent import BaseAgent
from ..exceptions import (
    InvalidIndicatorDataError,
    InsufficientMarketDataError,
    TradingBotError,
)


class QuantAnalystAgent(BaseAgent):
    """Quant analyst agent - generates technical signals.

    Analyzes market using technical indicators (RSI, MACD, Bollinger Bands)
    and generates trading signals for trend, oscillators, and sentiment.
    """

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate technical analysis signals using TA-Lib indicators.

        Args:
            context: Must contain "market_data" from DataSyncAgent

        Returns:
            Context updated with "quant_signals" containing:
                - "indicators": Dict with RSI, MACD, Bollinger Bands data
                - "overall_signal": Overall signal interpretation
                - "trend": Trend signal (for backward compatibility)
                - "oscillator": Oscillator reading (for backward compatibility)
                - "sentiment": Market sentiment (for backward compatibility)
                - "regime": Market regime (for backward compatibility)

        Example:
            >>> result = await agent.execute(context)
            >>> result["quant_signals"]["indicators"]["rsi"]["signal"]
            'overbought'
        """
        market_data = context.get("market_data", {})
        if not market_data:
            raise ValueError("market_data is required in context")

        self.log_decision("Generating technical signals with TA-Lib indicators")

        # Get 1h candles for analysis (TA-Lib indicators work better on higher timeframes)
        candles_1h = market_data.get("1h", [])
        if len(candles_1h) < 26:  # MACD needs 26 periods minimum
            self.log_decision(
                "insufficient_market_data",
                level="warning",
                required=26,
                got=len(candles_1h),
            )
            raise InsufficientMarketDataError(
                f"Need 26+ candles for TA-Lib indicators, got {len(candles_1h)}"
            )

        # Extract close prices as numpy array (TA-Lib requirement)
        try:
            close_prices = np.array([float(c.close) for c in candles_1h])
        except (AttributeError, ValueError, TypeError) as e:
            self.log_decision(
                "invalid_candle_data",
                level="error",
                error=str(e),
            )
            raise InvalidIndicatorDataError(f"Invalid candle data structure: {e}")

        # Validate price data
        if not all(isinstance(p, (int, float)) and p > 0 for p in close_prices):
            self.log_decision(
                "invalid_price_data",
                level="error",
                prices=close_prices[:5].tolist(),  # Log first 5 for debugging
            )
            raise InvalidIndicatorDataError("Close prices contain invalid values")

        try:
            # Calculate RSI (14-period)
            rsi = talib.RSI(close_prices, timeperiod=14)
            if rsi is None or len(rsi) == 0:
                raise InvalidIndicatorDataError("RSI calculation returned empty result")
            rsi_current = rsi[-1] if not np.isnan(rsi[-1]) else 50.0  # Default to neutral if NaN
        except Exception as e:
            if isinstance(e, TradingBotError):
                raise
            self.log_decision(
                "rsi_calculation_failed",
                level="error",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise InvalidIndicatorDataError(f"RSI calculation failed: {e}")

        # Determine RSI signal
        rsi_signal = 'overbought' if rsi_current > 70 else 'oversold' if rsi_current < 30 else 'neutral'

        try:
            # Calculate MACD (12, 26, 9)
            macd, macdsignal, macdhist = talib.MACD(
                close_prices,
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )

            if macd is None or macdsignal is None or macdhist is None:
                raise InvalidIndicatorDataError("MACD calculation returned None")

            macd_current = macd[-1] if not np.isnan(macd[-1]) else 0.0
            signal_current = macdsignal[-1] if not np.isnan(macdsignal[-1]) else 0.0
            hist_current = macdhist[-1] if not np.isnan(macdhist[-1]) else 0.0

            # Determine MACD trend
            macd_signal = 'bullish' if macd_current > signal_current else 'bearish'
        except Exception as e:
            if isinstance(e, TradingBotError):
                raise
            self.log_decision(
                "macd_calculation_failed",
                level="error",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise InvalidIndicatorDataError(f"MACD calculation failed: {e}")

        try:
            # Calculate Bollinger Bands (20-period, 2 std dev)
            upperband, middleband, lowerband = talib.BBANDS(
                close_prices,
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0  # SMA
            )

            if upperband is None or middleband is None or lowerband is None:
                raise InvalidIndicatorDataError("Bollinger Bands calculation returned None")

            upper = upperband[-1] if not np.isnan(upperband[-1]) else close_prices[-1]
            middle = middleband[-1] if not np.isnan(middleband[-1]) else close_prices[-1]
            lower = lowerband[-1] if not np.isnan(lowerband[-1]) else close_prices[-1]
            current_price = close_prices[-1]
        except Exception as e:
            if isinstance(e, TradingBotError):
                raise
            self.log_decision(
                "bollinger_calculation_failed",
                level="error",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise InvalidIndicatorDataError(f"Bollinger Bands calculation failed: {e}")

        # Determine Bollinger Band position
        if current_price > middle:
            bb_position = 'upper' if current_price > (middle + (upper - middle) * 0.5) else 'middle_upper'
        else:
            bb_position = 'lower' if current_price < (middle - (middle - lower) * 0.5) else 'middle_lower'

        # Calculate bandwidth (volatility measure)
        bandwidth = float((upper - lower) / middle) if middle != 0 else 0.0

        # Build structured indicators output
        indicators = {
            'rsi': {
                'value': float(rsi_current),
                'signal': rsi_signal,
                'overbought': rsi_current > 70,
                'oversold': rsi_current < 30
            },
            'macd': {
                'macd': float(macd_current),
                'signal': float(signal_current),
                'histogram': float(hist_current),
                'trend': macd_signal,
                'bullish': macd_current > signal_current
            },
            'bollinger': {
                'upper': float(upper),
                'middle': float(middle),
                'lower': float(lower),
                'current_price': float(current_price),
                'position': bb_position,
                'bandwidth': bandwidth
            }
        }

        # Determine overall signal (for backward compatibility)
        # Priority: RSI oversold/overbought > MACD trend
        if rsi_signal == 'oversold':
            overall_signal = 'oversold'
            trend = 'down'
            sentiment = -0.7
        elif rsi_signal == 'overbought':
            overall_signal = 'overbought'
            trend = 'up'
            sentiment = 0.7
        elif macd_signal == 'bullish':
            overall_signal = 'bullish'
            trend = 'up'
            sentiment = 0.5
        elif macd_signal == 'bearish':
            overall_signal = 'bearish'
            trend = 'down'
            sentiment = -0.5
        else:
            overall_signal = 'neutral'
            trend = 'neutral'
            sentiment = 0.0

        # Oscillator value (use RSI directly)
        oscillator = rsi_current

        # Regime detection: use Bollinger Bandwidth
        # Low bandwidth = choppy/consolidation, high bandwidth = trending
        regime = 'trending' if bandwidth > 0.04 else 'choppy'

        quant_signals = {
            'indicators': indicators,
            'overall_signal': overall_signal,
            'trend': trend,
            'oscillator': oscillator,
            'sentiment': sentiment,
            'regime': regime,
        }

        self.log_decision(
            f"TA-Lib Signals: RSI={rsi_current:.1f}({rsi_signal}), "
            f"MACD={macd_signal}, BB_pos={bb_position}, regime={regime}"
        )

        return {"quant_signals": quant_signals, "regime": regime}
