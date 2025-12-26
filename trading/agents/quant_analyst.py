"""QuantAnalystAgent - Technical signal generation.

This is Agent #2 in the 8-agent system.
Generates technical signals via trend, oscillator, and sentiment sub-analysis.
"""

from typing import Any

from .base_agent import BaseAgent


class QuantAnalystAgent(BaseAgent):
    """Quant analyst agent - generates technical signals.

    Analyzes market using technical indicators and generates
    trading signals for trend, oscillators, and sentiment.

    TODO: Full implementation with TA-Lib indicators in future iteration.
    """

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate technical analysis signals.

        Args:
            context: Must contain "market_data" from DataSyncAgent

        Returns:
            Context updated with "quant_signals" containing:
                - "trend": Trend signal ("up", "down", "neutral")
                - "oscillator": Oscillator reading (0-100)
                - "sentiment": Market sentiment (-1 to 1)
                - "regime": Market regime ("trending" or "choppy")

        Example:
            >>> result = await agent.execute(context)
            >>> result["quant_signals"]["trend"]
            'up'
        """
        market_data = context.get("market_data", {})
        if not market_data:
            raise ValueError("market_data is required in context")

        self.log_decision("Generating technical signals")

        # Get 1h candles for analysis
        candles_1h = market_data.get("1h", [])
        if len(candles_1h) < 20:
            # Not enough data for analysis
            return {
                "quant_signals": {
                    "trend": "neutral",
                    "oscillator": 50,
                    "sentiment": 0.0,
                    "regime": "neutral",
                }
            }

        # Simple trend detection (MVP - Phase 1)
        # TODO: Replace with proper RSI, MACD, Bollinger Bands from TA-Lib
        recent_closes = [c.close for c in candles_1h[-20:]]

        # Trend: compare recent average to older average
        recent_avg = sum(recent_closes[-5:]) / 5
        older_avg = sum(recent_closes[:5]) / 5

        if recent_avg > older_avg * 1.02:  # 2% higher
            trend = "up"
            sentiment = 0.5
        elif recent_avg < older_avg * 0.98:  # 2% lower
            trend = "down"
            sentiment = -0.5
        else:
            trend = "neutral"
            sentiment = 0.0

        # Oscillator: simple momentum (0-100)
        momentum = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
        oscillator = max(0, min(100, 50 + momentum * 500))  # Normalize to 0-100

        # Regime detection: trending if consistent direction
        price_changes = [
            recent_closes[i] - recent_closes[i-1]
            for i in range(1, len(recent_closes))
        ]
        consistent_direction = sum(1 for pc in price_changes if pc > 0) / len(price_changes)

        if consistent_direction > 0.7 or consistent_direction < 0.3:
            regime = "trending"
        else:
            regime = "choppy"

        quant_signals = {
            "trend": trend,
            "oscillator": oscillator,
            "sentiment": sentiment,
            "regime": regime,
        }

        self.log_decision(
            f"Signals: trend={trend}, oscillator={oscillator:.1f}, regime={regime}"
        )

        return {"quant_signals": quant_signals, "regime": regime}
