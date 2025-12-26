"""BearAgent - Bearish market analysis.

This is Agent #5 in the 8-agent system.
Analyzes market exclusively for short (bearish) opportunities.
"""

from typing import Any

from .base_agent import BaseAgent


class BearAgent(BaseAgent):
    """Bear agent - exclusively analyzes bearish signals.

    Provides pessimistic market analysis with short bias.
    """

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze market for bearish opportunities.

        Args:
            context: Must contain "market_data" from DataSyncAgent

        Returns:
            Context updated with "bear_vote" containing:
                - "action": "sell" or "hold"
                - "confidence": Float 0-1
                - "reasoning": String explanation

        Example:
            >>> result = await agent.execute(context)
            >>> result["bear_vote"]["action"]
            'sell'
        """
        market_data = context.get("market_data", {})
        if not market_data:
            raise ValueError("market_data is required in context")

        self.log_decision("Analyzing bearish signals")

        # Get recent price action
        candles_1h = market_data.get("1h", [])
        if not candles_1h:
            return {"bear_vote": {"action": "hold", "confidence": 0.0, "reasoning": "No data"}}

        # Simple bearish analysis (Phase 1 MVP implementation)
        # TODO: Full implementation with technical indicators
        recent_candles = candles_1h[-10:] if len(candles_1h) >= 10 else candles_1h
        closes = [c.close for c in recent_candles]

        # Check if price is trending down
        is_downtrend = closes[-1] < closes[0] if len(closes) > 1 else False

        # Calculate simple momentum (negative for downtrend)
        momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0

        if is_downtrend and momentum < -0.02:  # 2% loss
            action = "sell"
            confidence = min(abs(momentum) * 10, 0.8)  # Cap at 0.8
            reasoning = f"Downtrend detected with {momentum*100:.2f}% momentum"
        else:
            action = "hold"
            confidence = 0.3
            reasoning = "No strong bearish signal"

        bear_vote = {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        self.log_decision(f"Bear vote: {action} (confidence={confidence:.2f})")

        return {"bear_vote": bear_vote}
