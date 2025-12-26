"""BullAgent - Bullish market analysis.

This is Agent #4 in the 8-agent system.
Analyzes market exclusively for long (bullish) opportunities.
"""

from typing import Any

from .base_agent import BaseAgent


class BullAgent(BaseAgent):
    """Bull agent - exclusively analyzes bullish signals.

    Provides optimistic market analysis with long bias.
    """

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze market for bullish opportunities.

        Args:
            context: Must contain "market_data" from DataSyncAgent

        Returns:
            Context updated with "bull_vote" containing:
                - "action": "buy" or "hold"
                - "confidence": Float 0-1
                - "reasoning": String explanation

        Example:
            >>> result = await agent.execute(context)
            >>> result["bull_vote"]["action"]
            'buy'
        """
        market_data = context.get("market_data", {})
        if not market_data:
            raise ValueError("market_data is required in context")

        self.log_decision("Analyzing bullish signals")

        # Get recent price action
        candles_1h = market_data.get("1h", [])
        if not candles_1h:
            return {"bull_vote": {"action": "hold", "confidence": 0.0, "reasoning": "No data"}}

        # Simple bullish analysis (Phase 1 MVP implementation)
        # TODO: Full implementation with technical indicators
        recent_candles = candles_1h[-10:] if len(candles_1h) >= 10 else candles_1h
        closes = [c.close for c in recent_candles]

        # Check if price is trending up
        is_uptrend = closes[-1] > closes[0] if len(closes) > 1 else False

        # Calculate simple momentum
        momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0

        if is_uptrend and momentum > 0.02:  # 2% gain
            action = "buy"
            confidence = min(abs(momentum) * 10, 0.8)  # Cap at 0.8
            reasoning = f"Uptrend detected with {momentum*100:.2f}% momentum"
        else:
            action = "hold"
            confidence = 0.3
            reasoning = "No strong bullish signal"

        bull_vote = {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        self.log_decision(f"Bull vote: {action} (confidence={confidence:.2f})")

        return {"bull_vote": bull_vote}
