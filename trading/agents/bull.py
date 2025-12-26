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

        Uses multi-factor technical analysis combining RSI, MACD, and Bollinger Bands
        from QuantAnalyst to identify bullish opportunities.

        Args:
            context: Must contain "quant_analyst" with indicators from QuantAnalystAgent

        Returns:
            Context updated with "bull_vote" containing:
                - "action": "buy" or "hold"
                - "confidence": Float 0-1
                - "reasoning": String explanation
                - "factors": List of contributing factors

        Example:
            >>> result = await agent.execute(context)
            >>> result["bull_vote"]["action"]
            'buy'
        """
        self.log_decision("Analyzing bullish signals")

        # Get indicators from QuantAnalyst
        quant_data = context.get("quant_analyst", {})
        indicators = quant_data.get("indicators", {})

        if not indicators:
            return {
                "bull_vote": {
                    "action": "hold",
                    "confidence": 0.0,
                    "reasoning": "No indicator data available",
                    "factors": []
                }
            }

        # Extract indicator signals
        rsi = indicators.get("rsi", {})
        macd = indicators.get("macd", {})
        bb = indicators.get("bollinger", {})

        # Calculate bullish factors (each contributes to confidence)
        factors = []
        reasons = []

        # Factor 1: RSI oversold (strong bullish signal) - 40% weight
        rsi_value = rsi.get("value", 50)
        if rsi.get("oversold", False):
            factors.append(0.4)
            reasons.append(f"RSI oversold ({rsi_value:.1f})")
        elif rsi_value < 50:
            factors.append(0.2)  # Mildly bullish
            reasons.append(f"RSI below neutral ({rsi_value:.1f})")

        # Factor 2: MACD bullish crossover - 30% weight
        if macd.get("bullish", False):
            factors.append(0.3)
            macd_hist = macd.get("histogram", 0)
            reasons.append(f"MACD bullish crossover (hist={macd_hist:.4f})")

        # Factor 3: Price near lower Bollinger Band (reversion opportunity) - 30% weight
        bb_position = bb.get("position", "middle")
        if bb_position in ["lower", "middle_lower"]:
            factors.append(0.3)
            reasons.append(f"Price near lower BB ({bb_position})")

        # Calculate overall confidence (additive, capped at 1.0)
        confidence = min(sum(factors), 1.0)

        # Vote: Bullish if confidence > threshold
        vote_threshold = 0.3
        action = "buy" if confidence > vote_threshold else "hold"

        bull_vote = {
            "action": action,
            "confidence": confidence,
            "reasoning": "; ".join(reasons) if reasons else "No bullish signals detected",
            "factors": reasons,
            "direction": "bullish" if action == "buy" else "neutral"
        }

        self.log_decision(f"Bull vote: {action} (confidence={confidence:.2f}, factors={len(factors)})")

        return {"bull_vote": bull_vote}
