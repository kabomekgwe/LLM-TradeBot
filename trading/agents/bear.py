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

        Uses multi-factor technical analysis combining RSI, MACD, and Bollinger Bands
        from QuantAnalyst to identify bearish opportunities.

        Args:
            context: Must contain "quant_analyst" with indicators from QuantAnalystAgent

        Returns:
            Context updated with "bear_vote" containing:
                - "action": "sell" or "hold"
                - "confidence": Float 0-1
                - "reasoning": String explanation
                - "factors": List of contributing factors

        Example:
            >>> result = await agent.execute(context)
            >>> result["bear_vote"]["action"]
            'sell'
        """
        self.log_decision("Analyzing bearish signals")

        # Get indicators from QuantAnalyst
        quant_data = context.get("quant_analyst", {})
        indicators = quant_data.get("indicators", {})

        if not indicators:
            return {
                "bear_vote": {
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

        # Calculate bearish factors (each contributes to confidence)
        factors = []
        reasons = []

        # Factor 1: RSI overbought (strong bearish signal) - 40% weight
        rsi_value = rsi.get("value", 50)
        if rsi.get("overbought", False):
            factors.append(0.4)
            reasons.append(f"RSI overbought ({rsi_value:.1f})")
        elif rsi_value > 50:
            factors.append(0.2)  # Mildly bearish
            reasons.append(f"RSI above neutral ({rsi_value:.1f})")

        # Factor 2: MACD bearish crossover - 30% weight
        macd_bullish = macd.get("bullish", True)
        if not macd_bullish:  # Bearish when not bullish
            factors.append(0.3)
            macd_hist = macd.get("histogram", 0)
            reasons.append(f"MACD bearish crossover (hist={macd_hist:.4f})")

        # Factor 3: Price near upper Bollinger Band (overextension) - 30% weight
        bb_position = bb.get("position", "middle")
        if bb_position in ["upper", "middle_upper"]:
            factors.append(0.3)
            reasons.append(f"Price near upper BB ({bb_position})")

        # Calculate overall confidence (additive, capped at 1.0)
        confidence = min(sum(factors), 1.0)

        # Vote: Bearish if confidence > threshold
        vote_threshold = 0.3
        action = "sell" if confidence > vote_threshold else "hold"

        bear_vote = {
            "action": action,
            "confidence": confidence,
            "reasoning": "; ".join(reasons) if reasons else "No bearish signals detected",
            "factors": reasons,
            "direction": "bearish" if action == "sell" else "neutral"
        }

        self.log_decision(f"Bear vote: {action} (confidence={confidence:.2f}, factors={len(factors)})")

        return {"bear_vote": bear_vote}
