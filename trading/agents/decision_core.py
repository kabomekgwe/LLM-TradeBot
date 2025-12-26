"""DecisionCoreAgent - Weighted voting with adversarial alignment.

This is Agent #6 in the 8-agent system and the CORE of the adversarial
decision framework (ADF). Aggregates votes from Bull and Bear agents with
regime-aware weighting.
"""

from typing import Any

from .base_agent import BaseAgent


class DecisionCoreAgent(BaseAgent):
    """Decision core agent with weighted voting.

    Implements the adversarial decision framework by:
    1. Receiving votes from Bull and Bear agents
    2. Applying regime-based weighting (trending vs choppy)
    3. Calculating confidence scores with adversarial alignment
    4. Outputting final trading decision
    """

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Aggregate votes from Bull/Bear agents with regime awareness.

        Args:
            context: Must contain:
                - "bull_vote": Dict with action, confidence, reasoning
                - "bear_vote": Dict with action, confidence, reasoning
                - "regime": Market regime ("trending", "choppy", or "neutral")

        Returns:
            Context updated with "decision" containing:
                - "action": "buy", "sell", or "hold"
                - "confidence": Float 0-1
                - "regime": Market regime
                - "bull_vote": Original bull vote
                - "bear_vote": Original bear vote

        Example:
            >>> context = {
            ...     "bull_vote": {"action": "buy", "confidence": 0.8},
            ...     "bear_vote": {"action": "sell", "confidence": 0.3},
            ...     "regime": "trending"
            ... }
            >>> result = await agent.execute(context)
            >>> result["decision"]["action"]
            'buy'
        """
        bull_vote = context.get("bull_vote", {})
        bear_vote = context.get("bear_vote", {})
        regime = context.get("regime", "neutral")

        self.log_decision(
            f"Aggregating votes in {regime} regime: "
            f"Bull={bull_vote.get('action')} ({bull_vote.get('confidence', 0):.2f}), "
            f"Bear={bear_vote.get('action')} ({bear_vote.get('confidence', 0):.2f})"
        )

        # Determine regime-based weights
        weights = self._get_regime_weights(regime)

        # Extract confidence scores
        bull_confidence = float(bull_vote.get("confidence", 0.0))
        bear_confidence = float(bear_vote.get("confidence", 0.0))

        # Calculate weighted score
        # Bull votes are positive, bear votes are negative
        bull_score = bull_confidence if bull_vote.get("action") == "buy" else 0
        bear_score = bear_confidence if bear_vote.get("action") == "sell" else 0

        weighted_score = (
            bull_score * weights["bull"] -
            bear_score * weights["bear"]
        )

        # Determine final action based on weighted score
        action, confidence = self._determine_action(
            weighted_score, bull_vote, bear_vote
        )

        # Check adversarial alignment
        alignment = self._check_adversarial_alignment(bull_vote, bear_vote)

        decision = {
            "action": action,
            "confidence": confidence,
            "regime": regime,
            "weighted_score": weighted_score,
            "adversarial_alignment": alignment,
            "bull_vote": bull_vote,
            "bear_vote": bear_vote,
        }

        self.log_decision(
            f"Final decision: {action} (confidence={confidence:.2f}, "
            f"alignment={alignment})"
        )

        return {"decision": decision}

    def _get_regime_weights(self, regime: str) -> dict[str, float]:
        """Get agent weights based on market regime.

        In trending markets, favor trend followers (Bull).
        In choppy markets, favor mean reversion (Bear).

        Args:
            regime: Market regime ("trending", "choppy", "neutral")

        Returns:
            Dict with "bull" and "bear" weights
        """
        regime_weights = {
            "trending": {"bull": 0.6, "bear": 0.4},
            "choppy": {"bull": 0.4, "bear": 0.6},
            "neutral": {"bull": 0.5, "bear": 0.5},
        }

        return regime_weights.get(regime, {"bull": 0.5, "bear": 0.5})

    def _determine_action(
        self,
        weighted_score: float,
        bull_vote: dict,
        bear_vote: dict
    ) -> tuple[str, float]:
        """Determine final action from weighted score.

        Args:
            weighted_score: Calculated weighted score (-1 to 1)
            bull_vote: Bull agent vote
            bear_vote: Bear agent vote

        Returns:
            (action, confidence) tuple
        """
        threshold = self.config.decision_threshold

        if weighted_score > threshold:
            # Strong buy signal
            return "buy", min(abs(weighted_score), 1.0)
        elif weighted_score < -threshold:
            # Strong sell signal
            return "sell", min(abs(weighted_score), 1.0)
        else:
            # No clear signal, hold
            return "hold", 0.0

    def _check_adversarial_alignment(
        self, bull_vote: dict, bear_vote: dict
    ) -> str:
        """Check if Bull and Bear agents agree or disagree.

        Strong alignment (both bullish or both bearish) can indicate
        clearer market conditions. Disagreement requires more caution.

        Args:
            bull_vote: Bull agent vote
            bear_vote: Bear agent vote

        Returns:
            "aligned", "opposed", or "neutral"
        """
        bull_action = bull_vote.get("action", "hold")
        bear_action = bear_vote.get("action", "hold")

        if bull_action == "buy" and bear_action == "hold":
            return "aligned"  # Both not bearish
        elif bull_action == "hold" and bear_action == "sell":
            return "aligned"  # Both not bullish
        elif bull_action == "buy" and bear_action == "sell":
            return "opposed"  # Direct conflict
        else:
            return "neutral"
