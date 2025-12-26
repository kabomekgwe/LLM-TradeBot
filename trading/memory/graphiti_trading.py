"""Optional Graphiti integration for trading memory.

Provides semantic search and cross-session learning capabilities.
Gracefully degrades if Graphiti is not available or configured.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from .trade_history import TradeRecord
from ..models.decision import TradingDecision


class TradingMemory:
    """Graphiti-powered trading memory with semantic search.

    Falls back to basic functionality if Graphiti is not available.
    """

    def __init__(self, spec_name: str):
        """Initialize trading memory.

        Args:
            spec_name: Spec name for memory isolation
        """
        self.spec_name = spec_name
        self.logger = logging.getLogger(__name__)
        self.graphiti = None
        self._init_graphiti()

    def _init_graphiti(self):
        """Initialize Graphiti if available and configured."""
        try:
            from graphiti_core import Graphiti
            from graphiti_core.nodes import EpisodeType

            # Check if Graphiti is configured
            import os

            if not os.getenv("GRAPHITI_ENABLED", "").lower() == "true":
                self.logger.info("Graphiti not enabled (GRAPHITI_ENABLED=false)")
                return

            # Initialize Graphiti
            self.graphiti = Graphiti(f"trading_{self.spec_name}")
            self.logger.info("Graphiti trading memory initialized")

        except ImportError:
            self.logger.info(
                "Graphiti not available - semantic memory disabled. "
                "Install with: pip install graphiti-core"
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize Graphiti: {e}")

    def is_available(self) -> bool:
        """Check if Graphiti memory is available."""
        return self.graphiti is not None

    async def save_trade_episode(
        self,
        trade: TradeRecord,
        decision: Optional[TradingDecision] = None,
    ):
        """Save a trade as an episode in Graphiti memory.

        Args:
            trade: Completed trade record
            decision: Decision context that led to the trade
        """
        if not self.is_available():
            return

        try:
            from graphiti_core.nodes import EpisodeType

            # Build episode narrative
            narrative = self._build_trade_narrative(trade, decision)

            # Create episode
            episode = await self.graphiti.add_episode(
                name=f"Trade {trade.trade_id}",
                episode_body=narrative,
                source_description=f"Trading episode for {trade.symbol}",
                reference_time=datetime.fromtimestamp(trade.timestamp / 1000),
                episode_type=EpisodeType.json,
            )

            self.logger.info(f"Saved trade episode: {trade.trade_id}")

        except Exception as e:
            self.logger.error(f"Failed to save trade episode: {e}")

    def _build_trade_narrative(
        self,
        trade: TradeRecord,
        decision: Optional[TradingDecision] = None,
    ) -> str:
        """Build a narrative description of the trade for Graphiti.

        Args:
            trade: Trade record
            decision: Decision context

        Returns:
            Narrative string
        """
        narrative = f"Executed {trade.side} order for {trade.amount} {trade.symbol} at ${trade.entry_price:.2f}. "

        if trade.closed and trade.exit_price:
            outcome = "won" if trade.won else "lost"
            narrative += f"Trade closed at ${trade.exit_price:.2f} and {outcome}, "
            narrative += f"realizing ${trade.realized_pnl:.2f} P&L ({trade.pnl_pct:.2f}%). "

        if trade.market_regime:
            narrative += f"Market regime was {trade.market_regime}. "

        if decision:
            narrative += f"Decision made with {decision.confidence * 100:.1f}% confidence. "
            narrative += f"Bull agent: {decision.bull_vote.confidence * 100:.1f}% confidence - {decision.bull_vote.reasoning[:100]}. "
            narrative += f"Bear agent: {decision.bear_vote.confidence * 100:.1f}% confidence - {decision.bear_vote.reasoning[:100]}. "

            if decision.risk_veto:
                narrative += f"Risk audit veto: {decision.veto_reason}. "

        return narrative

    async def find_similar_market_conditions(
        self,
        current_decision: TradingDecision,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find similar past market conditions using semantic search.

        Args:
            current_decision: Current trading decision
            limit: Number of similar episodes to return

        Returns:
            List of similar episodes with context
        """
        if not self.is_available():
            return []

        try:
            # Build search query
            query = f"Market regime {current_decision.regime.value}, "
            query += f"Bull confidence {current_decision.bull_vote.confidence * 100:.1f}%, "
            query += f"Bear confidence {current_decision.bear_vote.confidence * 100:.1f}%, "
            query += f"Trading {current_decision.symbol}"

            # Search for similar episodes
            results = await self.graphiti.search(query, num_results=limit)

            return [
                {
                    "episode_id": result.uuid,
                    "narrative": result.content,
                    "relevance": result.relevance_score,
                    "timestamp": result.created_at,
                }
                for result in results
            ]

        except Exception as e:
            self.logger.error(f"Failed to search similar conditions: {e}")
            return []

    async def learn_from_mistakes(self) -> List[str]:
        """Analyze past losing trades to identify patterns.

        Returns:
            List of insights from losing trades
        """
        if not self.is_available():
            return []

        try:
            # Search for losing trades
            query = "losing trade, negative PnL, loss"
            results = await self.graphiti.search(query, num_results=20)

            insights = []

            # Group by common patterns
            regime_losses = {}
            for result in results:
                # Extract regime from narrative if present
                for regime in ["trending", "choppy", "volatile", "neutral"]:
                    if regime in result.content.lower():
                        regime_losses[regime] = regime_losses.get(regime, 0) + 1

            # Generate insights
            if regime_losses:
                worst_regime = max(regime_losses, key=regime_losses.get)
                insights.append(
                    f"High frequency of losses in {worst_regime} markets "
                    f"({regime_losses[worst_regime]} occurrences). "
                    f"Consider reducing exposure in this regime."
                )

            return insights

        except Exception as e:
            self.logger.error(f"Failed to learn from mistakes: {e}")
            return []

    async def get_regime_success_rate(self, regime: str) -> Optional[float]:
        """Get historical success rate for a specific market regime.

        Args:
            regime: Market regime to query

        Returns:
            Success rate (0.0 to 1.0) or None if insufficient data
        """
        if not self.is_available():
            return None

        try:
            # Search for trades in this regime
            query = f"Market regime {regime}"
            results = await self.graphiti.search(query, num_results=50)

            if len(results) < 5:
                return None  # Insufficient data

            # Count wins vs losses
            wins = sum(1 for r in results if "won" in r.content.lower())
            total = len(results)

            return wins / total if total > 0 else None

        except Exception as e:
            self.logger.error(f"Failed to get regime success rate: {e}")
            return None

    async def get_contextual_recommendation(
        self,
        decision: TradingDecision,
    ) -> Optional[str]:
        """Get a contextual recommendation based on similar past scenarios.

        Args:
            decision: Current trading decision

        Returns:
            Recommendation string or None
        """
        if not self.is_available():
            return None

        try:
            similar = await self.find_similar_market_conditions(decision, limit=10)

            if not similar:
                return None

            # Analyze outcomes of similar scenarios
            wins = sum(1 for s in similar if "won" in s["narrative"].lower())
            total = len(similar)

            if total < 3:
                return None

            win_rate = wins / total * 100

            if win_rate > 70:
                return (
                    f"Historical data shows {win_rate:.0f}% success rate in similar conditions "
                    f"({wins}/{total} trades). Strong signal to proceed."
                )
            elif win_rate < 40:
                return (
                    f"Historical data shows only {win_rate:.0f}% success rate in similar conditions "
                    f"({wins}/{total} trades). Consider passing on this trade."
                )
            else:
                return (
                    f"Historical data shows {win_rate:.0f}% success rate in similar conditions "
                    f"({wins}/{total} trades). Mixed signal - proceed with caution."
                )

        except Exception as e:
            self.logger.error(f"Failed to get contextual recommendation: {e}")
            return None
