"""Market regime detection models.

Models for classifying and detecting market regimes.
"""

from dataclasses import dataclass
from enum import Enum


class MarketRegime(str, Enum):
    """Market regime enumeration."""
    TRENDING = "trending"  # Strong directional movement
    CHOPPY = "choppy"  # Sideways, range-bound
    VOLATILE = "volatile"  # High volatility, unpredictable
    NEUTRAL = "neutral"  # No clear pattern


@dataclass
class RegimeDetector:
    """Market regime detection utility.

    Analyzes price action to classify current market conditions.
    """

    @staticmethod
    def detect_from_candles(candles: list, lookback: int = 20) -> MarketRegime:
        """Detect market regime from OHLCV candles.

        Args:
            candles: List of OHLCV objects
            lookback: Number of candles to analyze

        Returns:
            MarketRegime classification

        Example:
            >>> from models.market_data import OHLCV
            >>> candles = [OHLCV(...), ...]  # List of candles
            >>> regime = RegimeDetector.detect_from_candles(candles)
            >>> regime
            MarketRegime.TRENDING
        """
        if not candles or len(candles) < lookback:
            return MarketRegime.NEUTRAL

        recent = candles[-lookback:]
        closes = [c.close for c in recent]

        # Calculate directional consistency
        price_changes = [
            closes[i] - closes[i-1]
            for i in range(1, len(closes))
        ]

        positive_count = sum(1 for pc in price_changes if pc > 0)
        positive_ratio = positive_count / len(price_changes)

        # Calculate volatility
        price_range = max(closes) - min(closes)
        avg_price = sum(closes) / len(closes)
        volatility_pct = (price_range / avg_price) * 100

        # Classify regime
        if positive_ratio > 0.7 or positive_ratio < 0.3:
            # Strong directional bias
            if volatility_pct > 10:
                return MarketRegime.VOLATILE
            else:
                return MarketRegime.TRENDING
        elif 0.4 <= positive_ratio <= 0.6:
            # No clear direction
            if volatility_pct > 10:
                return MarketRegime.VOLATILE
            else:
                return MarketRegime.CHOPPY
        else:
            return MarketRegime.NEUTRAL

    @staticmethod
    def get_regime_description(regime: MarketRegime) -> str:
        """Get human-readable description of regime.

        Args:
            regime: Market regime

        Returns:
            Description string
        """
        descriptions = {
            MarketRegime.TRENDING: "Strong directional movement - favor trend followers",
            MarketRegime.CHOPPY: "Sideways range-bound - favor mean reversion",
            MarketRegime.VOLATILE: "High volatility - reduce position sizes",
            MarketRegime.NEUTRAL: "No clear pattern - exercise caution",
        }
        return descriptions.get(regime, "Unknown regime")

    @staticmethod
    def get_recommended_strategy(regime: MarketRegime) -> dict:
        """Get recommended trading strategy for regime.

        Args:
            regime: Market regime

        Returns:
            Dict with strategy recommendations
        """
        strategies = {
            MarketRegime.TRENDING: {
                "style": "trend_following",
                "bull_weight": 0.6,
                "bear_weight": 0.4,
                "decision_threshold": 0.5,
                "description": "Favor Bull agent in uptrends, Bear in downtrends",
            },
            MarketRegime.CHOPPY: {
                "style": "mean_reversion",
                "bull_weight": 0.4,
                "bear_weight": 0.6,
                "decision_threshold": 0.6,
                "description": "Favor mean reversion signals, higher threshold",
            },
            MarketRegime.VOLATILE: {
                "style": "reduced_exposure",
                "bull_weight": 0.5,
                "bear_weight": 0.5,
                "decision_threshold": 0.7,
                "description": "Equal weights, very high threshold for safety",
            },
            MarketRegime.NEUTRAL: {
                "style": "balanced",
                "bull_weight": 0.5,
                "bear_weight": 0.5,
                "decision_threshold": 0.6,
                "description": "Balanced approach, moderate threshold",
            },
        }
        return strategies.get(regime, strategies[MarketRegime.NEUTRAL])
