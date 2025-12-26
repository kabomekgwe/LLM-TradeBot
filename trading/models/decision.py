"""Trading decision models.

Models for agent votes and final trading decisions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VoteAction(str, Enum):
    """Vote action enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Vote:
    """Agent vote structure.

    Represents a single agent's trading recommendation.
    """

    agent_name: str  # Agent that created this vote (e.g., "BullAgent")
    action: VoteAction
    confidence: float  # Confidence score 0-1
    reasoning: str = ""  # Explanation of vote

    @property
    def is_bullish(self) -> bool:
        """Whether vote is bullish (buy)."""
        return self.action == VoteAction.BUY

    @property
    def is_bearish(self) -> bool:
        """Whether vote is bearish (sell)."""
        return self.action == VoteAction.SELL

    @property
    def is_neutral(self) -> bool:
        """Whether vote is neutral (hold)."""
        return self.action == VoteAction.HOLD

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "agent": self.agent_name,
            "action": self.action.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class Decision:
    """Final trading decision from DecisionCoreAgent.

    Aggregates all agent votes into a single actionable decision.
    """

    action: VoteAction
    confidence: float  # Final confidence score 0-1
    regime: str  # Market regime ("trending", "choppy", "neutral")
    weighted_score: float  # Calculated weighted score
    adversarial_alignment: str  # "aligned", "opposed", or "neutral"

    # Individual votes
    bull_vote: dict = field(default_factory=dict)
    bear_vote: dict = field(default_factory=dict)
    quant_signals: dict = field(default_factory=dict)
    ml_prediction: dict = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        """Whether decision is to buy."""
        return self.action == VoteAction.BUY

    @property
    def is_sell(self) -> bool:
        """Whether decision is to sell."""
        return self.action == VoteAction.SELL

    @property
    def is_hold(self) -> bool:
        """Whether decision is to hold."""
        return self.action == VoteAction.HOLD

    @property
    def is_high_confidence(self) -> bool:
        """Whether decision has high confidence (>= 0.7)."""
        return self.confidence >= 0.7

    @property
    def is_low_confidence(self) -> bool:
        """Whether decision has low confidence (<= 0.4)."""
        return self.confidence <= 0.4

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "regime": self.regime,
            "weighted_score": self.weighted_score,
            "adversarial_alignment": self.adversarial_alignment,
            "bull_vote": self.bull_vote,
            "bear_vote": self.bear_vote,
            "quant_signals": self.quant_signals,
            "ml_prediction": self.ml_prediction,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Decision":
        """Create from dictionary."""
        return cls(
            action=VoteAction(data["action"]),
            confidence=data["confidence"],
            regime=data["regime"],
            weighted_score=data["weighted_score"],
            adversarial_alignment=data["adversarial_alignment"],
            bull_vote=data.get("bull_vote", {}),
            bear_vote=data.get("bear_vote", {}),
            quant_signals=data.get("quant_signals", {}),
            ml_prediction=data.get("ml_prediction", {}),
        )
