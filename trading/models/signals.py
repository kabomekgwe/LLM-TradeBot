"""Trading signal models.

Models for technical analysis signals and indicators.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SignalType(str, Enum):
    """Signal type enumeration."""
    TREND = "trend"
    OSCILLATOR = "oscillator"
    SENTIMENT = "sentiment"
    VOLUME = "volume"
    MOMENTUM = "momentum"


class SignalStrength(str, Enum):
    """Signal strength enumeration."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class TradingSignal:
    """Unified trading signal structure.

    Represents a single technical analysis signal.
    """

    signal_type: SignalType
    strength: SignalStrength
    value: float  # Numerical value of indicator
    confidence: float  # Confidence score 0-1
    description: str = ""  # Human-readable description

    @property
    def is_bullish(self) -> bool:
        """Whether signal is bullish."""
        return self.strength in (SignalStrength.BUY, SignalStrength.STRONG_BUY)

    @property
    def is_bearish(self) -> bool:
        """Whether signal is bearish."""
        return self.strength in (SignalStrength.SELL, SignalStrength.STRONG_SELL)

    @property
    def is_neutral(self) -> bool:
        """Whether signal is neutral."""
        return self.strength == SignalStrength.NEUTRAL

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": self.signal_type.value,
            "strength": self.strength.value,
            "value": self.value,
            "confidence": self.confidence,
            "description": self.description,
        }


@dataclass
class IndicatorResult:
    """Result from a technical indicator calculation.

    Used by QuantAnalystAgent to store indicator values.
    """

    name: str  # Indicator name (e.g., "RSI", "MACD", "BB")
    value: float  # Primary value
    signal: Optional[SignalStrength] = None  # Interpretation
    metadata: dict = None  # Additional data (e.g., MACD signal line)

    def __post_init__(self):
        """Initialize metadata dict."""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "signal": self.signal.value if self.signal else None,
            "metadata": self.metadata,
        }
