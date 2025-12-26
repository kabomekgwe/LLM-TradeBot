"""Trading memory and learning subsystem.

This package provides:
- Trade journaling (file-based)
- Pattern detection and learning
- Optional Graphiti integration for semantic memory
"""

from .trade_history import TradeJournal, TradeRecord
from .patterns import PatternDetector, TradingInsight

__all__ = [
    "TradeJournal",
    "TradeRecord",
    "PatternDetector",
    "TradingInsight",
]
