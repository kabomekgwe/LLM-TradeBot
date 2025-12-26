"""Trading integration module.

This module provides the 8-agent adversarial trading system with multi-platform
support for exchanges and brokers.

Main components:
    - TradingManager: Orchestrates the 8-agent trading loop
    - TradingConfig: Configuration from environment variables
    - TradingState: Per-spec state tracking
    - BaseExchangeProvider: Abstract provider interface
    - 8 trading agents: DataSync, QuantAnalyst, Predict, Bull, Bear,
      DecisionCore, RiskAudit, ExecutionEngine

Example usage:
    >>> from pathlib import Path
    >>> from integrations.trading import TradingManager
    >>>
    >>> manager = TradingManager(Path("specs/001/"), provider="binance_futures")
    >>> if manager.enabled:
    ...     result = await manager.run_trading_loop("BTC/USDT")
    ...     print(f"Executed: {result['action']}")

For detailed documentation, see: /Users/kabo/.claude/plans/mighty-painting-nygaard.md
"""

from .config import TradingConfig
from .state import TradingState
from .manager import TradingManager
from .providers.base import BaseExchangeProvider
from .providers.factory import create_provider

__all__ = [
    "TradingConfig",
    "TradingState",
    "TradingManager",
    "BaseExchangeProvider",
    "create_provider",
]

__version__ = "1.0.0-phase1"
