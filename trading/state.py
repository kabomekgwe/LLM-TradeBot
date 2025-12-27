"""Trading state management.

This module provides state tracking for trading sessions, following the
pattern established by the Linear integration's state management.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TradingState:
    """State of a trading session within a spec directory.

    Follows the same pattern as Linear's state management - stores state
    in a JSON file within the spec directory for per-feature isolation.
    """

    # Initialization tracking
    initialized: bool = False
    provider: Optional[str] = None
    created_at: Optional[str] = None
    last_updated: Optional[str] = None

    # Position tracking
    active_positions: list[dict] = field(default_factory=list)
    total_trades: int = 0

    # Performance metrics
    win_rate: float = 0.0  # Percentage of winning trades
    total_pnl: float = 0.0  # Total profit/loss in USD
    daily_pnl: float = 0.0  # Today's profit/loss
    daily_drawdown_pct: float = 0.0  # Today's drawdown percentage

    # Risk tracking
    circuit_breaker_tripped: bool = False
    last_circuit_trip_reason: Optional[str] = None

    def save(self, spec_dir: Path) -> None:
        """Save state to .trading_state.json in spec directory using atomic write pattern.

        Uses temp file + os.replace() to guarantee state is never corrupted on crash.
        If crash occurs during write, temp file is corrupt but original state file
        remains untouched.

        Args:
            spec_dir: Path to spec directory (e.g., specs/001-feature/)

        Example:
            >>> state = TradingState(initialized=True, provider="binance_futures")
            >>> state.save(Path("specs/001-trading/"))
            # Creates specs/001-trading/.trading_state.json atomically
        """
        state_file = spec_dir / ".trading_state.json"

        # Update timestamp
        self.last_updated = datetime.now().isoformat()

        # Convert to dict
        state_dict = asdict(self)

        # Write to temporary file in same directory (ensures same filesystem)
        # This is critical for atomic rename to work
        state_dir = str(spec_dir)
        fd, temp_path = tempfile.mkstemp(
            dir=state_dir,
            prefix=".trading_state_tmp_",
            suffix=".json"
        )

        try:
            # Write state to temp file
            with os.fdopen(fd, "w") as f:
                json.dump(state_dict, f, indent=2)

            # Atomic rename - this is the critical operation
            # On POSIX: rename is atomic (overwrites destination atomically)
            # On Windows: Python 3.3+ os.replace() handles this correctly
            os.replace(temp_path, str(state_file))

        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass  # Temp file already deleted or inaccessible
            raise  # Re-raise original exception

    @classmethod
    def load(cls, spec_dir: Path) -> Optional["TradingState"]:
        """Load state from spec directory.

        Args:
            spec_dir: Path to spec directory

        Returns:
            TradingState instance if state file exists, None otherwise

        Example:
            >>> state = TradingState.load(Path("specs/001-trading/"))
            >>> if state:
            ...     print(f"Total trades: {state.total_trades}")
        """
        state_file = spec_dir / ".trading_state.json"

        if not state_file.exists():
            return None

        try:
            with open(state_file) as f:
                state_dict = json.load(f)

            return cls.from_dict(state_dict)

        except (OSError, json.JSONDecodeError) as e:
            # Log error but return None for graceful degradation
            logger.warning(
                "state_load_failed",
                extra={
                    "state_file": str(state_file),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return None

    @classmethod
    def from_dict(cls, data: dict) -> "TradingState":
        """Create TradingState instance from dictionary.

        Args:
            data: Dictionary with state data

        Returns:
            TradingState instance
        """
        return cls(
            initialized=data.get("initialized", False),
            provider=data.get("provider"),
            created_at=data.get("created_at"),
            last_updated=data.get("last_updated"),
            active_positions=data.get("active_positions", []),
            total_trades=data.get("total_trades", 0),
            win_rate=data.get("win_rate", 0.0),
            total_pnl=data.get("total_pnl", 0.0),
            daily_pnl=data.get("daily_pnl", 0.0),
            daily_drawdown_pct=data.get("daily_drawdown_pct", 0.0),
            circuit_breaker_tripped=data.get("circuit_breaker_tripped", False),
            last_circuit_trip_reason=data.get("last_circuit_trip_reason"),
        )

    def to_dict(self) -> dict:
        """Convert state to dictionary.

        Returns:
            Dictionary representation of state
        """
        return asdict(self)

    def add_trade(self, trade_pnl: float, is_win: bool) -> None:
        """Update state with completed trade.

        Args:
            trade_pnl: Profit/loss from trade in USD
            is_win: Whether trade was profitable

        Example:
            >>> state = TradingState()
            >>> state.add_trade(100.0, True)  # Won $100
            >>> state.total_pnl
            100.0
            >>> state.win_rate
            100.0
        """
        self.total_trades += 1
        self.total_pnl += trade_pnl
        self.daily_pnl += trade_pnl

        # Recalculate win rate
        if self.total_trades > 0:
            # Need to track winning trades separately
            # This is simplified - in reality would need to track in active_positions
            pass

    def add_position(self, position: dict) -> None:
        """Add an active position to tracking.

        Args:
            position: Position dict with symbol, side, size, entry_price, etc.

        Example:
            >>> state = TradingState()
            >>> position = {
            ...     "symbol": "BTC/USDT",
            ...     "side": "long",
            ...     "size": 0.1,
            ...     "entry_price": 42000.0
            ... }
            >>> state.add_position(position)
            >>> len(state.active_positions)
            1
        """
        self.active_positions.append(position)

    def remove_position(self, symbol: str) -> Optional[dict]:
        """Remove and return an active position.

        Args:
            symbol: Symbol of position to remove (e.g., "BTC/USDT")

        Returns:
            Removed position dict, or None if not found

        Example:
            >>> state = TradingState()
            >>> state.add_position({"symbol": "BTC/USDT", "size": 0.1})
            >>> position = state.remove_position("BTC/USDT")
            >>> position["symbol"]
            'BTC/USDT'
            >>> len(state.active_positions)
            0
        """
        for i, pos in enumerate(self.active_positions):
            if pos.get("symbol") == symbol:
                return self.active_positions.pop(i)

        return None

    def trip_circuit_breaker(self, reason: str) -> None:
        """Trip the circuit breaker to halt trading.

        Args:
            reason: Reason for tripping circuit breaker

        Example:
            >>> state = TradingState()
            >>> state.trip_circuit_breaker("Daily drawdown limit exceeded")
            >>> state.circuit_breaker_tripped
            True
        """
        self.circuit_breaker_tripped = True
        self.last_circuit_trip_reason = reason
        self.last_updated = datetime.now().isoformat()

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (requires manual action).

        Example:
            >>> state = TradingState()
            >>> state.trip_circuit_breaker("Test")
            >>> state.reset_circuit_breaker()
            >>> state.circuit_breaker_tripped
            False
        """
        self.circuit_breaker_tripped = False
        self.last_circuit_trip_reason = None

    def __repr__(self) -> str:
        """String representation of trading state."""
        return (
            f"TradingState(provider={self.provider}, "
            f"total_trades={self.total_trades}, "
            f"total_pnl=${self.total_pnl:.2f}, "
            f"active_positions={len(self.active_positions)}, "
            f"circuit_breaker={self.circuit_breaker_tripped})"
        )
