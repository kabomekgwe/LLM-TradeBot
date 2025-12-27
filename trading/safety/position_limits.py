"""Position Limit Enforcer - Multi-layer position size validation.

Enforces position limits at four levels:
1. Per-symbol: Max exposure to any single asset
2. Per-strategy: Max exposure to any single strategy
3. Portfolio-wide: Max total exposure across all positions
4. Max positions: Max number of concurrent positions
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .thresholds import SafetyThresholds


@dataclass
class Position:
    """Position information for limit tracking."""
    symbol: str
    strategy: str
    size_usd: float
    timestamp: float


class PositionLimitEnforcer:
    """Multi-layer position limit enforcement.

    Validates new positions against four independent limit layers.
    All four layers must pass for a position to be approved.

    Limit layers (all must pass):
    1. Per-symbol: No more than X% in any single symbol
    2. Per-strategy: No more than Y% in any single strategy
    3. Portfolio-wide: No more than Z% total exposure
    4. Max positions: No more than N concurrent positions

    Example:
        >>> thresholds = SafetyThresholds()
        >>> enforcer = PositionLimitEnforcer(thresholds)
        >>>
        >>> allowed, reason = enforcer.check_new_position(
        ...     symbol="BTC/USDT",
        ...     strategy="momentum",
        ...     position_size_usd=5000,
        ...     current_portfolio_value=10000
        ... )
        >>> if not allowed:
        ...     print(f"Position rejected: {reason}")
    """

    def __init__(self, thresholds: SafetyThresholds):
        """Initialize position limit enforcer.

        Args:
            thresholds: Safety threshold configuration
        """
        self.logger = logging.getLogger(__name__)
        self._thresholds = thresholds

        # Track positions by strategy
        self._positions_by_strategy: Dict[str, List[Position]] = {}

        # Track positions by symbol
        self._positions_by_symbol: Dict[str, List[Position]] = {}

        # All positions
        self._all_positions: List[Position] = []

    def check_new_position(
        self,
        symbol: str,
        strategy: str,
        position_size_usd: float,
        current_portfolio_value: float,
    ) -> Tuple[bool, Optional[str]]:
        """Check if new position violates any limits.

        All four limit layers are checked in order. First violation
        causes immediate rejection.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            strategy: Strategy name (e.g., "momentum", "mean_reversion")
            position_size_usd: Position size in USD
            current_portfolio_value: Current total portfolio value in USD

        Returns:
            (allowed: bool, rejection_reason: Optional[str])
            - If allowed=True, reason=None
            - If allowed=False, reason contains detailed rejection message
        """
        if current_portfolio_value <= 0:
            return False, "Invalid portfolio value (must be > 0)"

        if position_size_usd <= 0:
            return False, "Invalid position size (must be > 0)"

        # Layer 1: Per-symbol limit
        symbol_exposure_pct = position_size_usd / current_portfolio_value
        existing_symbol_exposure = self._get_symbol_exposure(symbol)
        total_symbol_exposure = (existing_symbol_exposure + position_size_usd) / current_portfolio_value

        if total_symbol_exposure > self._thresholds.max_position_pct_per_symbol:
            self.logger.warning(
                "position_rejected_symbol_limit",
                extra={
                    "symbol": symbol,
                    "new_position_usd": position_size_usd,
                    "existing_exposure_usd": existing_symbol_exposure,
                    "total_exposure_pct": total_symbol_exposure * 100,
                    "limit_pct": self._thresholds.max_position_pct_per_symbol * 100,
                }
            )
            return False, (
                f"Per-symbol limit exceeded for {symbol}: "
                f"{total_symbol_exposure:.1%} > {self._thresholds.max_position_pct_per_symbol:.1%} "
                f"(existing: ${existing_symbol_exposure:,.2f}, new: ${position_size_usd:,.2f})"
            )

        # Layer 2: Per-strategy limit
        strategy_total = self._get_strategy_exposure(strategy)
        strategy_exposure_pct = (strategy_total + position_size_usd) / current_portfolio_value

        if strategy_exposure_pct > self._thresholds.max_position_pct_per_strategy:
            self.logger.warning(
                "position_rejected_strategy_limit",
                extra={
                    "strategy": strategy,
                    "new_position_usd": position_size_usd,
                    "existing_exposure_usd": strategy_total,
                    "total_exposure_pct": strategy_exposure_pct * 100,
                    "limit_pct": self._thresholds.max_position_pct_per_strategy * 100,
                }
            )
            return False, (
                f"Per-strategy limit exceeded for {strategy}: "
                f"{strategy_exposure_pct:.1%} > {self._thresholds.max_position_pct_per_strategy:.1%} "
                f"(existing: ${strategy_total:,.2f}, new: ${position_size_usd:,.2f})"
            )

        # Layer 3: Portfolio-wide limit
        total_exposure = self._get_total_exposure()
        total_exposure_pct = (total_exposure + position_size_usd) / current_portfolio_value

        if total_exposure_pct > self._thresholds.max_total_exposure_pct:
            self.logger.warning(
                "position_rejected_total_exposure_limit",
                extra={
                    "new_position_usd": position_size_usd,
                    "existing_exposure_usd": total_exposure,
                    "total_exposure_pct": total_exposure_pct * 100,
                    "limit_pct": self._thresholds.max_total_exposure_pct * 100,
                }
            )
            return False, (
                f"Total exposure limit exceeded: "
                f"{total_exposure_pct:.1%} > {self._thresholds.max_total_exposure_pct:.1%} "
                f"(existing: ${total_exposure:,.2f}, new: ${position_size_usd:,.2f})"
            )

        # Layer 4: Max concurrent positions
        current_position_count = self._get_position_count()

        if current_position_count >= self._thresholds.max_concurrent_positions:
            self.logger.warning(
                "position_rejected_max_positions",
                extra={
                    "current_positions": current_position_count,
                    "limit": self._thresholds.max_concurrent_positions,
                }
            )
            return False, (
                f"Max concurrent positions limit exceeded: "
                f"{current_position_count} >= {self._thresholds.max_concurrent_positions}"
            )

        # All checks passed
        self.logger.info(
            "position_approved",
            extra={
                "symbol": symbol,
                "strategy": strategy,
                "position_size_usd": position_size_usd,
                "symbol_exposure_pct": total_symbol_exposure * 100,
                "strategy_exposure_pct": strategy_exposure_pct * 100,
                "total_exposure_pct": total_exposure_pct * 100,
                "position_count": current_position_count + 1,
            }
        )

        return True, None

    def add_position(
        self,
        symbol: str,
        strategy: str,
        position_size_usd: float,
        timestamp: float,
    ):
        """Add position to tracking after it has been approved and executed.

        Args:
            symbol: Trading symbol
            strategy: Strategy name
            position_size_usd: Position size in USD
            timestamp: Position open timestamp
        """
        position = Position(
            symbol=symbol,
            strategy=strategy,
            size_usd=position_size_usd,
            timestamp=timestamp,
        )

        # Add to all positions
        self._all_positions.append(position)

        # Add to strategy tracking
        if strategy not in self._positions_by_strategy:
            self._positions_by_strategy[strategy] = []
        self._positions_by_strategy[strategy].append(position)

        # Add to symbol tracking
        if symbol not in self._positions_by_symbol:
            self._positions_by_symbol[symbol] = []
        self._positions_by_symbol[symbol].append(position)

        self.logger.info(
            "position_added",
            extra={
                "symbol": symbol,
                "strategy": strategy,
                "size_usd": position_size_usd,
                "total_positions": len(self._all_positions),
            }
        )

    def remove_position(self, symbol: str, strategy: str, position_size_usd: float):
        """Remove position from tracking when closed.

        Args:
            symbol: Trading symbol
            strategy: Strategy name
            position_size_usd: Position size in USD (for matching)
        """
        # Remove from all positions
        self._all_positions = [
            p for p in self._all_positions
            if not (p.symbol == symbol and p.strategy == strategy and abs(p.size_usd - position_size_usd) < 0.01)
        ]

        # Remove from strategy tracking
        if strategy in self._positions_by_strategy:
            self._positions_by_strategy[strategy] = [
                p for p in self._positions_by_strategy[strategy]
                if not (p.symbol == symbol and abs(p.size_usd - position_size_usd) < 0.01)
            ]
            if not self._positions_by_strategy[strategy]:
                del self._positions_by_strategy[strategy]

        # Remove from symbol tracking
        if symbol in self._positions_by_symbol:
            self._positions_by_symbol[symbol] = [
                p for p in self._positions_by_symbol[symbol]
                if not (p.strategy == strategy and abs(p.size_usd - position_size_usd) < 0.01)
            ]
            if not self._positions_by_symbol[symbol]:
                del self._positions_by_symbol[symbol]

        self.logger.info(
            "position_removed",
            extra={
                "symbol": symbol,
                "strategy": strategy,
                "size_usd": position_size_usd,
                "remaining_positions": len(self._all_positions),
            }
        )

    def _get_strategy_exposure(self, strategy: str) -> float:
        """Calculate total USD exposure for a strategy.

        Args:
            strategy: Strategy name

        Returns:
            Total exposure in USD
        """
        if strategy not in self._positions_by_strategy:
            return 0.0

        return sum(p.size_usd for p in self._positions_by_strategy[strategy])

    def _get_symbol_exposure(self, symbol: str) -> float:
        """Calculate total USD exposure for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Total exposure in USD
        """
        if symbol not in self._positions_by_symbol:
            return 0.0

        return sum(p.size_usd for p in self._positions_by_symbol[symbol])

    def _get_total_exposure(self) -> float:
        """Calculate total USD exposure across all positions.

        Returns:
            Total exposure in USD
        """
        return sum(p.size_usd for p in self._all_positions)

    def _get_position_count(self) -> int:
        """Count total open positions.

        Returns:
            Number of open positions
        """
        return len(self._all_positions)

    def get_status(self) -> dict:
        """Get current position limit status.

        Returns:
            Dictionary with exposure information
        """
        strategy_exposures = {
            strategy: sum(p.size_usd for p in positions)
            for strategy, positions in self._positions_by_strategy.items()
        }

        symbol_exposures = {
            symbol: sum(p.size_usd for p in positions)
            for symbol, positions in self._positions_by_symbol.items()
        }

        return {
            "total_positions": len(self._all_positions),
            "total_exposure_usd": self._get_total_exposure(),
            "strategy_exposures": strategy_exposures,
            "symbol_exposures": symbol_exposures,
            "limits": {
                "max_position_pct_per_symbol": self._thresholds.max_position_pct_per_symbol,
                "max_position_pct_per_strategy": self._thresholds.max_position_pct_per_strategy,
                "max_total_exposure_pct": self._thresholds.max_total_exposure_pct,
                "max_concurrent_positions": self._thresholds.max_concurrent_positions,
            }
        }
