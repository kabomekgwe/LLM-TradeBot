"""Safety Thresholds - Centralized configuration for safety limits.

Defines all threshold values for circuit breaker, position limits,
and safety triggers with validation logic.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SafetyThresholds:
    """Centralized threshold configuration for safety systems.

    All thresholds are production-ready conservative values designed to
    protect capital while allowing normal trading operations.
    """

    # Drawdown thresholds (%)
    max_daily_drawdown_pct: float = 5.0      # 5% daily drawdown limit
    max_weekly_drawdown_pct: float = 10.0    # 10% weekly drawdown limit
    max_total_drawdown_pct: float = 20.0     # 20% maximum drawdown limit

    # Loss thresholds
    max_consecutive_losses: int = 10         # Max losing trades in a row
    max_failed_trades_per_hour: int = 5      # Max failed trades per hour

    # API error thresholds
    max_api_errors_per_minute: int = 3       # Max API errors per minute
    max_order_failures: int = 5              # Max order placement failures

    # Position limit thresholds
    max_position_pct_per_symbol: float = 0.3   # 30% max per symbol
    max_position_pct_per_strategy: float = 0.6  # 60% max per strategy
    max_total_exposure_pct: float = 0.9        # 90% max total exposure
    max_concurrent_positions: int = 10         # Max simultaneous positions

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate threshold configuration for sanity.

        Returns:
            (is_valid, error_message)

        Validation rules:
        - Drawdown thresholds: daily < weekly < total
        - All percentages: 0 < value <= 1.0
        - All counts: positive integers
        """
        # Validate drawdown progression
        if not (self.max_daily_drawdown_pct < self.max_weekly_drawdown_pct < self.max_total_drawdown_pct):
            return False, "Drawdown thresholds must be: daily < weekly < total"

        # Validate drawdown ranges
        if not (0 < self.max_daily_drawdown_pct <= 100):
            return False, f"Daily drawdown {self.max_daily_drawdown_pct}% must be between 0-100%"

        if not (0 < self.max_weekly_drawdown_pct <= 100):
            return False, f"Weekly drawdown {self.max_weekly_drawdown_pct}% must be between 0-100%"

        if not (0 < self.max_total_drawdown_pct <= 100):
            return False, f"Total drawdown {self.max_total_drawdown_pct}% must be between 0-100%"

        # Validate loss thresholds
        if self.max_consecutive_losses <= 0:
            return False, f"Max consecutive losses {self.max_consecutive_losses} must be positive"

        if self.max_failed_trades_per_hour <= 0:
            return False, f"Max failed trades per hour {self.max_failed_trades_per_hour} must be positive"

        # Validate API error thresholds
        if self.max_api_errors_per_minute <= 0:
            return False, f"Max API errors per minute {self.max_api_errors_per_minute} must be positive"

        if self.max_order_failures <= 0:
            return False, f"Max order failures {self.max_order_failures} must be positive"

        # Validate position limit percentages
        if not (0 < self.max_position_pct_per_symbol <= 1.0):
            return False, f"Per-symbol limit {self.max_position_pct_per_symbol} must be between 0-1"

        if not (0 < self.max_position_pct_per_strategy <= 1.0):
            return False, f"Per-strategy limit {self.max_position_pct_per_strategy} must be between 0-1"

        if not (0 < self.max_total_exposure_pct <= 1.0):
            return False, f"Total exposure limit {self.max_total_exposure_pct} must be between 0-1"

        # Validate position limit progression (per-symbol < per-strategy < total)
        if not (self.max_position_pct_per_symbol <= self.max_position_pct_per_strategy <= self.max_total_exposure_pct):
            return False, "Position limits must be: per-symbol <= per-strategy <= total"

        # Validate max positions
        if self.max_concurrent_positions <= 0:
            return False, f"Max concurrent positions {self.max_concurrent_positions} must be positive"

        return True, None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "max_daily_drawdown_pct": self.max_daily_drawdown_pct,
            "max_weekly_drawdown_pct": self.max_weekly_drawdown_pct,
            "max_total_drawdown_pct": self.max_total_drawdown_pct,
            "max_consecutive_losses": self.max_consecutive_losses,
            "max_failed_trades_per_hour": self.max_failed_trades_per_hour,
            "max_api_errors_per_minute": self.max_api_errors_per_minute,
            "max_order_failures": self.max_order_failures,
            "max_position_pct_per_symbol": self.max_position_pct_per_symbol,
            "max_position_pct_per_strategy": self.max_position_pct_per_strategy,
            "max_total_exposure_pct": self.max_total_exposure_pct,
            "max_concurrent_positions": self.max_concurrent_positions,
        }
