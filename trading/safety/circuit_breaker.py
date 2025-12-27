"""Circuit Breaker - Automatic trading pause on threshold breaches.

Monitors risk metrics and automatically halts trading when safety
thresholds are exceeded. Graduated response to prevent catastrophic losses.
"""

import logging
from enum import Enum
from typing import List, Optional
from datetime import datetime, timedelta
from collections import deque

from .thresholds import SafetyThresholds
from ..analytics.risk_calculator import RiskCalculator
from ..memory.trade_history import TradeRecord


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation - trading enabled
    OPEN = "open"      # Circuit tripped - trading paused


class CircuitBreaker:
    """Automatic circuit breaker for threshold-based trading pause.

    Monitors multiple risk dimensions and automatically opens the circuit
    (pauses trading) when any threshold is breached.

    Threshold categories:
    - Drawdown: Daily, weekly, total drawdown limits
    - Loss streaks: Consecutive losing trades
    - API errors: Error rate and order failures
    - Failed trades: Recent failed trade count

    Example:
        >>> thresholds = SafetyThresholds()
        >>> risk_calculator = RiskCalculator()
        >>> circuit_breaker = CircuitBreaker(thresholds, risk_calculator)
        >>>
        >>> # Check before each trade
        >>> if circuit_breaker.check_and_update(trades, current_equity):
        ...     print("Circuit tripped - trading paused!")
    """

    def __init__(self, thresholds: SafetyThresholds, risk_calculator: RiskCalculator):
        """Initialize circuit breaker.

        Args:
            thresholds: Safety threshold configuration
            risk_calculator: Risk calculator for drawdown metrics
        """
        self.logger = logging.getLogger(__name__)

        self._state = CircuitState.CLOSED
        self._thresholds = thresholds
        self._risk_calculator = risk_calculator

        # Counters for threshold tracking
        self._consecutive_losses = 0
        self._api_errors_last_minute: deque = deque(maxlen=100)
        self._order_failures = 0
        self._failed_trades_last_hour: deque = deque(maxlen=100)

        # Trip tracking
        self._tripped_at: Optional[datetime] = None
        self._trip_reason: Optional[str] = None

        self.logger.info("Circuit breaker initialized", extra={"state": self._state.value})

    def check_and_update(
        self,
        trades: List[TradeRecord],
        current_equity: float,
    ) -> bool:
        """Check all thresholds and trip circuit if any breached.

        This method should be called BEFORE placing any order to verify
        that trading is still safe to continue.

        Args:
            trades: Historical trades for analysis
            current_equity: Current account equity

        Returns:
            True if circuit tripped (trading should be paused)
            False if all checks passed (safe to trade)
        """
        # If already open, return True (still tripped)
        if self._state == CircuitState.OPEN:
            return True

        # Check all thresholds - first breach wins

        # 1. Check drawdown thresholds
        if trades:
            risk_metrics = self._risk_calculator.calculate_risk_metrics(trades, current_equity)

            # Check total drawdown
            if risk_metrics.max_drawdown_pct > self._thresholds.max_total_drawdown_pct:
                self.trip(f"Total drawdown {risk_metrics.max_drawdown_pct:.1f}% exceeded limit {self._thresholds.max_total_drawdown_pct:.1f}%")
                return True

            # Check daily drawdown (simplified - using recent trades)
            recent_trades_1d = self._get_recent_trades(trades, hours=24)
            if recent_trades_1d:
                daily_metrics = self._risk_calculator.calculate_risk_metrics(recent_trades_1d, current_equity)
                if daily_metrics.max_drawdown_pct > self._thresholds.max_daily_drawdown_pct:
                    self.trip(f"Daily drawdown {daily_metrics.max_drawdown_pct:.1f}% exceeded limit {self._thresholds.max_daily_drawdown_pct:.1f}%")
                    return True

            # Check weekly drawdown
            recent_trades_7d = self._get_recent_trades(trades, hours=24*7)
            if recent_trades_7d:
                weekly_metrics = self._risk_calculator.calculate_risk_metrics(recent_trades_7d, current_equity)
                if weekly_metrics.max_drawdown_pct > self._thresholds.max_weekly_drawdown_pct:
                    self.trip(f"Weekly drawdown {weekly_metrics.max_drawdown_pct:.1f}% exceeded limit {self._thresholds.max_weekly_drawdown_pct:.1f}%")
                    return True

        # 2. Check consecutive losses
        if trades:
            recent_closed = [t for t in trades if t.closed][-self._thresholds.max_consecutive_losses:]
            if len(recent_closed) >= self._thresholds.max_consecutive_losses:
                # Check if ALL recent trades are losses
                if all(not t.won for t in recent_closed):
                    self._consecutive_losses = len(recent_closed)
                    self.trip(f"Consecutive losses {self._consecutive_losses} exceeded limit {self._thresholds.max_consecutive_losses}")
                    return True

        # 3. Check API error rate (sliding window)
        now = datetime.now()
        # Remove errors older than 1 minute
        while self._api_errors_last_minute and (now - self._api_errors_last_minute[0]) > timedelta(minutes=1):
            self._api_errors_last_minute.popleft()

        if len(self._api_errors_last_minute) >= self._thresholds.max_api_errors_per_minute:
            self.trip(f"API errors {len(self._api_errors_last_minute)} per minute exceeded limit {self._thresholds.max_api_errors_per_minute}")
            return True

        # 4. Check order failures
        if self._order_failures >= self._thresholds.max_order_failures:
            self.trip(f"Order failures {self._order_failures} exceeded limit {self._thresholds.max_order_failures}")
            return True

        # 5. Check failed trades per hour
        # Remove failed trades older than 1 hour
        while self._failed_trades_last_hour and (now - self._failed_trades_last_hour[0]) > timedelta(hours=1):
            self._failed_trades_last_hour.popleft()

        if len(self._failed_trades_last_hour) >= self._thresholds.max_failed_trades_per_hour:
            self.trip(f"Failed trades {len(self._failed_trades_last_hour)} per hour exceeded limit {self._thresholds.max_failed_trades_per_hour}")
            return True

        # All checks passed
        return False

    def trip(self, reason: str):
        """Open circuit (pause trading).

        Args:
            reason: Reason for circuit trip
        """
        if self._state == CircuitState.OPEN:
            # Already tripped
            return

        self._state = CircuitState.OPEN
        self._tripped_at = datetime.now()
        self._trip_reason = reason

        self.logger.critical(
            "CIRCUIT_BREAKER_TRIPPED",
            extra={
                "state": self._state.value,
                "reason": reason,
                "tripped_at": self._tripped_at.isoformat(),
            }
        )

    def reset(self):
        """Close circuit (resume trading) - manual only.

        This is a manual operation that requires operator intervention.
        Circuit breaker does NOT automatically reset.
        """
        if self._state == CircuitState.CLOSED:
            self.logger.warning("circuit_breaker_already_closed")
            return

        self.logger.warning(
            "CIRCUIT_BREAKER_RESET",
            extra={
                "previous_state": self._state.value,
                "was_tripped_at": self._tripped_at.isoformat() if self._tripped_at else None,
                "reason": self._trip_reason,
                "reset_at": datetime.now().isoformat(),
            }
        )

        self._state = CircuitState.CLOSED
        # Reset counters
        self._consecutive_losses = 0
        self._api_errors_last_minute.clear()
        self._order_failures = 0
        self._failed_trades_last_hour.clear()

    def is_open(self) -> bool:
        """Check if circuit is open (trading paused).

        Returns:
            True if circuit is open (trading should be blocked)
        """
        return self._state == CircuitState.OPEN

    def record_api_error(self):
        """Record an API error for rate limiting.

        Call this whenever an API call fails.
        """
        self._api_errors_last_minute.append(datetime.now())
        self.logger.warning(
            "api_error_recorded",
            extra={"errors_last_minute": len(self._api_errors_last_minute)}
        )

    def record_order_failure(self):
        """Record an order placement failure.

        Call this whenever an order fails to execute.
        """
        self._order_failures += 1
        self.logger.warning(
            "order_failure_recorded",
            extra={"total_failures": self._order_failures}
        )

    def record_failed_trade(self):
        """Record a failed trade (execution error, rejection, etc.).

        Call this when a trade attempt fails for any reason.
        """
        self._failed_trades_last_hour.append(datetime.now())
        self.logger.warning(
            "failed_trade_recorded",
            extra={"failures_last_hour": len(self._failed_trades_last_hour)}
        )

    def get_status(self) -> dict:
        """Get current circuit breaker status.

        Returns:
            Dictionary with status information
        """
        return {
            "state": self._state.value,
            "is_open": self.is_open(),
            "tripped_at": self._tripped_at.isoformat() if self._tripped_at else None,
            "trip_reason": self._trip_reason,
            "consecutive_losses": self._consecutive_losses,
            "api_errors_last_minute": len(self._api_errors_last_minute),
            "order_failures": self._order_failures,
            "failed_trades_last_hour": len(self._failed_trades_last_hour),
        }

    def _get_recent_trades(self, trades: List[TradeRecord], hours: int) -> List[TradeRecord]:
        """Get trades from the last N hours.

        Args:
            trades: All trades
            hours: Number of hours to look back

        Returns:
            List of recent trades
        """
        if not trades:
            return []

        cutoff_timestamp = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
        return [t for t in trades if t.timestamp >= cutoff_timestamp]
