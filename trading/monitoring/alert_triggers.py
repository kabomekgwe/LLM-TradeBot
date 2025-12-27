"""Alert trigger configuration and checking logic.

Defines alert triggers for critical trading events:
- Drawdown exceeds daily/weekly/total limits
- Circuit breaker trips
- Kill switch activation
- Consecutive losses exceed limit
- API errors exceed rate

Includes debouncing logic to prevent alert spam (max 1 alert per trigger per 5 minutes).
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum


class TriggerType(str, Enum):
    """Alert trigger types."""

    # Drawdown triggers
    DAILY_DRAWDOWN_LIMIT = "daily_drawdown_limit"
    WEEKLY_DRAWDOWN_LIMIT = "weekly_drawdown_limit"
    TOTAL_DRAWDOWN_LIMIT = "total_drawdown_limit"

    # Safety control triggers
    CIRCUIT_BREAKER_TRIP = "circuit_breaker_trip"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"

    # Trading pattern triggers
    CONSECUTIVE_LOSSES = "consecutive_losses"
    LOW_WIN_RATE = "low_win_rate"

    # System triggers
    API_ERROR_RATE = "api_error_rate"
    API_DISCONNECTED = "api_disconnected"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertTrigger:
    """Alert trigger definition."""

    trigger_type: str
    severity: str
    threshold: float
    description: str
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertTrigger":
        """Create from dictionary."""
        return cls(**data)


class AlertTriggerChecker:
    """Alert trigger condition checker with debouncing.

    Checks current system state against configured triggers
    and determines which alerts should fire.

    Features:
    - Configurable trigger thresholds
    - Debouncing (max 1 alert per trigger per 5 minutes)
    - Multiple severity levels
    - Enable/disable individual triggers

    Example:
        >>> checker = AlertTriggerChecker()
        >>> triggers = checker.check_triggers(metrics, health_status)
        >>> for trigger in triggers:
        ...     print(f"Alert: {trigger['message']}")
    """

    def __init__(self, debounce_seconds: int = 300):
        """Initialize trigger checker.

        Args:
            debounce_seconds: Cooldown period between alerts (default 300 = 5 minutes)
        """
        self.logger = logging.getLogger(__name__)
        self.debounce_seconds = debounce_seconds

        # Last alert time per trigger type
        self.last_alert_time: Dict[str, datetime] = {}

        # Configure default triggers
        self.triggers = self._create_default_triggers()

        self.logger.info(
            "AlertTriggerChecker initialized",
            extra={"debounce_seconds": debounce_seconds, "triggers": len(self.triggers)}
        )

    def _create_default_triggers(self) -> Dict[str, AlertTrigger]:
        """Create default alert triggers.

        Returns:
            Dict mapping trigger type to AlertTrigger
        """
        return {
            TriggerType.DAILY_DRAWDOWN_LIMIT.value: AlertTrigger(
                trigger_type=TriggerType.DAILY_DRAWDOWN_LIMIT.value,
                severity=AlertSeverity.WARNING.value,
                threshold=5.0,  # 5% daily drawdown
                description="Daily drawdown exceeds 5%",
                enabled=True,
            ),
            TriggerType.WEEKLY_DRAWDOWN_LIMIT.value: AlertTrigger(
                trigger_type=TriggerType.WEEKLY_DRAWDOWN_LIMIT.value,
                severity=AlertSeverity.ERROR.value,
                threshold=10.0,  # 10% weekly drawdown
                description="Weekly drawdown exceeds 10%",
                enabled=True,
            ),
            TriggerType.TOTAL_DRAWDOWN_LIMIT.value: AlertTrigger(
                trigger_type=TriggerType.TOTAL_DRAWDOWN_LIMIT.value,
                severity=AlertSeverity.CRITICAL.value,
                threshold=20.0,  # 20% total drawdown
                description="Total drawdown exceeds 20%",
                enabled=True,
            ),
            TriggerType.CIRCUIT_BREAKER_TRIP.value: AlertTrigger(
                trigger_type=TriggerType.CIRCUIT_BREAKER_TRIP.value,
                severity=AlertSeverity.CRITICAL.value,
                threshold=0.0,  # Any circuit breaker trip
                description="Circuit breaker has tripped",
                enabled=True,
            ),
            TriggerType.KILL_SWITCH_ACTIVATED.value: AlertTrigger(
                trigger_type=TriggerType.KILL_SWITCH_ACTIVATED.value,
                severity=AlertSeverity.CRITICAL.value,
                threshold=0.0,  # Any kill switch activation
                description="Kill switch has been activated",
                enabled=True,
            ),
            TriggerType.CONSECUTIVE_LOSSES.value: AlertTrigger(
                trigger_type=TriggerType.CONSECUTIVE_LOSSES.value,
                severity=AlertSeverity.WARNING.value,
                threshold=10.0,  # 10 consecutive losses
                description="10 consecutive losing trades",
                enabled=True,
            ),
            TriggerType.LOW_WIN_RATE.value: AlertTrigger(
                trigger_type=TriggerType.LOW_WIN_RATE.value,
                severity=AlertSeverity.WARNING.value,
                threshold=30.0,  # Win rate below 30%
                description="Win rate dropped below 30%",
                enabled=True,
            ),
            TriggerType.API_DISCONNECTED.value: AlertTrigger(
                trigger_type=TriggerType.API_DISCONNECTED.value,
                severity=AlertSeverity.ERROR.value,
                threshold=0.0,  # Any disconnection
                description="API connection lost",
                enabled=True,
            ),
        }

    def check_triggers(
        self,
        metrics: Optional[Dict[str, Any]] = None,
        health: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Check all triggers against current state.

        Args:
            metrics: Current performance metrics
            health: Current health status

        Returns:
            List of triggered alerts (after debouncing)
        """
        triggered_alerts = []

        if metrics is None:
            metrics = {}
        if health is None:
            health = {}

        # Check each trigger
        for trigger_type, trigger in self.triggers.items():
            if not trigger.enabled:
                continue

            # Check if trigger condition is met
            alert_data = self._check_trigger(trigger, metrics, health)

            if alert_data:
                # Check debouncing
                if self._should_send_alert(trigger_type):
                    # Record alert time
                    self.last_alert_time[trigger_type] = datetime.now()

                    # Add trigger metadata
                    alert_data["trigger_type"] = trigger_type
                    alert_data["severity"] = trigger.severity
                    alert_data["timestamp"] = datetime.now().isoformat()

                    triggered_alerts.append(alert_data)

                    self.logger.info(
                        "Alert triggered",
                        extra={
                            "trigger": trigger_type,
                            "severity": trigger.severity,
                        }
                    )

        return triggered_alerts

    def _check_trigger(
        self,
        trigger: AlertTrigger,
        metrics: Dict[str, Any],
        health: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Check individual trigger condition.

        Args:
            trigger: Trigger definition
            metrics: Current metrics
            health: Current health status

        Returns:
            Alert data dict if triggered, None otherwise
        """
        trigger_type = trigger.trigger_type

        # Daily drawdown check
        if trigger_type == TriggerType.DAILY_DRAWDOWN_LIMIT.value:
            daily_pnl = metrics.get("daily_pnl", 0.0)
            current_equity = metrics.get("current_equity", 10000.0)
            daily_drawdown = abs(daily_pnl / current_equity * 100) if daily_pnl < 0 else 0.0

            if daily_drawdown > trigger.threshold:
                return {
                    "message": f"Daily drawdown {daily_drawdown:.2f}% exceeds limit {trigger.threshold:.2f}%",
                    "value": daily_drawdown,
                    "threshold": trigger.threshold,
                }

        # Weekly drawdown check (using current_drawdown as proxy)
        elif trigger_type == TriggerType.WEEKLY_DRAWDOWN_LIMIT.value:
            weekly_pnl = metrics.get("weekly_pnl", 0.0)
            current_equity = metrics.get("current_equity", 10000.0)
            weekly_drawdown = abs(weekly_pnl / current_equity * 100) if weekly_pnl < 0 else 0.0

            if weekly_drawdown > trigger.threshold:
                return {
                    "message": f"Weekly drawdown {weekly_drawdown:.2f}% exceeds limit {trigger.threshold:.2f}%",
                    "value": weekly_drawdown,
                    "threshold": trigger.threshold,
                }

        # Total drawdown check
        elif trigger_type == TriggerType.TOTAL_DRAWDOWN_LIMIT.value:
            current_drawdown = metrics.get("current_drawdown", 0.0)

            if current_drawdown > trigger.threshold:
                return {
                    "message": f"Total drawdown {current_drawdown:.2f}% exceeds limit {trigger.threshold:.2f}%",
                    "value": current_drawdown,
                    "threshold": trigger.threshold,
                }

        # Circuit breaker trip
        elif trigger_type == TriggerType.CIRCUIT_BREAKER_TRIP.value:
            circuit_breaker_open = health.get("circuit_breaker_open", False)
            circuit_breaker_reason = health.get("circuit_breaker_reason")

            if circuit_breaker_open:
                return {
                    "message": f"Circuit breaker tripped: {circuit_breaker_reason or 'Unknown reason'}",
                    "value": 1,
                    "threshold": 0,
                    "reason": circuit_breaker_reason,
                }

        # Kill switch activation
        elif trigger_type == TriggerType.KILL_SWITCH_ACTIVATED.value:
            kill_switch_active = health.get("kill_switch_active", False)
            kill_switch_reason = health.get("kill_switch_reason")

            if kill_switch_active:
                return {
                    "message": f"Kill switch activated: {kill_switch_reason or 'Unknown reason'}",
                    "value": 1,
                    "threshold": 0,
                    "reason": kill_switch_reason,
                }

        # Consecutive losses
        elif trigger_type == TriggerType.CONSECUTIVE_LOSSES.value:
            consecutive_losses = metrics.get("consecutive_losses", 0)

            if consecutive_losses >= trigger.threshold:
                return {
                    "message": f"{consecutive_losses} consecutive losses (limit: {int(trigger.threshold)})",
                    "value": consecutive_losses,
                    "threshold": trigger.threshold,
                }

        # Low win rate
        elif trigger_type == TriggerType.LOW_WIN_RATE.value:
            win_rate = metrics.get("win_rate", 100.0)
            total_trades = metrics.get("total_trades", 0)

            # Only check if we have enough trades (at least 10)
            if total_trades >= 10 and win_rate < trigger.threshold:
                return {
                    "message": f"Win rate {win_rate:.1f}% below threshold {trigger.threshold:.1f}%",
                    "value": win_rate,
                    "threshold": trigger.threshold,
                }

        # API disconnected
        elif trigger_type == TriggerType.API_DISCONNECTED.value:
            api_status = health.get("api_status", "connected")

            if api_status == "disconnected":
                return {
                    "message": "API connection lost",
                    "value": 1,
                    "threshold": 0,
                }

        return None

    def _should_send_alert(self, trigger_type: str) -> bool:
        """Check if alert should be sent (debouncing).

        Args:
            trigger_type: Type of trigger

        Returns:
            True if alert should be sent, False if still in debounce period
        """
        if trigger_type not in self.last_alert_time:
            return True

        elapsed = datetime.now() - self.last_alert_time[trigger_type]
        return elapsed.total_seconds() > self.debounce_seconds

    def set_trigger_enabled(self, trigger_type: str, enabled: bool):
        """Enable or disable a specific trigger.

        Args:
            trigger_type: Type of trigger to modify
            enabled: Whether to enable the trigger
        """
        if trigger_type in self.triggers:
            self.triggers[trigger_type].enabled = enabled
            self.logger.info(
                f"Trigger {'enabled' if enabled else 'disabled'}",
                extra={"trigger": trigger_type}
            )

    def set_trigger_threshold(self, trigger_type: str, threshold: float):
        """Update trigger threshold.

        Args:
            trigger_type: Type of trigger to modify
            threshold: New threshold value
        """
        if trigger_type in self.triggers:
            self.triggers[trigger_type].threshold = threshold
            self.logger.info(
                "Trigger threshold updated",
                extra={"trigger": trigger_type, "threshold": threshold}
            )

    def get_trigger_config(self) -> Dict[str, Dict[str, Any]]:
        """Get current trigger configuration.

        Returns:
            Dict mapping trigger types to trigger config
        """
        return {
            trigger_type: trigger.to_dict()
            for trigger_type, trigger in self.triggers.items()
        }
