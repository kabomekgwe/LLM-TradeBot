"""Alert manager for coordinating triggers and notifications.

Coordinates alert trigger checking and notification delivery:
- Checks alert triggers after trades, safety events, health changes
- Sends notifications via NotificationManager to all enabled channels
- Logs all sent alerts with timestamp and trigger reason
- Supports alert configuration from environment (enable/disable channels)
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..notifications.manager import NotificationManager, Notification, NotificationLevel
from .alert_triggers import AlertTriggerChecker, AlertSeverity


@dataclass
class AlertRecord:
    """Record of sent alert for logging/tracking."""

    trigger_type: str
    severity: str
    message: str
    timestamp: str
    channels_sent: List[str]
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trigger_type": self.trigger_type,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp,
            "channels_sent": self.channels_sent,
            "success": self.success,
            "error": self.error,
        }


class AlertManager:
    """Alert coordination and delivery manager.

    Coordinates:
    - Trigger condition checking
    - Notification delivery to multiple channels
    - Alert logging and tracking
    - Channel enable/disable configuration

    Example:
        >>> alert_mgr = AlertManager(notification_manager, config)
        >>> await alert_mgr.check_and_send_alerts(metrics, health_status)
    """

    def __init__(
        self,
        notification_manager: Optional[NotificationManager] = None,
        config: Optional[Any] = None,
        debounce_seconds: int = 300,
    ):
        """Initialize alert manager.

        Args:
            notification_manager: NotificationManager instance for sending alerts
            config: Trading configuration with alert settings
            debounce_seconds: Cooldown period between alerts (default 300 = 5 minutes)
        """
        self.logger = logging.getLogger(__name__)

        # Notification manager
        self.notification_manager = notification_manager

        # Trigger checker
        self.trigger_checker = AlertTriggerChecker(debounce_seconds=debounce_seconds)

        # Alert history (for logging/debugging)
        self.alert_history: List[AlertRecord] = []
        self.max_history = 1000  # Keep last 1000 alerts

        # Configuration
        self.config = config
        self.alerts_enabled = self._check_alerts_enabled()

        self.logger.info(
            "AlertManager initialized",
            extra={
                "alerts_enabled": self.alerts_enabled,
                "notification_channels": len(notification_manager.channels) if notification_manager else 0,
            }
        )

    def _check_alerts_enabled(self) -> bool:
        """Check if alerts are enabled in config.

        Returns:
            True if alerts are enabled
        """
        if not self.config:
            return False

        # Check if notifications are enabled
        return getattr(self.config, "notifications_enabled", False)

    async def check_and_send_alerts(
        self,
        metrics: Optional[Dict[str, Any]] = None,
        health: Optional[Dict[str, Any]] = None,
    ) -> List[AlertRecord]:
        """Check triggers and send alerts if conditions met.

        Args:
            metrics: Current performance metrics
            health: Current health status

        Returns:
            List of AlertRecords for sent alerts
        """
        if not self.alerts_enabled:
            self.logger.debug("Alerts disabled, skipping check")
            return []

        if not self.notification_manager:
            self.logger.debug("No notification manager, skipping alerts")
            return []

        # Check triggers
        triggered_alerts = self.trigger_checker.check_triggers(metrics, health)

        if not triggered_alerts:
            return []

        # Send each triggered alert
        sent_alerts = []

        for alert_data in triggered_alerts:
            try:
                alert_record = await self._send_alert(alert_data)
                sent_alerts.append(alert_record)

                # Add to history
                self._add_to_history(alert_record)

            except Exception as e:
                self.logger.error(
                    "Failed to send alert",
                    extra={"error": str(e), "alert": alert_data},
                    exc_info=True
                )

        return sent_alerts

    async def _send_alert(self, alert_data: Dict[str, Any]) -> AlertRecord:
        """Send individual alert to notification channels.

        Args:
            alert_data: Alert data from trigger checker

        Returns:
            AlertRecord documenting the sent alert
        """
        trigger_type = alert_data.get("trigger_type", "unknown")
        severity = alert_data.get("severity", AlertSeverity.INFO.value)
        message = alert_data.get("message", "Alert triggered")
        timestamp = alert_data.get("timestamp", datetime.now().isoformat())

        # Map severity to notification level
        notification_level = self._map_severity_to_level(severity)

        # Build notification
        notification = Notification(
            title=f"Trading Alert: {trigger_type.replace('_', ' ').title()}",
            message=message,
            level=notification_level,
            timestamp=datetime.fromisoformat(timestamp),
            metadata={
                "trigger_type": trigger_type,
                "severity": severity,
                **{k: v for k, v in alert_data.items() if k not in ["trigger_type", "severity", "message", "timestamp"]}
            }
        )

        # Send to all enabled channels
        try:
            await self.notification_manager.send_notification(notification)

            # Track which channels were sent to
            channels_sent = list(self.notification_manager.channels.keys())
            channels_sent = [ch.value for ch in channels_sent]

            return AlertRecord(
                trigger_type=trigger_type,
                severity=severity,
                message=message,
                timestamp=timestamp,
                channels_sent=channels_sent,
                success=True,
            )

        except Exception as e:
            self.logger.error(
                "Notification send failed",
                extra={"error": str(e)},
                exc_info=True
            )

            return AlertRecord(
                trigger_type=trigger_type,
                severity=severity,
                message=message,
                timestamp=timestamp,
                channels_sent=[],
                success=False,
                error=str(e),
            )

    def _map_severity_to_level(self, severity: str) -> NotificationLevel:
        """Map alert severity to notification level.

        Args:
            severity: Alert severity

        Returns:
            NotificationLevel enum value
        """
        severity_map = {
            AlertSeverity.INFO.value: NotificationLevel.INFO,
            AlertSeverity.WARNING.value: NotificationLevel.WARNING,
            AlertSeverity.ERROR.value: NotificationLevel.ERROR,
            AlertSeverity.CRITICAL.value: NotificationLevel.CRITICAL,
        }

        return severity_map.get(severity, NotificationLevel.INFO)

    def _add_to_history(self, alert_record: AlertRecord):
        """Add alert to history log.

        Args:
            alert_record: Alert record to log
        """
        self.alert_history.append(alert_record)

        # Trim history if too large
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of alert dicts
        """
        recent = self.alert_history[-limit:] if len(self.alert_history) > limit else self.alert_history
        return [alert.to_dict() for alert in reversed(recent)]

    def get_trigger_config(self) -> Dict[str, Dict[str, Any]]:
        """Get current trigger configuration.

        Returns:
            Dict of trigger configurations
        """
        return self.trigger_checker.get_trigger_config()

    def set_trigger_enabled(self, trigger_type: str, enabled: bool):
        """Enable or disable a specific trigger.

        Args:
            trigger_type: Trigger type to modify
            enabled: Whether to enable
        """
        self.trigger_checker.set_trigger_enabled(trigger_type, enabled)

    def set_trigger_threshold(self, trigger_type: str, threshold: float):
        """Update trigger threshold.

        Args:
            trigger_type: Trigger type to modify
            threshold: New threshold value
        """
        self.trigger_checker.set_trigger_threshold(trigger_type, threshold)

    async def send_test_alert(self, channel: Optional[str] = None) -> Dict[str, Any]:
        """Send test alert to verify channel configuration.

        Args:
            channel: Specific channel to test (None = all channels)

        Returns:
            Dict with test results
        """
        if not self.notification_manager:
            return {
                "success": False,
                "error": "No notification manager configured",
            }

        try:
            # Create test notification
            notification = Notification(
                title="Test Alert - Trading Bot",
                message="This is a test alert to verify notification channel configuration.",
                level=NotificationLevel.INFO,
                timestamp=datetime.now(),
                metadata={
                    "test": True,
                    "source": "AlertManager",
                }
            )

            # Send to specified channel or all
            channels = None
            if channel:
                from ..notifications.manager import NotificationChannel
                try:
                    channels = [NotificationChannel(channel)]
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Invalid channel: {channel}",
                    }

            await self.notification_manager.send_notification(notification, channels)

            return {
                "success": True,
                "message": "Test alert sent successfully",
                "channels": [channel] if channel else ["all"],
            }

        except Exception as e:
            self.logger.error(
                "Test alert failed",
                extra={"error": str(e)},
                exc_info=True
            )

            return {
                "success": False,
                "error": str(e),
            }
