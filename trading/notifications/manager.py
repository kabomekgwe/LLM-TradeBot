"""Notification Manager - Unified interface for multi-channel alerts.

Coordinates notifications across Telegram, Discord, Email, and Slack.
Sends real-time alerts for trade execution, risk events, and system status.
"""

import logging
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from ..models.positions import Order
from ..config import TradingConfig


class NotificationLevel(str, Enum):
    """Notification priority level."""
    INFO = "info"  # General updates
    SUCCESS = "success"  # Successful trades
    WARNING = "warning"  # Risk warnings
    ERROR = "error"  # Errors, circuit breakers
    CRITICAL = "critical"  # Critical failures


class NotificationChannel(str, Enum):
    """Available notification channels."""
    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"
    SLACK = "slack"


@dataclass
class Notification:
    """Notification message."""

    title: str
    message: str
    level: NotificationLevel
    timestamp: datetime
    metadata: Dict[str, Any]

    def format_for_channel(self, channel: NotificationChannel) -> str:
        """Format notification for specific channel.

        Args:
            channel: Target notification channel

        Returns:
            Formatted message string
        """
        # Add emoji based on level
        emoji = {
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.SUCCESS: "✅",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🚨",
        }[self.level]

        # Basic format
        if channel == NotificationChannel.TELEGRAM:
            # Telegram supports markdown
            msg = f"{emoji} **{self.title}**\n\n{self.message}"
            if self.metadata:
                msg += f"\n\n📊 Details:\n"
                for key, value in self.metadata.items():
                    msg += f"• {key}: {value}\n"
            return msg

        elif channel == NotificationChannel.DISCORD:
            # Discord uses different markdown
            msg = f"{emoji} **{self.title}**\n{self.message}"
            if self.metadata:
                msg += f"\n\n**Details:**\n"
                for key, value in self.metadata.items():
                    msg += f"• {key}: `{value}`\n"
            return msg

        elif channel == NotificationChannel.EMAIL:
            # Email uses HTML
            msg = f"<h2>{emoji} {self.title}</h2>"
            msg += f"<p>{self.message}</p>"
            if self.metadata:
                msg += "<h3>Details:</h3><ul>"
                for key, value in self.metadata.items():
                    msg += f"<li><strong>{key}:</strong> {value}</li>"
                msg += "</ul>"
            return msg

        elif channel == NotificationChannel.SLACK:
            # Slack uses blocks
            msg = f"{emoji} *{self.title}*\n{self.message}"
            if self.metadata:
                msg += f"\n\n*Details:*\n"
                for key, value in self.metadata.items():
                    msg += f"• {key}: {value}\n"
            return msg

        return f"{emoji} {self.title}\n{self.message}"


class NotificationManager:
    """Unified notification manager for all channels.

    Sends alerts across multiple channels (Telegram, Discord, Email, Slack)
    with configurable filtering and rate limiting.

    Features:
    - Multi-channel broadcasting
    - Priority-based filtering
    - Rate limiting to avoid spam
    - Template-based messages
    - Async delivery

    Example:
        >>> manager = NotificationManager(config)
        >>> await manager.send_trade_executed(order, "BTC/USDT", 42000)
        >>> await manager.send_stop_loss_hit(order, 41000, -1000)
    """

    def __init__(self, config: TradingConfig):
        """Initialize notification manager.

        Args:
            config: Trading configuration with notification settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize channel handlers
        self.channels: Dict[NotificationChannel, Any] = {}

        # Only initialize enabled channels
        if self._is_telegram_enabled():
            from .telegram import TelegramNotifier
            self.channels[NotificationChannel.TELEGRAM] = TelegramNotifier(config)

        if self._is_discord_enabled():
            from .discord import DiscordNotifier
            self.channels[NotificationChannel.DISCORD] = DiscordNotifier(config)

        if self._is_email_enabled():
            from .email import EmailNotifier
            self.channels[NotificationChannel.EMAIL] = EmailNotifier(config)

        if self._is_slack_enabled():
            from .slack import SlackNotifier
            self.channels[NotificationChannel.SLACK] = SlackNotifier(config)

        self.logger.info(f"Notification manager initialized with {len(self.channels)} channels")

    def _is_telegram_enabled(self) -> bool:
        """Check if Telegram is configured."""
        return hasattr(self.config, 'telegram_bot_token') and bool(self.config.telegram_bot_token)

    def _is_discord_enabled(self) -> bool:
        """Check if Discord is configured."""
        return hasattr(self.config, 'discord_webhook') and bool(self.config.discord_webhook)

    def _is_email_enabled(self) -> bool:
        """Check if Email is configured."""
        return hasattr(self.config, 'smtp_host') and bool(self.config.smtp_host)

    def _is_slack_enabled(self) -> bool:
        """Check if Slack is configured."""
        return hasattr(self.config, 'slack_webhook') and bool(self.config.slack_webhook)

    async def send_notification(
        self,
        notification: Notification,
        channels: Optional[List[NotificationChannel]] = None,
    ):
        """Send notification to specified channels.

        Args:
            notification: Notification to send
            channels: Target channels (None = all enabled channels)
        """
        if channels is None:
            channels = list(self.channels.keys())

        # Send to each channel
        for channel in channels:
            if channel not in self.channels:
                continue

            try:
                notifier = self.channels[channel]
                formatted_message = notification.format_for_channel(channel)

                await notifier.send(formatted_message, notification.level)

                self.logger.debug(f"Sent notification to {channel.value}")

            except Exception as e:
                self.logger.error(f"Failed to send to {channel.value}: {e}")

    async def send_trade_executed(
        self,
        order: Order,
        symbol: str,
        price: float,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
    ):
        """Send trade execution notification.

        Args:
            order: Executed order
            symbol: Trading symbol
            price: Execution price
            stop_loss_price: Stop-loss price if set
            take_profit_price: Take-profit price if set
        """
        side_emoji = "📈" if order.side.value == "buy" else "📉"

        notification = Notification(
            title=f"{side_emoji} Trade Executed",
            message=f"{order.side.value.upper()} {order.amount:.4f} {symbol} @ ${price:,.2f}",
            level=NotificationLevel.SUCCESS,
            timestamp=datetime.now(),
            metadata={
                "Order ID": order.id,
                "Symbol": symbol,
                "Side": order.side.value.upper(),
                "Amount": f"{order.amount:.4f}",
                "Price": f"${price:,.2f}",
                "Stop Loss": f"${stop_loss_price:,.2f}" if stop_loss_price else "Not set",
                "Take Profit": f"${take_profit_price:,.2f}" if take_profit_price else "Not set",
                "Order Type": order.order_type.value,
            }
        )

        await self.send_notification(notification)

    async def send_stop_loss_hit(
        self,
        order: Order,
        stop_price: float,
        pnl: float,
    ):
        """Send stop-loss triggered notification.

        Args:
            order: Stop-loss order
            stop_price: Stop price that was hit
            pnl: Realized P&L
        """
        notification = Notification(
            title="🛑 Stop-Loss Hit",
            message=f"Position closed at ${stop_price:,.2f}\nP&L: ${pnl:+,.2f}",
            level=NotificationLevel.WARNING if pnl < 0 else NotificationLevel.INFO,
            timestamp=datetime.now(),
            metadata={
                "Order ID": order.id,
                "Symbol": order.symbol,
                "Stop Price": f"${stop_price:,.2f}",
                "P&L": f"${pnl:+,.2f}",
                "Amount": f"{order.amount:.4f}",
            }
        )

        await self.send_notification(notification)

    async def send_take_profit_hit(
        self,
        order: Order,
        take_profit_price: float,
        pnl: float,
    ):
        """Send take-profit triggered notification.

        Args:
            order: Take-profit order
            take_profit_price: Take-profit price that was hit
            pnl: Realized P&L
        """
        notification = Notification(
            title="🎯 Take-Profit Hit",
            message=f"Position closed at ${take_profit_price:,.2f}\nP&L: ${pnl:+,.2f}",
            level=NotificationLevel.SUCCESS,
            timestamp=datetime.now(),
            metadata={
                "Order ID": order.id,
                "Symbol": order.symbol,
                "Take Profit": f"${take_profit_price:,.2f}",
                "P&L": f"${pnl:+,.2f}",
                "Amount": f"{order.amount:.4f}",
            }
        )

        await self.send_notification(notification)

    async def send_circuit_breaker_triggered(
        self,
        reason: str,
        current_drawdown: float,
        max_drawdown: float,
    ):
        """Send circuit breaker alert.

        Args:
            reason: Reason for circuit breaker
            current_drawdown: Current drawdown percentage
            max_drawdown: Maximum allowed drawdown
        """
        notification = Notification(
            title="🚨 CIRCUIT BREAKER TRIGGERED",
            message=f"Trading halted: {reason}\n"
                   f"Current drawdown: {current_drawdown:.1f}%\n"
                   f"Max allowed: {max_drawdown:.1f}%",
            level=NotificationLevel.CRITICAL,
            timestamp=datetime.now(),
            metadata={
                "Reason": reason,
                "Current Drawdown": f"{current_drawdown:.1f}%",
                "Max Allowed": f"{max_drawdown:.1f}%",
                "Action": "All trading suspended",
            }
        )

        await self.send_notification(notification)

    async def send_daily_summary(
        self,
        total_trades: int,
        winning_trades: int,
        total_pnl: float,
        win_rate: float,
    ):
        """Send daily performance summary.

        Args:
            total_trades: Number of trades today
            winning_trades: Number of winning trades
            total_pnl: Total P&L
            win_rate: Win rate percentage
        """
        pnl_emoji = "📈" if total_pnl > 0 else "📉"

        notification = Notification(
            title=f"📊 Daily Summary - {datetime.now().strftime('%Y-%m-%d')}",
            message=f"{pnl_emoji} Today's Performance:\n"
                   f"P&L: ${total_pnl:+,.2f}\n"
                   f"Trades: {total_trades} ({winning_trades} wins)\n"
                   f"Win Rate: {win_rate:.1f}%",
            level=NotificationLevel.INFO,
            timestamp=datetime.now(),
            metadata={
                "Total Trades": total_trades,
                "Winning Trades": winning_trades,
                "Losing Trades": total_trades - winning_trades,
                "Win Rate": f"{win_rate:.1f}%",
                "Total P&L": f"${total_pnl:+,.2f}",
            }
        )

        await self.send_notification(notification)

    async def send_error(self, error_message: str, error_type: str = "Unknown"):
        """Send error notification.

        Args:
            error_message: Error description
            error_type: Type of error
        """
        notification = Notification(
            title=f"❌ Error: {error_type}",
            message=error_message,
            level=NotificationLevel.ERROR,
            timestamp=datetime.now(),
            metadata={
                "Error Type": error_type,
                "Timestamp": datetime.now().isoformat(),
            }
        )

        await self.send_notification(notification)

    async def send_startup(self, provider: str, symbol: str):
        """Send system startup notification.

        Args:
            provider: Exchange provider name
            symbol: Trading symbol
        """
        notification = Notification(
            title="🚀 Trading System Started",
            message=f"LLM-TradeBot is now live trading {symbol} on {provider}",
            level=NotificationLevel.INFO,
            timestamp=datetime.now(),
            metadata={
                "Provider": provider,
                "Symbol": symbol,
                "Start Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        await self.send_notification(notification)

    async def send_shutdown(self, reason: str = "Manual shutdown"):
        """Send system shutdown notification.

        Args:
            reason: Shutdown reason
        """
        notification = Notification(
            title="🛑 Trading System Stopped",
            message=f"LLM-TradeBot has stopped: {reason}",
            level=NotificationLevel.WARNING,
            timestamp=datetime.now(),
            metadata={
                "Reason": reason,
                "Stop Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        await self.send_notification(notification)
