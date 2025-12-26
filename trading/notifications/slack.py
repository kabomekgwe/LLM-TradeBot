"""Slack Notifier - Send alerts via Slack webhooks.

Sends trading notifications to Slack channels using webhooks.
Supports rich formatting with blocks and attachments.
"""

import logging
import aiohttp
from typing import Optional, List, Dict, Any

from ..config import TradingConfig
from .manager import NotificationLevel


class SlackNotifier:
    """Slack notification handler.

    Sends messages to Slack using Incoming Webhooks.

    Setup:
    1. Go to https://api.slack.com/apps
    2. Create New App → From scratch
    3. Incoming Webhooks → Activate → Add New Webhook
    4. Select channel and copy webhook URL
    5. Set SLACK_WEBHOOK in config

    Example:
        >>> notifier = SlackNotifier(config)
        >>> await notifier.send("Trade executed", NotificationLevel.SUCCESS)
    """

    def __init__(self, config: TradingConfig):
        """Initialize Slack notifier.

        Args:
            config: Trading configuration with Slack webhook
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.webhook_url = getattr(config, 'slack_webhook', '')

        if not self.webhook_url:
            self.logger.warning("Slack webhook not configured")

    async def send(self, message: str, level: NotificationLevel = NotificationLevel.INFO):
        """Send message to Slack.

        Args:
            message: Message text
            level: Notification priority level
        """
        if not self.webhook_url:
            self.logger.warning("Slack not configured, skipping notification")
            return

        # Map level to color
        colors = {
            NotificationLevel.INFO: "#36a64f",  # Green
            NotificationLevel.SUCCESS: "#2eb886",  # Bright green
            NotificationLevel.WARNING: "#ffa500",  # Orange
            NotificationLevel.ERROR: "#ff0000",  # Red
            NotificationLevel.CRITICAL: "#8b0000",  # Dark red
        }

        # Create payload with attachment
        payload = {
            "text": "LLM-TradeBot Notification",
            "attachments": [
                {
                    "color": colors.get(level, "#36a64f"),
                    "text": message,
                    "footer": "LLM-TradeBot",
                    "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png",
                    "ts": self._get_timestamp(),
                }
            ]
        }

        # Send request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        self.logger.debug("Slack notification sent")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Slack webhook error: {error_text}")

        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")

    async def send_blocks(
        self,
        text: str,
        blocks: List[Dict[str, Any]],
        level: NotificationLevel = NotificationLevel.INFO,
    ):
        """Send rich block message to Slack.

        Args:
            text: Fallback text
            blocks: Slack block elements
            level: Notification level
        """
        if not self.webhook_url:
            return

        payload = {
            "text": text,
            "blocks": blocks,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        self.logger.debug("Slack blocks sent")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Slack blocks error: {error_text}")

        except Exception as e:
            self.logger.error(f"Failed to send Slack blocks: {e}")

    async def send_trade_notification(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        pnl: Optional[float] = None,
    ):
        """Send formatted trade notification.

        Args:
            symbol: Trading symbol
            side: Trade side (buy/sell)
            amount: Trade amount
            price: Execution price
            pnl: P&L if closing position
        """
        if not self.webhook_url:
            return

        emoji = "📈" if side.lower() == "buy" else "📉"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Trade Executed",
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Symbol:*\n{symbol}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Side:*\n{side.upper()}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Amount:*\n{amount:.4f}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Price:*\n${price:,.2f}"
                    }
                ]
            }
        ]

        if pnl is not None:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*P&L:* ${pnl:+,.2f}"
                }
            })

        payload = {
            "text": f"Trade executed: {side.upper()} {amount:.4f} {symbol} @ ${price:,.2f}",
            "blocks": blocks,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        self.logger.debug("Slack trade notification sent")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Slack trade notification error: {error_text}")

        except Exception as e:
            self.logger.error(f"Failed to send Slack trade notification: {e}")

    def _get_timestamp(self) -> int:
        """Get current Unix timestamp."""
        from datetime import datetime
        return int(datetime.now().timestamp())
