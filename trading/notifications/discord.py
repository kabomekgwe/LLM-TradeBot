"""Discord Notifier - Send alerts via Discord webhooks.

Sends trading notifications to Discord channels using webhooks.
Supports rich embeds with colors and fields.
"""

import logging
import aiohttp
from typing import Optional, Dict, Any, List

from ..config import TradingConfig
from .manager import NotificationLevel


class DiscordNotifier:
    """Discord notification handler.

    Sends messages to Discord using webhooks.

    Setup:
    1. Go to Discord channel settings
    2. Integrations → Webhooks → New Webhook
    3. Copy webhook URL
    4. Set DISCORD_WEBHOOK in config

    Example:
        >>> notifier = DiscordNotifier(config)
        >>> await notifier.send("Trade executed", NotificationLevel.SUCCESS)
    """

    def __init__(self, config: TradingConfig):
        """Initialize Discord notifier.

        Args:
            config: Trading configuration with Discord webhook
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.webhook_url = getattr(config, 'discord_webhook', '')

        if not self.webhook_url:
            self.logger.warning("Discord webhook not configured")

    async def send(self, message: str, level: NotificationLevel = NotificationLevel.INFO):
        """Send message to Discord.

        Args:
            message: Message text
            level: Notification priority level
        """
        if not self.webhook_url:
            self.logger.warning("Discord not configured, skipping notification")
            return

        # Map level to color
        colors = {
            NotificationLevel.INFO: 0x3498db,  # Blue
            NotificationLevel.SUCCESS: 0x2ecc71,  # Green
            NotificationLevel.WARNING: 0xf39c12,  # Orange
            NotificationLevel.ERROR: 0xe74c3c,  # Red
            NotificationLevel.CRITICAL: 0x992d22,  # Dark red
        }

        # Create embed
        embed = {
            "description": message,
            "color": colors.get(level, 0x3498db),
            "timestamp": self._get_timestamp(),
        }

        payload = {
            "username": "LLM-TradeBot",
            "embeds": [embed],
        }

        # Send request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status in (200, 204):
                        self.logger.debug("Discord notification sent")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Discord webhook error: {error_text}")

        except Exception as e:
            self.logger.error(f"Failed to send Discord notification: {e}")

    async def send_embed(
        self,
        title: str,
        description: str,
        fields: Optional[List[Dict[str, Any]]] = None,
        level: NotificationLevel = NotificationLevel.INFO,
    ):
        """Send rich embed to Discord.

        Args:
            title: Embed title
            description: Embed description
            fields: List of field dicts with 'name' and 'value'
            level: Notification level
        """
        if not self.webhook_url:
            return

        colors = {
            NotificationLevel.INFO: 0x3498db,
            NotificationLevel.SUCCESS: 0x2ecc71,
            NotificationLevel.WARNING: 0xf39c12,
            NotificationLevel.ERROR: 0xe74c3c,
            NotificationLevel.CRITICAL: 0x992d22,
        }

        embed = {
            "title": title,
            "description": description,
            "color": colors.get(level, 0x3498db),
            "timestamp": self._get_timestamp(),
        }

        if fields:
            embed["fields"] = fields

        payload = {
            "username": "LLM-TradeBot",
            "embeds": [embed],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status in (200, 204):
                        self.logger.debug("Discord embed sent")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Discord embed error: {error_text}")

        except Exception as e:
            self.logger.error(f"Failed to send Discord embed: {e}")

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
