"""Telegram Notifier - Send alerts via Telegram bot.

Sends trading notifications to Telegram using the Bot API.
Supports markdown formatting and inline buttons.
"""

import logging
import aiohttp
from typing import Optional

from ..config import TradingConfig
from .manager import NotificationLevel


class TelegramNotifier:
    """Telegram notification handler.

    Sends messages to Telegram using Bot API.

    Setup:
    1. Create bot via @BotFather on Telegram
    2. Get bot token
    3. Start conversation with bot
    4. Get your chat_id (use @userinfobot)
    5. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in config

    Example:
        >>> notifier = TelegramNotifier(config)
        >>> await notifier.send("Trade executed: BTC/USDT @ $42000", NotificationLevel.SUCCESS)
    """

    def __init__(self, config: TradingConfig):
        """Initialize Telegram notifier.

        Args:
            config: Trading configuration with Telegram credentials
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Get credentials from config
        self.bot_token = getattr(config, 'telegram_bot_token', '')
        self.chat_id = getattr(config, 'telegram_chat_id', '')

        if not self.bot_token or not self.chat_id:
            self.logger.warning("Telegram not configured (missing bot_token or chat_id)")

        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    async def send(self, message: str, level: NotificationLevel = NotificationLevel.INFO):
        """Send message to Telegram.

        Args:
            message: Message text (supports markdown)
            level: Notification priority level
        """
        if not self.bot_token or not self.chat_id:
            self.logger.warning("Telegram not configured, skipping notification")
            return

        # Prepare request
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",  # Enable markdown formatting
            "disable_web_page_preview": True,
        }

        # Send request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.logger.debug("Telegram notification sent")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Telegram API error: {error_text}")

        except Exception as e:
            self.logger.error(f"Failed to send Telegram notification: {e}")

    async def send_photo(
        self,
        photo_url: str,
        caption: Optional[str] = None,
    ):
        """Send photo to Telegram.

        Args:
            photo_url: URL of photo to send
            caption: Optional caption
        """
        if not self.bot_token or not self.chat_id:
            return

        url = f"{self.base_url}/sendPhoto"
        payload = {
            "chat_id": self.chat_id,
            "photo": photo_url,
        }

        if caption:
            payload["caption"] = caption
            payload["parse_mode"] = "Markdown"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.logger.debug("Telegram photo sent")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Telegram photo error: {error_text}")

        except Exception as e:
            self.logger.error(f"Failed to send Telegram photo: {e}")

    async def send_chart(
        self,
        chart_data: bytes,
        caption: Optional[str] = None,
    ):
        """Send chart image to Telegram.

        Args:
            chart_data: Chart image as bytes
            caption: Optional caption
        """
        if not self.bot_token or not self.chat_id:
            return

        url = f"{self.base_url}/sendPhoto"

        # Prepare multipart form data
        data = aiohttp.FormData()
        data.add_field('chat_id', self.chat_id)
        data.add_field('photo', chart_data, filename='chart.png', content_type='image/png')

        if caption:
            data.add_field('caption', caption)
            data.add_field('parse_mode', 'Markdown')

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        self.logger.debug("Telegram chart sent")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Telegram chart error: {error_text}")

        except Exception as e:
            self.logger.error(f"Failed to send Telegram chart: {e}")
