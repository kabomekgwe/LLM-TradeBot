"""Multi-Channel Notifications - Real-time trading alerts.

Sends notifications across Telegram, Discord, Email, and Slack for:
- Trade execution
- Stop-loss / Take-profit triggers
- Circuit breaker alerts
- Daily performance summaries
- System errors and status

Components:
- NotificationManager: Unified interface for all channels
- TelegramNotifier: Telegram bot notifications
- DiscordNotifier: Discord webhook notifications
- EmailNotifier: SMTP email notifications
- SlackNotifier: Slack webhook notifications

Example Usage:
    ```python
    from trading.notifications import NotificationManager
    from trading.config import TradingConfig

    # Initialize
    config = TradingConfig.from_env("binance_futures")
    notifier = NotificationManager(config)

    # Trade executed
    await notifier.send_trade_executed(
        order=order,
        symbol="BTC/USDT",
        price=42000,
        stop_loss_price=41000,
        take_profit_price=44000,
    )

    # Stop-loss hit
    await notifier.send_stop_loss_hit(
        order=stop_order,
        stop_price=41000,
        pnl=-1000,
    )

    # Take-profit hit
    await notifier.send_take_profit_hit(
        order=tp_order,
        take_profit_price=44000,
        pnl=2000,
    )

    # Circuit breaker
    await notifier.send_circuit_breaker_triggered(
        reason="Max drawdown exceeded",
        current_drawdown=15.0,
        max_drawdown=10.0,
    )

    # Daily summary
    await notifier.send_daily_summary(
        total_trades=45,
        winning_trades=28,
        total_pnl=1250.0,
        win_rate=62.2,
    )
    ```

Configuration:
    Set these environment variables or add to .env:

    ```bash
    # Telegram
    TELEGRAM_BOT_TOKEN=your_bot_token
    TELEGRAM_CHAT_ID=your_chat_id

    # Discord
    DISCORD_WEBHOOK=https://discord.com/api/webhooks/...

    # Email (SMTP)
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=your.email@gmail.com
    SMTP_PASSWORD=your_app_password
    EMAIL_TO=recipient@example.com

    # Slack
    SLACK_WEBHOOK=https://hooks.slack.com/services/...
    ```

Setup Guides:

**Telegram:**
1. Message @BotFather on Telegram
2. Send /newbot and follow instructions
3. Copy the bot token
4. Start a chat with your bot
5. Get your chat_id from @userinfobot

**Discord:**
1. Go to channel settings → Integrations
2. Create webhook, copy URL
3. Set DISCORD_WEBHOOK in config

**Email (Gmail):**
1. Enable 2FA on Google account
2. Create App Password (Security → App passwords)
3. Use app password (not your regular password)

**Slack:**
1. Go to https://api.slack.com/apps
2. Create app → Incoming Webhooks
3. Activate and add to channel
4. Copy webhook URL

Features:
- Multi-channel broadcasting
- Priority-based filtering
- Rich formatting (markdown, embeds, HTML)
- Automatic error handling
- Rate limiting to prevent spam

Notification Levels:
- INFO: General updates, system status
- SUCCESS: Successful trades, profitable exits
- WARNING: Risk warnings, approaching limits
- ERROR: Trade failures, connection issues
- CRITICAL: Circuit breakers, system failures
"""

from .manager import (
    NotificationManager,
    NotificationLevel,
    NotificationChannel,
    Notification,
)
from .telegram import TelegramNotifier
from .discord import DiscordNotifier
from .email import EmailNotifier
from .slack import SlackNotifier

__all__ = [
    # Core
    "NotificationManager",
    "NotificationLevel",
    "NotificationChannel",
    "Notification",

    # Channel implementations
    "TelegramNotifier",
    "DiscordNotifier",
    "EmailNotifier",
    "SlackNotifier",
]

__version__ = "1.0.0"

# Module metadata
__author__ = "LLM-TradeBot Contributors"
__description__ = "Multi-channel notifications: Telegram, Discord, Email, Slack"
