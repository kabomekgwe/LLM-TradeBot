"""Trading configuration management.

This module provides configuration dataclasses for the trading integration,
following the pattern established by the Linear integration.

SECURITY: All credentials loaded from environment variables only.
See .env.example for configuration template.
"""

import os
from dataclasses import dataclass
from typing import Optional

# Load .env file if it exists (for local development)
# In production, environment variables are set by the deployment platform
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed (production environment)

from .validation import validate_exchange_name, ValidationError


@dataclass
class TradingConfig:
    """Trading configuration loaded from environment variables.

    Follows the same pattern as LinearConfig for consistency with
    Auto Claude's integration architecture.

    SECURITY: Never instantiate this class directly with credentials.
    Always use TradingConfig.from_env() to load from environment variables.
    Credentials are NEVER hardcoded or passed as parameters.
    """

    # Provider configuration
    provider: str  # "binance_futures", "binance_spot", "paper", "kraken", "alpaca"
    api_key: str  # Loaded from {PROVIDER}_API_KEY env var (use from_env())
    api_secret: str  # Loaded from {PROVIDER}_API_SECRET env var (use from_env())
    testnet: bool = True  # Default to testnet for safety
    enabled: bool = True

    # Risk management parameters
    max_position_size_usd: float = 1000.0  # Maximum position size in USD
    max_daily_drawdown_pct: float = 5.0  # Maximum daily drawdown percentage
    max_open_positions: int = 3  # Maximum number of concurrent positions

    # Agent parameters
    decision_threshold: float = 0.6  # Minimum confidence for trade execution
    enable_ml_predictions: bool = True  # Enable LightGBM ML forecasting
    enable_adversarial_voting: bool = True  # Enable bull/bear adversarial system

    # Notification settings
    notifications_enabled: bool = False  # Master switch for notifications

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Discord
    discord_webhook: str = ""

    # Email (SMTP)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_to: str = ""
    email_from: str = ""

    # Slack
    slack_webhook: str = ""

    # Web Dashboard
    dashboard_enabled: bool = False
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 5173

    @classmethod
    def from_env(cls, provider: Optional[str] = None) -> "TradingConfig":
        """Load trading configuration from environment variables.

        Args:
            provider: Trading provider name. If None, reads from TRADING_PROVIDER env var.
                     Defaults to "binance_futures" if not set.

        Returns:
            TradingConfig instance with values from environment

        Raises:
            ValueError: If credentials are missing for non-paper providers

        Example:
            >>> config = TradingConfig.from_env("binance_futures")
            >>> config.is_valid()
            True
        """
        if provider is None:
            provider = os.getenv("TRADING_PROVIDER", "binance_futures")

        # Validate provider name at boundary (fail-fast)
        provider = validate_exchange_name(provider)

        # Normalize provider name for env var lookup
        provider_upper = provider.upper().replace("-", "_")

        # Load API credentials from environment (NEVER from code)
        api_key = os.getenv(f"{provider_upper}_API_KEY", "")
        api_secret = os.getenv(f"{provider_upper}_API_SECRET", "")

        # Validate credentials are present (fail-fast for non-paper providers)
        if provider != "paper" and (not api_key or not api_secret):
            raise ValueError(
                f"Missing credentials for {provider}. "
                f"Set {provider_upper}_API_KEY and {provider_upper}_API_SECRET environment variables. "
                f"See .env.example for template."
            )

        # Load testnet setting (default: true for safety)
        testnet_str = os.getenv("TRADING_TESTNET", "true").lower()
        testnet = testnet_str in ("true", "1", "yes")

        # Load risk parameters
        max_position_size = float(os.getenv("TRADING_MAX_POSITION_SIZE_USD", "1000.0"))
        max_drawdown = float(os.getenv("TRADING_MAX_DAILY_DRAWDOWN_PCT", "5.0"))
        max_positions = int(os.getenv("TRADING_MAX_OPEN_POSITIONS", "3"))

        # Load agent parameters
        decision_threshold = float(os.getenv("TRADING_DECISION_THRESHOLD", "0.6"))
        enable_ml = os.getenv("TRADING_ENABLE_ML_PREDICTIONS", "true").lower() in ("true", "1", "yes")
        enable_adf = os.getenv("TRADING_ENABLE_ADVERSARIAL_VOTING", "true").lower() in ("true", "1", "yes")

        # Determine if enabled (has API key, or is paper trading)
        enabled = bool(api_key) or provider == "paper"

        # Load notification settings
        notifications_enabled = os.getenv("NOTIFICATIONS_ENABLED", "false").lower() in ("true", "1", "yes")

        # Telegram
        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

        # Discord
        discord_webhook = os.getenv("DISCORD_WEBHOOK", "")

        # Email
        smtp_host = os.getenv("SMTP_HOST", "")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        email_to = os.getenv("EMAIL_TO", "")
        email_from = os.getenv("EMAIL_FROM", smtp_user)  # Default to smtp_user

        # Slack
        slack_webhook = os.getenv("SLACK_WEBHOOK", "")

        # Web Dashboard
        dashboard_enabled = os.getenv("DASHBOARD_ENABLED", "false").lower() in ("true", "1", "yes")
        dashboard_host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
        dashboard_port = int(os.getenv("DASHBOARD_PORT", "5173"))

        return cls(
            provider=provider,
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            enabled=enabled,
            max_position_size_usd=max_position_size,
            max_daily_drawdown_pct=max_drawdown,
            max_open_positions=max_positions,
            decision_threshold=decision_threshold,
            enable_ml_predictions=enable_ml,
            enable_adversarial_voting=enable_adf,
            notifications_enabled=notifications_enabled,
            telegram_bot_token=telegram_bot_token,
            telegram_chat_id=telegram_chat_id,
            discord_webhook=discord_webhook,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            email_to=email_to,
            email_from=email_from,
            slack_webhook=slack_webhook,
            dashboard_enabled=dashboard_enabled,
            dashboard_host=dashboard_host,
            dashboard_port=dashboard_port,
        )

    def is_valid(self) -> bool:
        """Check if configuration has minimum required values.

        Returns:
            True if config is valid and ready to use

        Note:
            Paper trading doesn't require API keys, so only checks
            that api_key is set for non-paper providers.
        """
        # Paper trading doesn't need API keys
        if self.provider == "paper":
            return True

        # Other providers need both API key and secret
        return bool(self.api_key and self.api_secret)

    def validate_risk_parameters(self) -> tuple[bool, Optional[str]]:
        """Validate risk management parameters.

        Returns:
            (is_valid, error_message) tuple
        """
        if self.max_position_size_usd <= 0:
            return False, "max_position_size_usd must be positive"

        if self.max_daily_drawdown_pct <= 0 or self.max_daily_drawdown_pct > 100:
            return False, "max_daily_drawdown_pct must be between 0 and 100"

        if self.max_open_positions <= 0:
            return False, "max_open_positions must be positive"

        if self.decision_threshold < 0 or self.decision_threshold > 1:
            return False, "decision_threshold must be between 0 and 1"

        return True, None

    def __repr__(self) -> str:
        """String representation with fully masked sensitive fields.

        Fully masks API secrets to prevent accidental leaks in logs, error messages,
        or debug output. API keys show first 8 chars for debugging (non-sensitive identifier).
        """
        # Fully mask all secrets - never expose even partial values
        masked_secret = "***REDACTED***" if self.api_secret else None
        masked_key = f"{self.api_key[:8]}..." if self.api_key else None

        # Also mask notification secrets
        masked_telegram_token = "***REDACTED***" if self.telegram_bot_token else None
        masked_smtp_password = "***REDACTED***" if self.smtp_password else None
        masked_discord_webhook = "***REDACTED***" if self.discord_webhook else None
        masked_slack_webhook = "***REDACTED***" if self.slack_webhook else None

        return (
            f"TradingConfig(provider={self.provider}, "
            f"testnet={self.testnet}, "
            f"enabled={self.enabled}, "
            f"api_key={masked_key}, "
            f"api_secret={masked_secret}, "
            f"telegram_bot_token={masked_telegram_token}, "
            f"smtp_password={masked_smtp_password}, "
            f"discord_webhook={masked_discord_webhook}, "
            f"slack_webhook={masked_slack_webhook})"
        )
