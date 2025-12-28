"""Docker secrets management for secure credential loading.

This module provides utilities for loading secrets from Docker secrets files
mounted at /run/secrets/<secret_name>, with fallback to environment variables
for local development.

Priority: Docker secrets > Environment variables

Usage:
    from trading.config.secrets import SecretsManager

    # Load exchange credentials
    api_key, api_secret = SecretsManager.get_exchange_credentials()

    # Load individual secret
    db_password = SecretsManager.get_db_password()
"""

import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SecretsManager:
    """Load secrets from Docker secrets or fallback to environment variables."""

    SECRETS_DIR = Path("/run/secrets")

    @classmethod
    def get_secret(cls, secret_name: str, env_var_name: Optional[str] = None) -> Optional[str]:
        """
        Load secret from Docker secrets file or environment variable.

        Args:
            secret_name: Name of the secret file (e.g., 'exchange_api_key')
            env_var_name: Fallback environment variable name (e.g., 'EXCHANGE_API_KEY')

        Returns:
            Secret value or None if not found

        Example:
            >>> secret = SecretsManager.get_secret("exchange_api_key", "EXCHANGE_API_KEY")
            >>> if secret:
            ...     print("Secret loaded successfully")
        """
        # Try Docker secrets first
        secret_path = cls.SECRETS_DIR / secret_name
        if secret_path.exists():
            try:
                secret_value = secret_path.read_text().strip()
                logger.info(f"Loaded secret from Docker secrets: {secret_name}")
                return secret_value
            except Exception as e:
                logger.error(f"Failed to read secret {secret_name}: {e}")

        # Fallback to environment variable
        if env_var_name:
            env_value = os.getenv(env_var_name)
            if env_value:
                logger.warning(f"Using environment variable {env_var_name} (Docker secrets not available)")
                return env_value

        logger.error(f"Secret {secret_name} not found in Docker secrets or environment variables")
        return None

    @classmethod
    def get_exchange_credentials(cls) -> tuple[Optional[str], Optional[str]]:
        """
        Load exchange API key and secret.

        Returns:
            Tuple of (api_key, api_secret) or (None, None) if not found

        Example:
            >>> api_key, api_secret = SecretsManager.get_exchange_credentials()
            >>> if api_key and api_secret:
            ...     client = ExchangeClient(api_key, api_secret)
        """
        api_key = cls.get_secret("exchange_api_key", "EXCHANGE_API_KEY")
        api_secret = cls.get_secret("exchange_api_secret", "EXCHANGE_API_SECRET")
        return api_key, api_secret

    @classmethod
    def get_kill_switch_secret(cls) -> Optional[str]:
        """
        Load kill switch secret for HMAC authentication.

        Returns:
            Kill switch secret or None if not found

        Example:
            >>> secret = SecretsManager.get_kill_switch_secret()
            >>> if secret:
            ...     kill_switch = KillSwitch(secret)
        """
        return cls.get_secret("kill_switch_secret", "KILL_SWITCH_SECRET")

    @classmethod
    def get_db_password(cls) -> Optional[str]:
        """
        Load database password.

        Returns:
            Database password or None if not found

        Example:
            >>> password = SecretsManager.get_db_password()
            >>> if password:
            ...     db_url = f"postgresql://user:{password}@host/db"
        """
        return cls.get_secret("db_password", "DB_PASSWORD")
