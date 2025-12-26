"""Provider factory for creating exchange provider instances.

This module implements the factory pattern to create the appropriate
exchange provider based on configuration.
"""

from ..config import TradingConfig
from .base import BaseExchangeProvider


def create_provider(config: TradingConfig) -> BaseExchangeProvider:
    """Create an exchange provider instance from configuration.

    This factory function implements the DRY principle by centralizing
    provider instantiation logic.

    Args:
        config: Trading configuration with provider name and credentials

    Returns:
        Initialized exchange provider instance

    Raises:
        ValueError: If provider name is not recognized

    Example:
        >>> config = TradingConfig.from_env("binance_futures")
        >>> provider = create_provider(config)
        >>> provider.get_provider_name()
        'binance_futures'
    """
    # Import providers locally to avoid circular imports
    # and to only load what's needed
    provider_map = {
        "binance_futures": "BinanceFuturesProvider",
        "binance_spot": "BinanceSpotProvider",
        "kraken": "KrakenProvider",
        "coinbase": "CoinbaseProvider",
        "alpaca": "AlpacaProvider",
        "paper": "PaperProvider",
    }

    provider_class_name = provider_map.get(config.provider)

    if not provider_class_name:
        available = ", ".join(provider_map.keys())
        raise ValueError(
            f"Unknown provider: {config.provider}. "
            f"Available providers: {available}"
        )

    # Dynamically import the provider class
    if config.provider == "binance_futures":
        from .binance_futures import BinanceFuturesProvider
        return BinanceFuturesProvider(config)
    elif config.provider == "binance_spot":
        from .binance_spot import BinanceSpotProvider
        return BinanceSpotProvider(config)
    elif config.provider == "kraken":
        from .kraken import KrakenProvider
        return KrakenProvider(config)
    elif config.provider == "coinbase":
        from .coinbase import CoinbaseProvider
        return CoinbaseProvider(config)
    elif config.provider == "alpaca":
        from .alpaca import AlpacaProvider
        return AlpacaProvider(config)
    elif config.provider == "paper":
        from .paper import PaperProvider
        return PaperProvider(config)

    # This shouldn't be reached, but just in case
    raise ValueError(f"Provider {config.provider} not implemented yet")
