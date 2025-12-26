"""Kraken exchange provider implementation.

Implements BaseExchangeProvider for Kraken cryptocurrency exchange.
"""

import logging
from typing import Optional

try:
    import ccxt.async_support as ccxt
except ImportError:
    raise ImportError(
        "ccxt library is required for trading integration. "
        "Install with: pip install ccxt>=4.0.0"
    )

from ..config import TradingConfig
from .base import BaseExchangeProvider
from ..models.market_data import OHLCV, Ticker, OrderBook, Balance
from ..models.positions import Position, Order


class KrakenProvider(BaseExchangeProvider):
    """Kraken exchange provider using ccxt.

    Provides access to Kraken cryptocurrency exchange with support
    for spot trading.
    """

    def __init__(self, config: TradingConfig):
        """Initialize Kraken provider.

        Args:
            config: Trading configuration with API credentials

        Raises:
            ValueError: If API credentials are missing or invalid
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        if not config.is_valid():
            raise ValueError(
                "Invalid Kraken configuration: API key and secret required"
            )

        # Initialize ccxt exchange instance
        self.exchange = ccxt.kraken({
            'apiKey': config.api_key,
            'secret': config.api_secret,
            'enableRateLimit': True,  # Respect rate limits
        })

        # Note: Kraken doesn't have a testnet like Binance
        if config.testnet:
            self.logger.warning(
                "Kraken does not have a testnet. "
                "Consider using paper trading provider for testing."
            )

        self.logger.info("Kraken provider initialized")

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> list[OHLCV]:
        """Fetch OHLCV candlestick data from Kraken.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle timeframe ("5m", "15m", "1h", etc.)
            limit: Number of candles to fetch

        Returns:
            List of OHLCV data points
        """
        try:
            raw_data = await self.exchange.fetch_ohlcv(
                symbol, timeframe, limit=limit
            )

            return [OHLCV.from_ccxt(row) for row in raw_data]

        except Exception as e:
            self.logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            raise

    async def fetch_ticker(self, symbol: str) -> Ticker:
        """Fetch current ticker data from Kraken.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")

        Returns:
            Ticker with current prices
        """
        try:
            raw_ticker = await self.exchange.fetch_ticker(symbol)
            return Ticker.from_ccxt(symbol, raw_ticker)

        except Exception as e:
            self.logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            raise

    async def fetch_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """Fetch order book from Kraken.

        Args:
            symbol: Trading pair
            limit: Order book depth

        Returns:
            OrderBook with bids and asks
        """
        try:
            raw_orderbook = await self.exchange.fetch_order_book(symbol, limit)
            return OrderBook.from_ccxt(symbol, raw_orderbook)

        except Exception as e:
            self.logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            raise

    async def fetch_balance(self) -> Balance:
        """Fetch account balance from Kraken.

        Returns:
            Balance (defaults to USD for Kraken)
        """
        try:
            raw_balance = await self.exchange.fetch_balance()

            # Kraken uses USD or USDT depending on account
            # Try USDT first, fall back to USD
            for currency in ['USDT', 'USD']:
                balance_data = raw_balance.get(currency, {})
                if balance_data:
                    return Balance.from_ccxt(currency, balance_data)

            # If neither found, return first available currency
            for currency, balance_data in raw_balance.items():
                if currency not in ['free', 'used', 'total', 'info']:
                    return Balance.from_ccxt(currency, balance_data)

            # Fallback
            return Balance(currency='USD', free=0.0, used=0.0, total=0.0)

        except Exception as e:
            self.logger.error(f"Failed to fetch balance: {e}")
            raise

    async def fetch_positions(self) -> list[Position]:
        """Fetch open positions from Kraken.

        Note: Kraken spot trading doesn't have positions in the same way
        as futures. This returns an empty list for spot accounts.

        Returns:
            List of open positions (empty for spot)
        """
        # Kraken spot doesn't have positions like futures
        # For derivatives/margin trading, would need to implement separately
        self.logger.debug("Kraken spot trading does not use positions")
        return []

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Order:
        """Create an order on Kraken.

        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            order_type: "market" or "limit"
            amount: Order size
            price: Limit price (required for limit orders)

        Returns:
            Created order
        """
        try:
            # Validate inputs
            if order_type == "limit" and price is None:
                raise ValueError("Price is required for limit orders")

            # Create order via ccxt
            raw_order = await self.exchange.create_order(
                symbol, order_type, side, amount, price
            )

            return Order.from_ccxt(raw_order)

        except Exception as e:
            self.logger.error(f"Failed to create order: {e}")
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order on Kraken.

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair

        Returns:
            True if canceled successfully
        """
        try:
            await self.exchange.cancel_order(order_id, symbol)
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_provider_name(self) -> str:
        """Return provider name.

        Returns:
            "kraken"
        """
        return "kraken"

    async def close(self):
        """Close exchange connection (cleanup)."""
        if self.exchange:
            await self.exchange.close()
