"""Coinbase Advanced Trade provider implementation.

Implements BaseExchangeProvider for Coinbase cryptocurrency exchange.
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


class CoinbaseProvider(BaseExchangeProvider):
    """Coinbase Advanced Trade provider using ccxt.

    Provides access to Coinbase cryptocurrency exchange (formerly Coinbase Pro).
    """

    def __init__(self, config: TradingConfig):
        """Initialize Coinbase provider.

        Args:
            config: Trading configuration with API credentials

        Raises:
            ValueError: If API credentials are missing or invalid
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        if not config.is_valid():
            raise ValueError(
                "Invalid Coinbase configuration: API key and secret required"
            )

        # Initialize ccxt exchange instance
        # Note: ccxt uses 'coinbase' for Advanced Trade (formerly Pro)
        self.exchange = ccxt.coinbase({
            'apiKey': config.api_key,
            'secret': config.api_secret,
            'enableRateLimit': True,
        })

        # Coinbase has sandbox mode
        if config.testnet:
            self.exchange.set_sandbox_mode(True)
            self.logger.info("Coinbase: Sandbox mode enabled")
        else:
            self.logger.warning("Coinbase: LIVE TRADING MODE - real money at risk!")

        self.logger.info("Coinbase Advanced Trade provider initialized")

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> list[OHLCV]:
        """Fetch OHLCV candlestick data from Coinbase.

        Args:
            symbol: Trading pair (e.g., "BTC/USD")
            timeframe: Candle timeframe
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
        """Fetch current ticker data from Coinbase.

        Args:
            symbol: Trading pair

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
        """Fetch order book from Coinbase.

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
        """Fetch account balance from Coinbase.

        Returns:
            Balance (defaults to USD)
        """
        try:
            raw_balance = await self.exchange.fetch_balance()

            # Coinbase typically uses USD
            for currency in ['USD', 'USDC', 'USDT']:
                balance_data = raw_balance.get(currency, {})
                if balance_data:
                    return Balance.from_ccxt(currency, balance_data)

            # Fallback to first available
            for currency, balance_data in raw_balance.items():
                if currency not in ['free', 'used', 'total', 'info']:
                    return Balance.from_ccxt(currency, balance_data)

            return Balance(currency='USD', free=0.0, used=0.0, total=0.0)

        except Exception as e:
            self.logger.error(f"Failed to fetch balance: {e}")
            raise

    async def fetch_positions(self) -> list[Position]:
        """Fetch open positions from Coinbase.

        Note: Coinbase spot trading doesn't have positions.

        Returns:
            Empty list (spot trading only)
        """
        # Coinbase Advanced Trade is spot only
        return []

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Order:
        """Create an order on Coinbase.

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
            if order_type == "limit" and price is None:
                raise ValueError("Price is required for limit orders")

            raw_order = await self.exchange.create_order(
                symbol, order_type, side, amount, price
            )

            return Order.from_ccxt(raw_order)

        except Exception as e:
            self.logger.error(f"Failed to create order: {e}")
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order on Coinbase.

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
            "coinbase"
        """
        return "coinbase"

    async def close(self):
        """Close exchange connection (cleanup)."""
        if self.exchange:
            await self.exchange.close()
