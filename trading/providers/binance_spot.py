"""Binance Spot exchange provider implementation.

Implements BaseExchangeProvider for Binance Spot (not futures) trading.
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


class BinanceSpotProvider(BaseExchangeProvider):
    """Binance Spot exchange provider using ccxt.

    Provides access to Binance Spot trading (not futures/derivatives).
    """

    def __init__(self, config: TradingConfig):
        """Initialize Binance Spot provider.

        Args:
            config: Trading configuration with API credentials

        Raises:
            ValueError: If API credentials are missing or invalid
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        if not config.is_valid():
            raise ValueError(
                "Invalid Binance Spot configuration: API key and secret required"
            )

        # Initialize ccxt exchange instance
        self.exchange = ccxt.binance({
            'apiKey': config.api_key,
            'secret': config.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Important: spot, not future
            }
        })

        # Enable testnet if configured
        if config.testnet:
            self.exchange.set_sandbox_mode(True)
            self.logger.info("Binance Spot: Testnet mode enabled")
        else:
            self.logger.warning("Binance Spot: LIVE TRADING MODE - real money at risk!")

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> list[OHLCV]:
        """Fetch OHLCV candlestick data from Binance Spot.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
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
        """Fetch current ticker data from Binance Spot.

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
        """Fetch order book from Binance Spot.

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
        """Fetch account balance from Binance Spot.

        Returns:
            Balance (typically USDT for spot)
        """
        try:
            raw_balance = await self.exchange.fetch_balance()

            # Spot typically uses USDT
            usdt_balance = raw_balance.get('USDT', {})

            return Balance.from_ccxt('USDT', usdt_balance)

        except Exception as e:
            self.logger.error(f"Failed to fetch balance: {e}")
            raise

    async def fetch_positions(self) -> list[Position]:
        """Fetch open positions from Binance Spot.

        Note: Spot trading doesn't have positions like futures.

        Returns:
            Empty list (spot trading only)
        """
        # Spot trading doesn't have positions
        return []

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Order:
        """Create an order on Binance Spot.

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
        """Cancel an order on Binance Spot.

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
            "binance_spot"
        """
        return "binance_spot"

    async def close(self):
        """Close exchange connection (cleanup)."""
        if self.exchange:
            await self.exchange.close()
