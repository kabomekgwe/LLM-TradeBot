"""Binance Futures exchange provider implementation.

This module implements the BaseExchangeProvider interface for Binance Futures
using the ccxt library for unified exchange access.
"""

import asyncio
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


class BinanceFuturesProvider(BaseExchangeProvider):
    """Binance Futures exchange provider using ccxt.

    Provides access to Binance Futures perpetual contracts with support
    for testnet trading (default) and live trading.
    """

    def __init__(self, config: TradingConfig):
        """Initialize Binance Futures provider.

        Args:
            config: Trading configuration with API credentials

        Raises:
            ValueError: If API credentials are missing or invalid
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        if not config.is_valid():
            raise ValueError(
                "Invalid Binance Futures configuration: API key and secret required"
            )

        # Initialize ccxt exchange instance
        self.exchange = ccxt.binance({
            'apiKey': config.api_key,
            'secret': config.api_secret,
            'enableRateLimit': True,  # Respect rate limits
            'options': {
                'defaultType': 'future',  # Use futures, not spot
            }
        })

        # Enable testnet mode if configured (default: true for safety)
        if config.testnet:
            self.exchange.set_sandbox_mode(True)
            self.logger.info("Binance Futures: Testnet mode enabled")
        else:
            self.logger.warning("Binance Futures: LIVE TRADING MODE - real money at risk!")

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> list[OHLCV]:
        """Fetch OHLCV candlestick data from Binance Futures.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle timeframe ("5m", "15m", "1h", etc.)
            limit: Number of candles to fetch

        Returns:
            List of OHLCV data points, sorted by timestamp

        Example:
            >>> provider = BinanceFuturesProvider(config)
            >>> candles = await provider.fetch_ohlcv("BTC/USDT", "1h", limit=100)
            >>> candles[0].close
            42000.0
        """
        try:
            raw_data = await self.exchange.fetch_ohlcv(
                symbol, timeframe, limit=limit
            )

            # Convert to unified OHLCV format using from_ccxt method
            return [OHLCV.from_ccxt(row) for row in raw_data]

        except Exception as e:
            self.logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            raise

    async def fetch_ticker(self, symbol: str) -> Ticker:
        """Fetch current ticker data from Binance Futures.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")

        Returns:
            Ticker with current prices

        Example:
            >>> ticker = await provider.fetch_ticker("BTC/USDT")
            >>> ticker.last
            42000.0
        """
        try:
            raw_ticker = await self.exchange.fetch_ticker(symbol)

            # Use from_ccxt method for consistent conversion
            return Ticker.from_ccxt(symbol, raw_ticker)

        except Exception as e:
            self.logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            raise

    async def fetch_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """Fetch order book from Binance Futures.

        Args:
            symbol: Trading pair
            limit: Order book depth

        Returns:
            OrderBook with bids and asks

        Example:
            >>> orderbook = await provider.fetch_orderbook("BTC/USDT", limit=10)
            >>> orderbook.bids[0]  # Best bid
            (41999.5, 0.5)
        """
        try:
            raw_orderbook = await self.exchange.fetch_order_book(symbol, limit)

            # Use from_ccxt method for consistent conversion
            return OrderBook.from_ccxt(symbol, raw_orderbook)

        except Exception as e:
            self.logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            raise

    async def fetch_balance(self) -> Balance:
        """Fetch account balance from Binance Futures.

        Returns:
            Balance (defaults to USDT for futures)

        Example:
            >>> balance = await provider.fetch_balance()
            >>> balance.total
            10000.0
        """
        try:
            raw_balance = await self.exchange.fetch_balance()

            # For futures, we typically care about USDT balance
            usdt_balance = raw_balance.get('USDT', {})

            # Use from_ccxt method for consistent conversion
            return Balance.from_ccxt('USDT', usdt_balance)

        except Exception as e:
            self.logger.error(f"Failed to fetch balance: {e}")
            raise

    async def fetch_positions(self) -> list[Position]:
        """Fetch open positions from Binance Futures.

        Returns:
            List of open positions

        Example:
            >>> positions = await provider.fetch_positions()
            >>> len(positions)
            2
        """
        try:
            raw_positions = await self.exchange.fetch_positions()

            positions = []
            for raw_pos in raw_positions:
                # Skip positions with zero size
                size = float(raw_pos.get('contracts', 0))
                if size == 0:
                    continue

                # Use from_ccxt method for consistent conversion
                positions.append(Position.from_ccxt(raw_pos))

            return positions

        except Exception as e:
            self.logger.error(f"Failed to fetch positions: {e}")
            raise

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Order:
        """Create an order on Binance Futures.

        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            order_type: "market" or "limit"
            amount: Order size
            price: Limit price (required for limit orders)

        Returns:
            Created order

        Example:
            >>> order = await provider.create_order(
            ...     "BTC/USDT", "buy", "market", 0.1
            ... )
            >>> order.status
            'closed'
        """
        try:
            # Validate inputs
            if order_type == "limit" and price is None:
                raise ValueError("Price is required for limit orders")

            # Create order via ccxt
            raw_order = await self.exchange.create_order(
                symbol, order_type, side, amount, price
            )

            # Use from_ccxt method for consistent conversion
            return Order.from_ccxt(raw_order)

        except Exception as e:
            self.logger.error(f"Failed to create order: {e}")
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order on Binance Futures.

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair

        Returns:
            True if canceled successfully

        Example:
            >>> success = await provider.cancel_order("12345", "BTC/USDT")
            >>> success
            True
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
            "binance_futures"
        """
        return "binance_futures"

    async def close(self):
        """Close exchange connection (cleanup).

        Call this when done with the provider to free resources.

        Example:
            >>> provider = BinanceFuturesProvider(config)
            >>> # ... use provider ...
            >>> await provider.close()
        """
        if self.exchange:
            await self.exchange.close()
