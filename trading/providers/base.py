"""Base exchange provider interface.

This module defines the abstract base class that all exchange providers must implement.
Provides a unified interface for interacting with different exchanges and brokers.
"""

from abc import ABC, abstractmethod
from typing import Optional

# Import unified data models
from ..models.market_data import OHLCV, Ticker, OrderBook, Balance
from ..models.positions import Position, Order


class BaseExchangeProvider(ABC):
    """Abstract base class for exchange providers.

    All exchange/broker integrations must implement this interface to ensure
    consistent behavior across different platforms.

    This follows the DRY principle by providing a single interface that
    eliminates platform-specific code duplication.
    """

    @abstractmethod
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> list[OHLCV]:
        """Fetch OHLCV (candlestick) data.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candlestick timeframe (e.g., "5m", "15m", "1h")
            limit: Number of candles to fetch

        Returns:
            List of OHLCV data points, newest first

        Example:
            >>> provider = BinanceFuturesProvider(config)
            >>> candles = await provider.fetch_ohlcv("BTC/USDT", "1h", limit=100)
            >>> len(candles)
            100
        """
        pass

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Ticker:
        """Fetch current ticker (price) data.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")

        Returns:
            Ticker with current bid/ask/last prices

        Example:
            >>> ticker = await provider.fetch_ticker("BTC/USDT")
            >>> ticker.last
            42000.0
        """
        pass

    @abstractmethod
    async def fetch_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """Fetch current order book.

        Args:
            symbol: Trading pair symbol
            limit: Depth of order book to fetch

        Returns:
            OrderBook with bids and asks

        Example:
            >>> orderbook = await provider.fetch_orderbook("BTC/USDT", limit=10)
            >>> orderbook.bids[0]  # Best bid
            (41999.5, 0.5)
        """
        pass

    @abstractmethod
    async def fetch_balance(self) -> Balance:
        """Fetch account balance.

        Returns:
            Balance with free, used, and total amounts

        Example:
            >>> balance = await provider.fetch_balance()
            >>> balance.total
            10000.0
        """
        pass

    @abstractmethod
    async def fetch_positions(self) -> list[Position]:
        """Fetch open positions.

        Returns:
            List of open positions

        Example:
            >>> positions = await provider.fetch_positions()
            >>> len(positions)
            2
        """
        pass

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        trailing_delta: Optional[float] = None,
    ) -> Order:
        """Create an order with optional advanced parameters.

        Args:
            symbol: Trading pair symbol
            side: "buy" or "sell"
            order_type: "market", "limit", "stop_loss", or "take_profit"
            amount: Order amount/size
            price: Limit price (required for limit orders)
            stop_price: Stop-loss trigger price (for stop_loss orders)
            take_profit_price: Take-profit trigger price (for take_profit orders)
            trailing_delta: Trailing stop distance (% or absolute)

        Returns:
            Created order

        Example:
            >>> # Simple market order
            >>> order = await provider.create_order(
            ...     "BTC/USDT", "buy", "market", 0.1
            ... )

            >>> # Stop-loss order
            >>> stop = await provider.create_order(
            ...     "BTC/USDT", "sell", "stop_loss", 0.1,
            ...     stop_price=41000
            ... )

            >>> # Take-profit order
            >>> tp = await provider.create_order(
            ...     "BTC/USDT", "sell", "take_profit", 0.1,
            ...     take_profit_price=44000
            ... )
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair symbol

        Returns:
            True if canceled successfully

        Example:
            >>> success = await provider.cancel_order("12345", "BTC/USDT")
            >>> success
            True
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name.

        Returns:
            Provider name (e.g., "binance_futures", "kraken")

        Example:
            >>> provider.get_provider_name()
            'binance_futures'
        """
        pass
