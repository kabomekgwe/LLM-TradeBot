"""Alpaca stock trading provider implementation.

Implements BaseExchangeProvider for Alpaca stock market trading.
"""

import logging
from typing import Optional
from datetime import datetime, timedelta

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except ImportError:
    raise ImportError(
        "alpaca-py library is required for Alpaca stock trading. "
        "Install with: pip install alpaca-py>=0.9.0"
    )

from ..config import TradingConfig
from .base import BaseExchangeProvider
from ..models.market_data import OHLCV, Ticker, OrderBook, Balance
from ..models.positions import Position, Order, PositionSide, OrderSide, OrderType, OrderStatus


class AlpacaProvider(BaseExchangeProvider):
    """Alpaca stock trading provider.

    Provides access to US stock market trading via Alpaca API.
    Supports both paper trading and live trading.
    """

    def __init__(self, config: TradingConfig):
        """Initialize Alpaca provider.

        Args:
            config: Trading configuration with API credentials

        Raises:
            ValueError: If API credentials are missing or invalid
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        if not config.is_valid():
            raise ValueError(
                "Invalid Alpaca configuration: API key and secret required"
            )

        # Alpaca has separate paper and live endpoints
        if config.testnet:
            # Paper trading (default)
            self.trading_client = TradingClient(
                api_key=config.api_key,
                secret_key=config.api_secret,
                paper=True,
            )
            self.data_client = StockHistoricalDataClient(
                api_key=config.api_key,
                secret_key=config.api_secret,
            )
            self.logger.info("Alpaca: Paper trading mode enabled")
        else:
            # Live trading - real money!
            self.trading_client = TradingClient(
                api_key=config.api_key,
                secret_key=config.api_secret,
                paper=False,
            )
            self.data_client = StockHistoricalDataClient(
                api_key=config.api_key,
                secret_key=config.api_secret,
            )
            self.logger.warning("Alpaca: LIVE TRADING MODE - real money at risk!")

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> list[OHLCV]:
        """Fetch OHLCV candlestick data from Alpaca.

        Args:
            symbol: Stock symbol (e.g., "AAPL", "TSLA")
            timeframe: Candle timeframe (e.g., "5m", "15m", "1h", "1d")
            limit: Number of candles to fetch

        Returns:
            List of OHLCV data points
        """
        try:
            # Convert timeframe to Alpaca format
            alpaca_timeframe = self._convert_timeframe(timeframe)

            # Calculate start time based on limit
            end = datetime.now()
            if "m" in timeframe or "Min" in timeframe:
                minutes = int(timeframe.replace("m", "").replace("Min", ""))
                start = end - timedelta(minutes=minutes * limit)
            elif "h" in timeframe or "Hour" in timeframe:
                hours = int(timeframe.replace("h", "").replace("Hour", ""))
                start = end - timedelta(hours=hours * limit)
            elif "d" in timeframe or "Day" in timeframe:
                days = int(timeframe.replace("d", "").replace("Day", ""))
                start = end - timedelta(days=days * limit)
            else:
                start = end - timedelta(days=limit)

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_timeframe,
                start=start,
                end=end,
                limit=limit,
            )

            bars = self.data_client.get_stock_bars(request)

            ohlcv_list = []
            if symbol in bars:
                for bar in bars[symbol]:
                    ohlcv_list.append(OHLCV(
                        timestamp=int(bar.timestamp.timestamp() * 1000),
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=float(bar.volume),
                    ))

            return ohlcv_list[-limit:] if len(ohlcv_list) > limit else ohlcv_list

        except Exception as e:
            self.logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            raise

    async def fetch_ticker(self, symbol: str) -> Ticker:
        """Fetch current ticker data from Alpaca.

        Args:
            symbol: Stock symbol

        Returns:
            Ticker with current prices
        """
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)

            quote = quotes[symbol]

            return Ticker(
                symbol=symbol,
                bid=float(quote.bid_price),
                ask=float(quote.ask_price),
                last=float(quote.ask_price),  # Use ask as last for stocks
                volume=float(quote.bid_size + quote.ask_size),
                timestamp=int(quote.timestamp.timestamp() * 1000),
            )

        except Exception as e:
            self.logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            raise

    async def fetch_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """Fetch order book from Alpaca.

        Note: Alpaca doesn't provide full order book data like crypto exchanges.
        This returns a simplified book with bid/ask from latest quote.

        Args:
            symbol: Stock symbol
            limit: Order book depth (ignored for Alpaca)

        Returns:
            OrderBook with best bid and ask
        """
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)

            quote = quotes[symbol]

            # Alpaca doesn't provide full order book, just best bid/ask
            return OrderBook(
                symbol=symbol,
                bids=[(float(quote.bid_price), float(quote.bid_size))],
                asks=[(float(quote.ask_price), float(quote.ask_size))],
                timestamp=int(quote.timestamp.timestamp() * 1000),
            )

        except Exception as e:
            self.logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            raise

    async def fetch_balance(self) -> Balance:
        """Fetch account balance from Alpaca.

        Returns:
            Balance (USD for stock trading)
        """
        try:
            account = self.trading_client.get_account()

            return Balance(
                currency="USD",
                free=float(account.buying_power),
                used=float(account.equity) - float(account.buying_power),
                total=float(account.equity),
            )

        except Exception as e:
            self.logger.error(f"Failed to fetch balance: {e}")
            raise

    async def fetch_positions(self) -> list[Position]:
        """Fetch open positions from Alpaca.

        Returns:
            List of open stock positions
        """
        try:
            alpaca_positions = self.trading_client.get_all_positions()

            positions = []
            for pos in alpaca_positions:
                side = PositionSide.LONG if float(pos.qty) > 0 else PositionSide.SHORT

                positions.append(Position(
                    symbol=pos.symbol,
                    side=side,
                    size=abs(float(pos.qty)),
                    entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    unrealized_pnl=float(pos.unrealized_pl),
                    leverage=1.0,  # Stocks don't use leverage like futures
                ))

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
        """Create an order on Alpaca.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            order_type: "market" or "limit"
            amount: Order size (number of shares)
            price: Limit price (required for limit orders)

        Returns:
            Created order
        """
        try:
            # Convert side to Alpaca enum
            alpaca_side = AlpacaOrderSide.BUY if side.lower() == "buy" else AlpacaOrderSide.SELL

            # Create order request based on type
            if order_type.lower() == "market":
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=amount,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                )
            elif order_type.lower() == "limit":
                if price is None:
                    raise ValueError("Price is required for limit orders")

                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=amount,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=price,
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            # Submit order
            alpaca_order = self.trading_client.submit_order(order_request)

            # Convert to unified Order model
            return Order(
                id=alpaca_order.id,
                symbol=alpaca_order.symbol,
                side=OrderSide.BUY if alpaca_order.side == AlpacaOrderSide.BUY else OrderSide.SELL,
                order_type=OrderType.MARKET if order_type.lower() == "market" else OrderType.LIMIT,
                amount=float(alpaca_order.qty),
                price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                status=self._convert_order_status(alpaca_order.status),
                filled=float(alpaca_order.filled_qty) if alpaca_order.filled_qty else 0.0,
                timestamp=int(alpaca_order.created_at.timestamp() * 1000),
            )

        except Exception as e:
            self.logger.error(f"Failed to create order: {e}")
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order on Alpaca.

        Args:
            order_id: Order ID to cancel
            symbol: Stock symbol (not used by Alpaca, but kept for interface)

        Returns:
            True if canceled successfully
        """
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_provider_name(self) -> str:
        """Return provider name.

        Returns:
            "alpaca"
        """
        return "alpaca"

    async def close(self):
        """Close connection (cleanup).

        Note: alpaca-py client doesn't require explicit cleanup.
        """
        self.logger.info("Alpaca provider closed")

    def _convert_timeframe(self, timeframe: str) -> TimeFrame:
        """Convert standard timeframe to Alpaca TimeFrame.

        Args:
            timeframe: Standard format (e.g., "5m", "1h", "1d")

        Returns:
            Alpaca TimeFrame object
        """
        # Parse timeframe
        if timeframe.endswith("m") or timeframe.endswith("Min"):
            amount = int(timeframe.replace("m", "").replace("Min", ""))
            return TimeFrame(amount, TimeFrameUnit.Minute)
        elif timeframe.endswith("h") or timeframe.endswith("Hour"):
            amount = int(timeframe.replace("h", "").replace("Hour", ""))
            return TimeFrame(amount, TimeFrameUnit.Hour)
        elif timeframe.endswith("d") or timeframe.endswith("Day"):
            amount = int(timeframe.replace("d", "").replace("Day", ""))
            return TimeFrame(amount, TimeFrameUnit.Day)
        else:
            # Default to 1 day
            return TimeFrame(1, TimeFrameUnit.Day)

    def _convert_order_status(self, alpaca_status: str) -> OrderStatus:
        """Convert Alpaca order status to unified OrderStatus.

        Args:
            alpaca_status: Alpaca status string

        Returns:
            Unified OrderStatus enum
        """
        status_map = {
            "new": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "done_for_day": OrderStatus.CANCELED,
            "canceled": OrderStatus.CANCELED,
            "expired": OrderStatus.CANCELED,
            "replaced": OrderStatus.CANCELED,
            "pending_cancel": OrderStatus.PENDING,
            "pending_replace": OrderStatus.PENDING,
            "accepted": OrderStatus.PENDING,
            "pending_new": OrderStatus.PENDING,
            "accepted_for_bidding": OrderStatus.PENDING,
            "stopped": OrderStatus.CANCELED,
            "rejected": OrderStatus.FAILED,
            "suspended": OrderStatus.PENDING,
            "calculated": OrderStatus.PENDING,
        }

        return status_map.get(alpaca_status.lower(), OrderStatus.PENDING)
