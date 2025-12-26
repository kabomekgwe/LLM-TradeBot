"""Paper trading provider implementation.

Simulates trading with virtual balance using real market data.
No API keys required - perfect for testing strategies safely.
"""

import logging
import asyncio
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, field
import random

try:
    import ccxt.async_support as ccxt
except ImportError:
    raise ImportError(
        "ccxt library is required for paper trading (uses public data). "
        "Install with: pip install ccxt>=4.0.0"
    )

from ..config import TradingConfig
from .base import BaseExchangeProvider
from ..models.market_data import OHLCV, Ticker, OrderBook, Balance
from ..models.positions import (
    Position,
    Order,
    PositionSide,
    OrderSide,
    OrderType,
    OrderStatus,
)


@dataclass
class PaperPosition:
    """Internal paper trading position tracking."""

    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    leverage: float = 1.0

    def to_position(self, current_price: float) -> Position:
        """Convert to unified Position model.

        Args:
            current_price: Current market price

        Returns:
            Position with calculated unrealized PnL
        """
        if self.side == PositionSide.LONG:
            unrealized_pnl = (current_price - self.entry_price) * self.size
        else:  # SHORT
            unrealized_pnl = (self.entry_price - current_price) * self.size

        return Position(
            symbol=self.symbol,
            side=self.side,
            size=self.size,
            entry_price=self.entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            leverage=self.leverage,
        )


@dataclass
class PaperOrder:
    """Internal paper trading order tracking."""

    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: Optional[float]
    status: OrderStatus
    filled: float = 0.0
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1000))

    def to_order(self) -> Order:
        """Convert to unified Order model."""
        return Order(
            id=self.id,
            symbol=self.symbol,
            side=self.side,
            order_type=self.order_type,
            amount=self.amount,
            price=self.price,
            status=self.status,
            filled=self.filled,
            timestamp=self.timestamp,
        )


class PaperProvider(BaseExchangeProvider):
    """Paper trading provider using simulated execution.

    Uses real market data from Binance public API but executes trades
    in a virtual environment. Perfect for strategy testing without risk.

    Features:
    - Real-time market data
    - Simulated order execution with slippage
    - Virtual balance tracking
    - No API keys required
    """

    def __init__(self, config: TradingConfig):
        """Initialize paper trading provider.

        Args:
            config: Trading configuration (API keys not required)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Use Binance public API for real market data (no authentication)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},  # Use futures data
        })

        # Virtual trading state
        self.virtual_balance_usd = config.max_position_size_usd * 10  # Start with 10x max position
        self.virtual_positions: dict[str, PaperPosition] = {}
        self.virtual_orders: dict[str, PaperOrder] = {}
        self.order_counter = 0
        self.total_trades = 0
        self.winning_trades = 0

        # Simulation parameters
        self.slippage_pct = 0.05  # 0.05% slippage simulation
        self.maker_fee_pct = 0.02  # 0.02% maker fee
        self.taker_fee_pct = 0.04  # 0.04% taker fee

        self.logger.info(
            f"Paper Trading initialized with ${self.virtual_balance_usd:.2f} virtual balance"
        )

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> list[OHLCV]:
        """Fetch real OHLCV data from Binance public API.

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
        """Fetch real ticker data from Binance public API.

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
        """Fetch real order book from Binance public API.

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
        """Fetch virtual account balance.

        Returns:
            Virtual balance (USDT)
        """
        # Calculate used balance (margin in open positions)
        used_balance = 0.0
        for pos in self.virtual_positions.values():
            used_balance += pos.entry_price * pos.size / pos.leverage

        free_balance = self.virtual_balance_usd - used_balance

        return Balance(
            currency="USDT",
            free=free_balance,
            used=used_balance,
            total=self.virtual_balance_usd,
        )

    async def fetch_positions(self) -> list[Position]:
        """Fetch virtual open positions.

        Returns:
            List of virtual positions with current prices
        """
        positions = []

        for pos in self.virtual_positions.values():
            # Get current price
            try:
                ticker = await self.fetch_ticker(pos.symbol)
                current_price = ticker.last

                positions.append(pos.to_position(current_price))

            except Exception as e:
                self.logger.error(f"Failed to get current price for {pos.symbol}: {e}")
                # Use entry price as fallback
                positions.append(pos.to_position(pos.entry_price))

        return positions

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Order:
        """Create a simulated order.

        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            order_type: "market" or "limit"
            amount: Order size
            price: Limit price (required for limit orders)

        Returns:
            Simulated order
        """
        try:
            # Validate inputs
            if order_type.lower() == "limit" and price is None:
                raise ValueError("Price is required for limit orders")

            # Generate order ID
            self.order_counter += 1
            order_id = f"paper_{self.order_counter}"

            # Convert to enums
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            order_type_enum = (
                OrderType.MARKET if order_type.lower() == "market" else OrderType.LIMIT
            )

            # Get current market price
            ticker = await self.fetch_ticker(symbol)
            market_price = ticker.last

            # Simulate order execution
            if order_type.lower() == "market":
                # Market orders execute immediately with slippage
                execution_price = await self._simulate_market_execution(
                    market_price, order_side
                )

                # Create filled order
                paper_order = PaperOrder(
                    id=order_id,
                    symbol=symbol,
                    side=order_side,
                    order_type=order_type_enum,
                    amount=amount,
                    price=execution_price,
                    status=OrderStatus.FILLED,
                    filled=amount,
                )

                # Update virtual positions
                await self._update_position(symbol, order_side, amount, execution_price)

                self.logger.info(
                    f"Paper Market Order FILLED: {side} {amount} {symbol} @ ${execution_price:.2f}"
                )

            else:
                # Limit orders are pending (would need price monitoring to fill)
                paper_order = PaperOrder(
                    id=order_id,
                    symbol=symbol,
                    side=order_side,
                    order_type=order_type_enum,
                    amount=amount,
                    price=price,
                    status=OrderStatus.PENDING,
                    filled=0.0,
                )

                self.logger.info(
                    f"Paper Limit Order CREATED: {side} {amount} {symbol} @ ${price:.2f}"
                )

            # Store order
            self.virtual_orders[order_id] = paper_order

            return paper_order.to_order()

        except Exception as e:
            self.logger.error(f"Failed to create paper order: {e}")
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a simulated order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair

        Returns:
            True if canceled successfully
        """
        if order_id in self.virtual_orders:
            order = self.virtual_orders[order_id]

            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELED
                self.logger.info(f"Paper order {order_id} canceled")
                return True
            else:
                self.logger.warning(
                    f"Cannot cancel order {order_id} - status: {order.status}"
                )
                return False
        else:
            self.logger.error(f"Order {order_id} not found")
            return False

    def get_provider_name(self) -> str:
        """Return provider name.

        Returns:
            "paper"
        """
        return "paper"

    async def close(self):
        """Close exchange connection (cleanup)."""
        if self.exchange:
            await self.exchange.close()

        # Log final performance
        win_rate = (
            (self.winning_trades / self.total_trades * 100)
            if self.total_trades > 0
            else 0.0
        )

        self.logger.info(
            f"Paper Trading Session Summary:\n"
            f"  Total Trades: {self.total_trades}\n"
            f"  Win Rate: {win_rate:.1f}%\n"
            f"  Final Balance: ${self.virtual_balance_usd:.2f}"
        )

    async def _simulate_market_execution(
        self, market_price: float, side: OrderSide
    ) -> float:
        """Simulate market order execution with slippage.

        Args:
            market_price: Current market price
            side: Order side (BUY/SELL)

        Returns:
            Simulated execution price
        """
        # Simulate slippage (random between 0 and max slippage)
        slippage = random.uniform(0, self.slippage_pct / 100)

        if side == OrderSide.BUY:
            # Buying - price slips up
            execution_price = market_price * (1 + slippage)
        else:
            # Selling - price slips down
            execution_price = market_price * (1 - slippage)

        return execution_price

    async def _update_position(
        self, symbol: str, side: OrderSide, amount: float, price: float
    ) -> None:
        """Update virtual position after trade execution.

        Args:
            symbol: Trading pair
            side: Order side
            amount: Trade size
            price: Execution price
        """
        position_side = PositionSide.LONG if side == OrderSide.BUY else PositionSide.SHORT

        if symbol in self.virtual_positions:
            # Modify existing position
            existing = self.virtual_positions[symbol]

            if existing.side == position_side:
                # Add to position - calculate new average entry
                total_size = existing.size + amount
                new_entry = (
                    (existing.entry_price * existing.size + price * amount) / total_size
                )

                existing.size = total_size
                existing.entry_price = new_entry

                self.logger.info(f"Increased {position_side} position in {symbol}")
            else:
                # Reduce/close/reverse position
                if amount >= existing.size:
                    # Close or reverse
                    pnl = self._calculate_pnl(existing, price)
                    self.virtual_balance_usd += pnl

                    if pnl > 0:
                        self.winning_trades += 1
                    self.total_trades += 1

                    if amount > existing.size:
                        # Reverse position
                        new_size = amount - existing.size
                        self.virtual_positions[symbol] = PaperPosition(
                            symbol=symbol,
                            side=position_side,
                            size=new_size,
                            entry_price=price,
                        )
                        self.logger.info(
                            f"Closed and reversed position in {symbol}, PnL: ${pnl:.2f}"
                        )
                    else:
                        # Fully closed
                        del self.virtual_positions[symbol]
                        self.logger.info(
                            f"Closed position in {symbol}, PnL: ${pnl:.2f}"
                        )
                else:
                    # Partial close
                    pnl = self._calculate_pnl(existing, price) * (
                        amount / existing.size
                    )
                    self.virtual_balance_usd += pnl

                    existing.size -= amount

                    if pnl > 0:
                        self.winning_trades += 1
                    self.total_trades += 1

                    self.logger.info(
                        f"Reduced position in {symbol}, PnL: ${pnl:.2f}"
                    )
        else:
            # New position
            self.virtual_positions[symbol] = PaperPosition(
                symbol=symbol,
                side=position_side,
                size=amount,
                entry_price=price,
            )

            self.logger.info(f"Opened new {position_side} position in {symbol}")

    def _calculate_pnl(self, position: PaperPosition, exit_price: float) -> float:
        """Calculate realized PnL for a position.

        Args:
            position: Position to close
            exit_price: Exit price

        Returns:
            Realized PnL in USD
        """
        if position.side == PositionSide.LONG:
            pnl = (exit_price - position.entry_price) * position.size
        else:  # SHORT
            pnl = (position.entry_price - exit_price) * position.size

        # Deduct fees
        entry_fee = position.entry_price * position.size * (self.taker_fee_pct / 100)
        exit_fee = exit_price * position.size * (self.taker_fee_pct / 100)
        pnl -= entry_fee + exit_fee

        return pnl
