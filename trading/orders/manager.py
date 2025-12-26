"""Order Manager - Unified interface for advanced order types.

This module provides the OrderManager class that handles creation and management
of bracket orders, trailing stops, OCO orders, and other advanced order types.

Integrates with the exchange provider layer and lifecycle management.
"""

import logging
import uuid
from typing import Optional, Tuple, List
from datetime import datetime

from ..providers.base import BaseExchangeProvider
from ..models.positions import Order, OrderSide, OrderType, OrderStatus
from ..models.market_data import OHLCV, Ticker
from ..config import TradingConfig

from .types import (
    BracketOrder,
    TrailingStopOrder,
    OCOOrder,
    OrderGroup,
    TrailingType,
    AdvancedOrderType,
)
from .lifecycle import OrderLifecycleManager


class OrderManager:
    """Unified order management with advanced order types.

    Provides high-level API for creating bracket orders, trailing stops,
    and OCO orders. Handles coordination with exchange providers and
    lifecycle tracking.

    Features:
    - Bracket orders (entry + stop loss + take profit)
    - Trailing stop orders
    - OCO (One-Cancels-Other) orders
    - Automatic risk validation
    - Order lifecycle tracking
    - Position size calculation

    Example:
        >>> manager = OrderManager(provider, config)
        >>> entry, stop, tp = await manager.create_bracket_order(
        ...     symbol="BTC/USDT",
        ...     side="buy",
        ...     amount=0.1,
        ...     entry_price=42000,
        ...     stop_loss_price=41000,
        ...     take_profit_price=44000
        ... )
    """

    def __init__(
        self,
        provider: BaseExchangeProvider,
        config: Optional[TradingConfig] = None,
    ):
        """Initialize order manager.

        Args:
            provider: Exchange provider instance
            config: Trading configuration
        """
        self.provider = provider
        self.config = config or TradingConfig()
        self.logger = logging.getLogger(__name__)

        # Lifecycle management
        self.lifecycle = OrderLifecycleManager()

        # Active trailing stops (need price updates)
        self.trailing_stops: dict[str, TrailingStopOrder] = {}

        # Active OCO orders
        self.oco_orders: dict[str, OCOOrder] = {}

        self.logger.info(f"OrderManager initialized with {provider.get_provider_name()}")

    async def create_bracket_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        trailing_stop: bool = False,
        trailing_delta: Optional[float] = None,
        order_type: str = "limit",
    ) -> Tuple[Order, Order, Order]:
        """Create a bracket order (entry + stop loss + take profit).

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            side: "buy" or "sell"
            amount: Position size
            entry_price: Entry limit price
            stop_loss_price: Stop-loss trigger price
            take_profit_price: Take-profit trigger price
            trailing_stop: Use trailing stop instead of fixed stop
            trailing_delta: Trailing distance (% if < 1, else absolute)
            order_type: "market" or "limit" for entry

        Returns:
            Tuple of (entry_order, stop_loss_order, take_profit_order)

        Example:
            >>> entry, sl, tp = await manager.create_bracket_order(
            ...     "BTC/USDT", "buy", 0.1, 42000, 41000, 44000
            ... )
            >>> entry.id
            '12345'
        """
        # Create entry order
        entry_order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            order_type=OrderType.LIMIT if order_type == "limit" else OrderType.MARKET,
            amount=amount,
            price=entry_price if order_type == "limit" else None,
            status=OrderStatus.PENDING,
        )

        # Create bracket order structure
        bracket = BracketOrder(
            entry_order=entry_order,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            trailing_stop=trailing_stop,
            trailing_delta=trailing_delta,
        )

        # Validate bracket
        is_valid, error_msg = bracket.validate()
        if not is_valid:
            raise ValueError(f"Invalid bracket order: {error_msg}")

        self.logger.info(
            f"Creating bracket order for {symbol}: Entry ${entry_price}, "
            f"SL ${stop_loss_price}, TP ${take_profit_price}, "
            f"R:R {bracket.risk_reward_ratio:.2f}"
        )

        # Create entry order on exchange
        try:
            executed_entry = await self.provider.create_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                amount=amount,
                price=entry_price if order_type == "limit" else None,
            )

            bracket.entry_order_id = executed_entry.id
            self.lifecycle.register_order(executed_entry)

            # Create stop-loss order (conditional on entry fill)
            stop_order = await self._create_stop_loss_order(
                symbol=symbol,
                side="sell" if side == "buy" else "buy",
                amount=amount,
                stop_price=stop_loss_price,
                parent_id=executed_entry.id,
            )

            bracket.stop_loss_order_id = stop_order.id

            # Create take-profit order (conditional on entry fill)
            tp_order = await self._create_take_profit_order(
                symbol=symbol,
                side="sell" if side == "buy" else "buy",
                amount=amount,
                take_profit_price=take_profit_price,
                parent_id=executed_entry.id,
            )

            bracket.take_profit_order_id = tp_order.id

            # Register bracket order group
            group_id = f"bracket_{executed_entry.id}"
            self.lifecycle.register_bracket(bracket, group_id)

            # Set up callbacks for parent order fill
            self.lifecycle.on_order_filled(
                executed_entry.id,
                lambda order: self._handle_bracket_entry_fill(bracket, order),
            )

            self.logger.info(
                f"Bracket order created: Entry={executed_entry.id}, "
                f"SL={stop_order.id}, TP={tp_order.id}"
            )

            return executed_entry, stop_order, tp_order

        except Exception as e:
            self.logger.error(f"Failed to create bracket order: {e}")
            raise

    async def _create_stop_loss_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
        parent_id: Optional[str] = None,
    ) -> Order:
        """Create a stop-loss order.

        Args:
            symbol: Trading symbol
            side: Order side
            amount: Order amount
            stop_price: Stop trigger price
            parent_id: Parent order ID

        Returns:
            Created stop-loss order
        """
        # Use provider's stop-loss order type if supported
        stop_order = await self.provider.create_order(
            symbol=symbol,
            side=side,
            order_type="stop_loss",
            amount=amount,
            price=None,  # Market order when triggered
            stop_price=stop_price,
        )

        self.lifecycle.register_order(stop_order, parent_id=parent_id)
        return stop_order

    async def _create_take_profit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        take_profit_price: float,
        parent_id: Optional[str] = None,
    ) -> Order:
        """Create a take-profit order.

        Args:
            symbol: Trading symbol
            side: Order side
            amount: Order amount
            take_profit_price: Take-profit trigger price
            parent_id: Parent order ID

        Returns:
            Created take-profit order
        """
        # Use provider's take-profit order type if supported
        tp_order = await self.provider.create_order(
            symbol=symbol,
            side=side,
            order_type="take_profit",
            amount=amount,
            price=None,  # Market order when triggered
            take_profit_price=take_profit_price,
        )

        self.lifecycle.register_order(tp_order, parent_id=parent_id)
        return tp_order

    async def _handle_bracket_entry_fill(self, bracket: BracketOrder, entry_order: Order):
        """Handle bracket entry order fill.

        When entry fills, activate stop-loss and take-profit orders.

        Args:
            bracket: BracketOrder instance
            entry_order: Filled entry order
        """
        self.logger.info(
            f"Bracket entry order {entry_order.id} filled, "
            f"activating SL and TP orders"
        )

        bracket.status = "active"

        # If trailing stop is enabled, create trailing stop order
        if bracket.trailing_stop and bracket.trailing_delta:
            await self.create_trailing_stop(
                symbol=entry_order.symbol,
                side="sell" if entry_order.side == OrderSide.BUY else "buy",
                amount=entry_order.amount,
                activation_price=entry_order.price or 0,
                trailing_delta=bracket.trailing_delta,
                trailing_type=bracket.trailing_type,
            )

    async def create_trailing_stop(
        self,
        symbol: str,
        side: str,
        amount: float,
        activation_price: float,
        trailing_delta: float,
        trailing_type: TrailingType = TrailingType.PERCENT,
    ) -> TrailingStopOrder:
        """Create a trailing stop order.

        Args:
            symbol: Trading symbol
            side: Order side (usually opposite of position)
            amount: Order amount
            activation_price: Price at which trailing starts
            trailing_delta: Trail distance (% if PERCENT, absolute if ABSOLUTE)
            trailing_type: PERCENT or ABSOLUTE

        Returns:
            TrailingStopOrder instance

        Example:
            >>> trailing = await manager.create_trailing_stop(
            ...     "BTC/USDT", "sell", 0.1, 42000, 2.0, TrailingType.PERCENT
            ... )
        """
        trailing_order = TrailingStopOrder(
            symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            amount=amount,
            activation_price=activation_price,
            trailing_delta=trailing_delta,
            trailing_type=trailing_type,
        )

        # Generate unique ID
        order_id = f"trailing_{uuid.uuid4().hex[:8]}"
        trailing_order.order_id = order_id
        trailing_order.status = "active"

        # Store for price updates
        self.trailing_stops[order_id] = trailing_order

        self.logger.info(
            f"Created trailing stop: {order_id} for {symbol}, "
            f"activation ${activation_price}, delta {trailing_delta} "
            f"({trailing_type.value})"
        )

        return trailing_order

    async def update_trailing_stops(self, current_prices: dict[str, float]):
        """Update all trailing stops with current prices.

        Should be called on every price tick.

        Args:
            current_prices: Dictionary of {symbol: current_price}
        """
        for order_id, trailing_order in list(self.trailing_stops.items()):
            symbol = trailing_order.symbol
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]

            # Update trailing stop price
            updated, new_stop = trailing_order.update_price(current_price)

            if updated:
                self.logger.debug(
                    f"Trailing stop {order_id} updated: "
                    f"new stop ${new_stop:.2f}"
                )

            # Check if triggered
            if trailing_order.is_triggered(current_price):
                self.logger.info(
                    f"Trailing stop {order_id} triggered at ${current_price:.2f}"
                )

                # Execute market order
                await self.provider.create_order(
                    symbol=symbol,
                    side=trailing_order.side.value,
                    order_type="market",
                    amount=trailing_order.amount,
                )

                # Remove from active trailing stops
                del self.trailing_stops[order_id]

    async def create_oco_order(
        self,
        symbol: str,
        amount: float,
        take_profit_price: float,
        stop_loss_price: float,
        side: str = "sell",
    ) -> OCOOrder:
        """Create an OCO (One-Cancels-Other) order.

        Typically used to exit a position with either take-profit or stop-loss.

        Args:
            symbol: Trading symbol
            amount: Order amount
            take_profit_price: Take-profit limit price
            stop_loss_price: Stop-loss trigger price
            side: Order side (usually "sell" for closing long)

        Returns:
            OCOOrder instance

        Example:
            >>> oco = await manager.create_oco_order(
            ...     "BTC/USDT", 0.1, 44000, 41000
            ... )
        """
        # Create primary order (take-profit limit)
        primary_order = await self.provider.create_order(
            symbol=symbol,
            side=side,
            order_type="limit",
            amount=amount,
            price=take_profit_price,
        )

        # Create secondary order (stop-loss)
        secondary_order = await self.provider.create_order(
            symbol=symbol,
            side=side,
            order_type="stop_loss",
            amount=amount,
            price=None,
            stop_price=stop_loss_price,
        )

        # Create OCO structure
        oco = OCOOrder(
            primary_order=primary_order,
            secondary_order=secondary_order,
            primary_order_id=primary_order.id,
            secondary_order_id=secondary_order.id,
        )

        # Validate
        is_valid, error_msg = oco.validate()
        if not is_valid:
            raise ValueError(f"Invalid OCO order: {error_msg}")

        # Register both orders
        self.lifecycle.register_order(primary_order)
        self.lifecycle.register_order(secondary_order)

        # Set up callbacks for cancellation
        self.lifecycle.on_order_filled(
            primary_order.id,
            lambda order: self.lifecycle.handle_oco_fill(oco, primary_order.id),
        )
        self.lifecycle.on_order_filled(
            secondary_order.id,
            lambda order: self.lifecycle.handle_oco_fill(oco, secondary_order.id),
        )

        # Store OCO
        oco_id = f"oco_{uuid.uuid4().hex[:8]}"
        self.oco_orders[oco_id] = oco
        oco.status = "active"

        self.logger.info(
            f"Created OCO order {oco_id}: TP={primary_order.id}, SL={secondary_order.id}"
        )

        return oco

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol

        Returns:
            True if successfully canceled
        """
        try:
            success = await self.provider.cancel_order(order_id, symbol)

            if success:
                await self.lifecycle.mark_canceled(order_id, "User canceled")
                self.logger.info(f"Order {order_id} canceled")
            else:
                self.logger.warning(f"Failed to cancel order {order_id}")

            return success

        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}")
            return False

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        risk_amount_usd: float,
    ) -> float:
        """Calculate position size based on risk parameters.

        Uses the formula: position_size = risk_amount / (entry_price - stop_price)

        Args:
            entry_price: Entry price
            stop_loss_price: Stop-loss price
            risk_amount_usd: Maximum risk in USD

        Returns:
            Position size in base currency

        Example:
            >>> size = manager.calculate_position_size(42000, 41000, 100)
            >>> size
            0.1  # Risk $100 with $1000 stop distance
        """
        risk_per_unit = abs(entry_price - stop_loss_price)

        if risk_per_unit == 0:
            raise ValueError("Stop loss cannot equal entry price")

        position_size = risk_amount_usd / risk_per_unit

        self.logger.debug(
            f"Position size: {position_size:.4f} for ${risk_amount_usd} risk "
            f"(entry ${entry_price}, stop ${stop_loss_price})"
        )

        return position_size

    def get_active_orders_summary(self) -> dict:
        """Get summary of all active orders.

        Returns:
            Dictionary with order counts and details
        """
        active_orders = self.lifecycle.get_active_orders()

        return {
            "total_active": len(active_orders),
            "trailing_stops": len(self.trailing_stops),
            "oco_orders": len(self.oco_orders),
            "lifecycle_summary": self.lifecycle.get_lifecycle_summary(),
        }
