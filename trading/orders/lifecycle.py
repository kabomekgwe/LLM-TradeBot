"""Order Lifecycle Management - Track and manage order state transitions.

This module handles the complete lifecycle of trading orders from creation to completion,
including status tracking, error handling, and parent-child relationships.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import asyncio

from ..models.positions import Order, OrderStatus, OrderSide
from .types import (
    BracketOrder,
    TrailingStopOrder,
    OCOOrder,
    OrderGroup,
    AdvancedOrderType,
    OrderRelation,
)


@dataclass
class OrderEvent:
    """Event representing a state change in order lifecycle."""

    order_id: str
    event_type: str  # "created", "filled", "partially_filled", "canceled", "rejected"
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class OrderLifecycleManager:
    """Manages the lifecycle of orders from creation to completion.

    Tracks order status, handles parent-child relationships, and triggers
    callbacks on state transitions.

    Features:
    - Order status tracking
    - Event history
    - Parent-child order relationships (bracket orders)
    - Automatic cancellation of child orders
    - Callback hooks for order events

    Example:
        >>> manager = OrderLifecycleManager()
        >>> manager.register_order(order)
        >>> manager.on_order_filled(order_id, handle_fill)
        >>> await manager.mark_filled(order_id)
    """

    def __init__(self):
        """Initialize lifecycle manager."""
        self.logger = logging.getLogger(__name__)

        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.order_events: Dict[str, List[OrderEvent]] = {}
        self.order_groups: Dict[str, OrderGroup] = {}

        # Callbacks
        self.fill_callbacks: Dict[str, List[Callable]] = {}
        self.cancel_callbacks: Dict[str, List[Callable]] = {}
        self.reject_callbacks: Dict[str, List[Callable]] = {}

        # Relationships (parent -> children)
        self.parent_child_map: Dict[str, List[str]] = {}
        self.child_parent_map: Dict[str, str] = {}

    def register_order(self, order: Order, parent_id: Optional[str] = None):
        """Register an order for lifecycle tracking.

        Args:
            order: Order to track
            parent_id: Optional parent order ID for linked orders
        """
        self.orders[order.id] = order
        self.order_events[order.id] = []

        # Track parent-child relationship
        if parent_id:
            self.child_parent_map[order.id] = parent_id
            if parent_id not in self.parent_child_map:
                self.parent_child_map[parent_id] = []
            self.parent_child_map[parent_id].append(order.id)

        # Log creation event
        self._add_event(order.id, "created", {
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.order_type.value,
            "amount": order.amount,
        })

        self.logger.info(f"Registered order {order.id} for {order.symbol}")

    def register_bracket(self, bracket: BracketOrder, group_id: str):
        """Register a bracket order group.

        Args:
            bracket: BracketOrder to track
            group_id: Unique group identifier
        """
        group = OrderGroup(
            group_id=group_id,
            parent_order_id=bracket.entry_order_id,
            child_order_ids=[
                bracket.stop_loss_order_id,
                bracket.take_profit_order_id,
            ] if bracket.stop_loss_order_id and bracket.take_profit_order_id else [],
            group_type=AdvancedOrderType.BRACKET,
            symbol=bracket.entry_order.symbol,
            total_risk_usd=bracket.total_risk_usd,
            total_reward_usd=bracket.total_reward_usd,
        )

        self.order_groups[group_id] = group
        self.logger.info(
            f"Registered bracket order group {group_id}: "
            f"Entry={bracket.entry_order_id}, SL={bracket.stop_loss_order_id}, "
            f"TP={bracket.take_profit_order_id}"
        )

    async def mark_filled(
        self,
        order_id: str,
        filled_price: Optional[float] = None,
        filled_amount: Optional[float] = None,
    ):
        """Mark an order as filled.

        Args:
            order_id: Order ID to mark as filled
            filled_price: Execution price
            filled_amount: Amount filled (if partial)
        """
        if order_id not in self.orders:
            self.logger.warning(f"Order {order_id} not found in lifecycle manager")
            return

        order = self.orders[order_id]

        # Update order status
        if filled_amount and filled_amount < order.amount:
            order.status = OrderStatus.PARTIALLY_FILLED
            order.filled = filled_amount
            event_type = "partially_filled"
        else:
            order.status = OrderStatus.FILLED
            order.filled = order.amount
            event_type = "filled"

        # Log event
        self._add_event(order_id, event_type, {
            "filled_price": filled_price,
            "filled_amount": order.filled,
        })

        self.logger.info(
            f"Order {order_id} {event_type}: {order.filled}/{order.amount} @ ${filled_price}"
        )

        # Trigger callbacks
        await self._trigger_callbacks(self.fill_callbacks, order_id, order)

        # If parent order filled, cancel all child orders
        if order.status == OrderStatus.FILLED:
            await self._handle_parent_filled(order_id)

    async def mark_canceled(self, order_id: str, reason: str = ""):
        """Mark an order as canceled.

        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason
        """
        if order_id not in self.orders:
            self.logger.warning(f"Order {order_id} not found in lifecycle manager")
            return

        order = self.orders[order_id]
        order.status = OrderStatus.CANCELED

        self._add_event(order_id, "canceled", {"reason": reason})
        self.logger.info(f"Order {order_id} canceled: {reason}")

        # Trigger callbacks
        await self._trigger_callbacks(self.cancel_callbacks, order_id, order)

    async def mark_rejected(self, order_id: str, reason: str = ""):
        """Mark an order as rejected.

        Args:
            order_id: Order ID to mark as rejected
            reason: Rejection reason
        """
        if order_id not in self.orders:
            self.logger.warning(f"Order {order_id} not found in lifecycle manager")
            return

        order = self.orders[order_id]
        order.status = OrderStatus.REJECTED

        self._add_event(order_id, "rejected", {"reason": reason})
        self.logger.error(f"Order {order_id} rejected: {reason}")

        # Trigger callbacks
        await self._trigger_callbacks(self.reject_callbacks, order_id, order)

    async def _handle_parent_filled(self, parent_id: str):
        """Handle parent order fill - cancel all child orders.

        Args:
            parent_id: Parent order ID
        """
        # Check if this order has children
        if parent_id not in self.parent_child_map:
            return

        child_ids = self.parent_child_map[parent_id]
        self.logger.info(
            f"Parent order {parent_id} filled, canceling {len(child_ids)} child orders"
        )

        # Cancel all child orders
        for child_id in child_ids:
            if child_id in self.orders:
                child_order = self.orders[child_id]
                if child_order.status in (OrderStatus.OPEN, OrderStatus.PENDING):
                    await self.mark_canceled(
                        child_id,
                        reason=f"Parent order {parent_id} filled"
                    )

    async def handle_oco_fill(self, oco: OCOOrder, filled_order_id: str):
        """Handle OCO order fill - cancel the other order.

        Args:
            oco: OCOOrder instance
            filled_order_id: Which order was filled
        """
        # Determine which order to cancel
        if filled_order_id == oco.primary_order_id:
            cancel_id = oco.secondary_order_id
            self.logger.info(f"OCO primary order {filled_order_id} filled, canceling secondary {cancel_id}")
        elif filled_order_id == oco.secondary_order_id:
            cancel_id = oco.primary_order_id
            self.logger.info(f"OCO secondary order {filled_order_id} filled, canceling primary {cancel_id}")
        else:
            self.logger.warning(f"OCO fill for unknown order {filled_order_id}")
            return

        # Cancel the other order
        if cancel_id:
            await self.mark_canceled(
                cancel_id,
                reason=f"OCO counterpart {filled_order_id} filled"
            )

        # Update OCO status
        oco.status = "filled"
        oco.filled_order_id = filled_order_id

    def on_order_filled(self, order_id: str, callback: Callable):
        """Register callback for order fill event.

        Args:
            order_id: Order ID to watch
            callback: Async function to call when filled
        """
        if order_id not in self.fill_callbacks:
            self.fill_callbacks[order_id] = []
        self.fill_callbacks[order_id].append(callback)

    def on_order_canceled(self, order_id: str, callback: Callable):
        """Register callback for order cancellation.

        Args:
            order_id: Order ID to watch
            callback: Async function to call when canceled
        """
        if order_id not in self.cancel_callbacks:
            self.cancel_callbacks[order_id] = []
        self.cancel_callbacks[order_id].append(callback)

    def on_order_rejected(self, order_id: str, callback: Callable):
        """Register callback for order rejection.

        Args:
            order_id: Order ID to watch
            callback: Async function to call when rejected
        """
        if order_id not in self.reject_callbacks:
            self.reject_callbacks[order_id] = []
        self.reject_callbacks[order_id].append(callback)

    async def _trigger_callbacks(
        self,
        callback_dict: Dict[str, List[Callable]],
        order_id: str,
        order: Order,
    ):
        """Trigger all registered callbacks for an order event.

        Args:
            callback_dict: Dictionary of callbacks
            order_id: Order ID
            order: Order object
        """
        if order_id not in callback_dict:
            return

        callbacks = callback_dict[order_id]
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order)
                else:
                    callback(order)
            except Exception as e:
                self.logger.error(f"Callback error for order {order_id}: {e}")

    def _add_event(self, order_id: str, event_type: str, details: dict):
        """Add event to order history.

        Args:
            order_id: Order ID
            event_type: Event type
            details: Event details
        """
        event = OrderEvent(
            order_id=order_id,
            event_type=event_type,
            details=details,
        )

        if order_id not in self.order_events:
            self.order_events[order_id] = []
        self.order_events[order_id].append(event)

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get current order status.

        Args:
            order_id: Order ID

        Returns:
            OrderStatus or None if not found
        """
        if order_id not in self.orders:
            return None
        return self.orders[order_id].status

    def get_order_events(self, order_id: str) -> List[OrderEvent]:
        """Get event history for an order.

        Args:
            order_id: Order ID

        Returns:
            List of OrderEvent objects
        """
        return self.order_events.get(order_id, [])

    def get_child_orders(self, parent_id: str) -> List[str]:
        """Get child order IDs for a parent order.

        Args:
            parent_id: Parent order ID

        Returns:
            List of child order IDs
        """
        return self.parent_child_map.get(parent_id, [])

    def get_parent_order(self, child_id: str) -> Optional[str]:
        """Get parent order ID for a child order.

        Args:
            child_id: Child order ID

        Returns:
            Parent order ID or None
        """
        return self.child_parent_map.get(child_id)

    def get_active_orders(self) -> List[Order]:
        """Get all active orders.

        Returns:
            List of orders with OPEN or PENDING status
        """
        return [
            order for order in self.orders.values()
            if order.status in (OrderStatus.OPEN, OrderStatus.PENDING)
        ]

    def get_group_status(self, group_id: str) -> Optional[str]:
        """Get status of an order group.

        Args:
            group_id: Group ID

        Returns:
            Group status or None if not found
        """
        if group_id not in self.order_groups:
            return None
        return self.order_groups[group_id].status

    def cleanup_completed_orders(self, retention_hours: int = 24):
        """Remove completed orders older than retention period.

        Args:
            retention_hours: Hours to retain completed orders
        """
        cutoff_time = datetime.now().timestamp() - (retention_hours * 3600)

        orders_to_remove = []
        for order_id, order in self.orders.items():
            if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
                if order.timestamp < cutoff_time * 1000:  # Convert to milliseconds
                    orders_to_remove.append(order_id)

        for order_id in orders_to_remove:
            del self.orders[order_id]
            if order_id in self.order_events:
                del self.order_events[order_id]

        self.logger.info(f"Cleaned up {len(orders_to_remove)} completed orders")

    def get_lifecycle_summary(self) -> dict:
        """Get summary of all tracked orders.

        Returns:
            Dictionary with order statistics
        """
        total = len(self.orders)
        by_status = {}

        for order in self.orders.values():
            status = order.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_orders": total,
            "by_status": by_status,
            "active_groups": len(self.order_groups),
            "parent_child_relationships": len(self.parent_child_map),
        }
