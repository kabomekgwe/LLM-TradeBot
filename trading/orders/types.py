"""Advanced Order Type Definitions - Bracket, Trailing, OCO orders.

This module extends the basic order types with professional trading order types
used for risk management and automated execution.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from datetime import datetime

from ..models.positions import Order, OrderSide, OrderType


class AdvancedOrderType(str, Enum):
    """Advanced order type enumeration."""
    BRACKET = "bracket"  # Entry + Stop Loss + Take Profit
    TRAILING_STOP = "trailing_stop"  # Stop that trails price
    OCO = "oco"  # One-Cancels-Other
    ICEBERG = "iceberg"  # Large order split into smaller chunks
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price


class TrailingType(str, Enum):
    """Trailing stop type enumeration."""
    PERCENT = "percent"  # Trail by percentage
    ABSOLUTE = "absolute"  # Trail by absolute price distance


class OrderRelation(str, Enum):
    """Relationship between linked orders."""
    PARENT = "parent"  # Parent order (entry)
    STOP_LOSS = "stop_loss"  # Child stop-loss order
    TAKE_PROFIT = "take_profit"  # Child take-profit order
    OCO_SIBLING = "oco_sibling"  # OCO sibling order


@dataclass
class BracketOrder:
    """Bracket order - Entry with automatic stop-loss and take-profit.

    Professional risk management: sets entry, stop, and profit targets in one order.

    Example:
        >>> bracket = BracketOrder(
        ...     entry_order=entry,
        ...     stop_loss_price=95.0,
        ...     take_profit_price=105.0
        ... )
        >>> bracket.risk_reward_ratio
        2.0
    """

    entry_order: Order
    stop_loss_price: float
    take_profit_price: float

    # Optional: Stop-loss as trailing stop
    trailing_stop: bool = False
    trailing_delta: Optional[float] = None  # For trailing stop
    trailing_type: TrailingType = TrailingType.PERCENT

    # Order IDs after creation
    entry_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None

    # Status tracking
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, active, filled, canceled

    @property
    def risk_amount(self) -> float:
        """Calculate risk amount per unit."""
        entry = self.entry_order.price or 0
        if self.entry_order.side == OrderSide.BUY:
            return abs(entry - self.stop_loss_price)
        else:
            return abs(self.stop_loss_price - entry)

    @property
    def reward_amount(self) -> float:
        """Calculate reward amount per unit."""
        entry = self.entry_order.price or 0
        if self.entry_order.side == OrderSide.BUY:
            return abs(self.take_profit_price - entry)
        else:
            return abs(entry - self.take_profit_price)

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio."""
        if self.risk_amount == 0:
            return 0.0
        return self.reward_amount / self.risk_amount

    @property
    def total_risk_usd(self) -> float:
        """Total risk in USD."""
        return self.risk_amount * self.entry_order.amount

    @property
    def total_reward_usd(self) -> float:
        """Total potential reward in USD."""
        return self.reward_amount * self.entry_order.amount

    def validate(self) -> tuple[bool, str]:
        """Validate bracket order parameters.

        Returns:
            (is_valid, error_message)
        """
        entry = self.entry_order.price or 0

        if entry <= 0:
            return False, "Entry price must be positive"

        if self.entry_order.side == OrderSide.BUY:
            # BUY: stop < entry < take_profit
            if self.stop_loss_price >= entry:
                return False, "Stop loss must be below entry for BUY orders"
            if self.take_profit_price <= entry:
                return False, "Take profit must be above entry for BUY orders"
        else:  # SELL
            # SELL: take_profit < entry < stop
            if self.stop_loss_price <= entry:
                return False, "Stop loss must be above entry for SELL orders"
            if self.take_profit_price >= entry:
                return False, "Take profit must be below entry for SELL orders"

        # Check risk/reward ratio
        if self.risk_reward_ratio < 0.5:
            return False, f"Risk/reward ratio too low: {self.risk_reward_ratio:.2f}"

        return True, ""

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "entry_order": self.entry_order.to_dict(),
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "trailing_stop": self.trailing_stop,
            "trailing_delta": self.trailing_delta,
            "trailing_type": self.trailing_type.value,
            "entry_order_id": self.entry_order_id,
            "stop_loss_order_id": self.stop_loss_order_id,
            "take_profit_order_id": self.take_profit_order_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "risk_reward_ratio": self.risk_reward_ratio,
            "total_risk_usd": self.total_risk_usd,
            "total_reward_usd": self.total_reward_usd,
        }


@dataclass
class TrailingStopOrder:
    """Trailing stop order - Stop loss that follows price movement.

    Automatically adjusts stop price as market moves favorably.

    Example:
        >>> trailing = TrailingStopOrder(
        ...     symbol="BTC/USDT",
        ...     side=OrderSide.SELL,
        ...     amount=0.1,
        ...     activation_price=42000,
        ...     trailing_delta=500,  # $500 trail
        ...     trailing_type=TrailingType.ABSOLUTE
        ... )
    """

    symbol: str
    side: OrderSide  # BUY or SELL
    amount: float

    # Trailing parameters
    activation_price: float  # Price at which trailing starts
    trailing_delta: float  # Trail distance (% or absolute)
    trailing_type: TrailingType = TrailingType.PERCENT

    # Current state
    current_stop_price: Optional[float] = None
    highest_price: Optional[float] = None  # For SELL trailing
    lowest_price: Optional[float] = None  # For BUY trailing

    # Order tracking
    order_id: Optional[str] = None
    status: str = "pending"  # pending, active, triggered, canceled
    created_at: datetime = field(default_factory=datetime.now)

    def update_price(self, current_price: float) -> tuple[bool, Optional[float]]:
        """Update trailing stop based on current price.

        Args:
            current_price: Current market price

        Returns:
            (stop_updated, new_stop_price)
        """
        # Initialize tracking prices
        if self.highest_price is None:
            self.highest_price = current_price
        if self.lowest_price is None:
            self.lowest_price = current_price

        # Update high/low
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)

        # Check if activated
        if current_price < self.activation_price and self.side == OrderSide.SELL:
            return False, None
        if current_price > self.activation_price and self.side == OrderSide.BUY:
            return False, None

        # Calculate new stop price
        if self.side == OrderSide.SELL:
            # Trailing stop for LONG positions (SELL to close)
            if self.trailing_type == TrailingType.PERCENT:
                new_stop = self.highest_price * (1 - self.trailing_delta / 100)
            else:  # ABSOLUTE
                new_stop = self.highest_price - self.trailing_delta
        else:
            # Trailing stop for SHORT positions (BUY to close)
            if self.trailing_type == TrailingType.PERCENT:
                new_stop = self.lowest_price * (1 + self.trailing_delta / 100)
            else:  # ABSOLUTE
                new_stop = self.lowest_price + self.trailing_delta

        # Only update if stop moves favorably
        if self.current_stop_price is None:
            self.current_stop_price = new_stop
            return True, new_stop

        updated = False
        if self.side == OrderSide.SELL and new_stop > self.current_stop_price:
            self.current_stop_price = new_stop
            updated = True
        elif self.side == OrderSide.BUY and new_stop < self.current_stop_price:
            self.current_stop_price = new_stop
            updated = True

        return updated, self.current_stop_price if updated else None

    def is_triggered(self, current_price: float) -> bool:
        """Check if trailing stop is triggered.

        Args:
            current_price: Current market price

        Returns:
            True if stop should execute
        """
        if self.current_stop_price is None:
            return False

        if self.side == OrderSide.SELL:
            # Triggered when price drops below stop
            return current_price <= self.current_stop_price
        else:
            # Triggered when price rises above stop
            return current_price >= self.current_stop_price

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "amount": self.amount,
            "activation_price": self.activation_price,
            "trailing_delta": self.trailing_delta,
            "trailing_type": self.trailing_type.value,
            "current_stop_price": self.current_stop_price,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price,
            "order_id": self.order_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class OCOOrder:
    """One-Cancels-Other order - Two orders where filling one cancels the other.

    Common use: Take profit OR stop loss (whichever hits first).

    Example:
        >>> oco = OCOOrder(
        ...     primary_order=take_profit_order,
        ...     secondary_order=stop_loss_order
        ... )
    """

    primary_order: Order  # Usually take-profit
    secondary_order: Order  # Usually stop-loss

    # Order IDs after creation
    primary_order_id: Optional[str] = None
    secondary_order_id: Optional[str] = None

    # Status
    status: str = "pending"  # pending, active, filled, canceled
    filled_order_id: Optional[str] = None  # Which order filled
    created_at: datetime = field(default_factory=datetime.now)

    def validate(self) -> tuple[bool, str]:
        """Validate OCO order.

        Returns:
            (is_valid, error_message)
        """
        if self.primary_order.symbol != self.secondary_order.symbol:
            return False, "Both orders must be for the same symbol"

        if self.primary_order.amount != self.secondary_order.amount:
            return False, "Both orders must have the same amount"

        # Typically primary and secondary should be opposite sides or different order types
        if self.primary_order.side == self.secondary_order.side:
            if self.primary_order.order_type == self.secondary_order.order_type:
                return False, "OCO orders should have different types or sides"

        return True, ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "primary_order": self.primary_order.to_dict(),
            "secondary_order": self.secondary_order.to_dict(),
            "primary_order_id": self.primary_order_id,
            "secondary_order_id": self.secondary_order_id,
            "status": self.status,
            "filled_order_id": self.filled_order_id,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class OrderGroup:
    """Group of related orders (e.g., all orders for a bracket).

    Tracks parent-child relationships and lifecycle.
    """

    group_id: str
    parent_order_id: Optional[str] = None
    child_order_ids: List[str] = field(default_factory=list)
    group_type: AdvancedOrderType = AdvancedOrderType.BRACKET

    # Metadata
    symbol: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, active, complete, canceled

    # Risk tracking
    total_risk_usd: float = 0.0
    total_reward_usd: float = 0.0

    def add_child(self, order_id: str):
        """Add child order to group."""
        if order_id not in self.child_order_ids:
            self.child_order_ids.append(order_id)

    def cancel_all_children(self) -> List[str]:
        """Get list of child orders to cancel."""
        return self.child_order_ids.copy()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "group_id": self.group_id,
            "parent_order_id": self.parent_order_id,
            "child_order_ids": self.child_order_ids,
            "group_type": self.group_type.value,
            "symbol": self.symbol,
            "status": self.status,
            "total_risk_usd": self.total_risk_usd,
            "total_reward_usd": self.total_reward_usd,
            "created_at": self.created_at.isoformat(),
        }
