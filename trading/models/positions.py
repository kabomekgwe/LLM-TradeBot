"""Position and order tracking models.

Unified models for managing trading positions and orders across exchanges.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class PositionSide(str, Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """Unified order structure.

    Represents a trading order across all exchanges.
    """

    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float  # Order size
    price: Optional[float] = None  # Limit price (None for market orders)
    status: OrderStatus = OrderStatus.PENDING
    filled: float = 0.0  # Amount filled
    remaining: float = 0.0  # Amount remaining
    timestamp: int = 0  # Unix timestamp in milliseconds
    fee: float = 0.0  # Trading fee
    fee_currency: str = ""  # Fee currency

    def __post_init__(self):
        """Calculate remaining amount."""
        if self.remaining == 0.0:
            self.remaining = self.amount - self.filled

    @property
    def is_complete(self) -> bool:
        """Whether order is fully filled."""
        return self.status == OrderStatus.FILLED

    @property
    def fill_percentage(self) -> float:
        """Percentage of order filled."""
        return (self.filled / self.amount * 100) if self.amount > 0 else 0

    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1000) if self.timestamp else datetime.now()

    @classmethod
    def from_ccxt(cls, data: dict) -> "Order":
        """Create from ccxt order format.

        Args:
            data: ccxt order dict

        Returns:
            Order instance
        """
        return cls(
            id=str(data.get('id', '')),
            symbol=data.get('symbol', ''),
            side=OrderSide(data.get('side', 'buy')),
            order_type=OrderType(data.get('type', 'market')),
            amount=float(data.get('amount', 0)),
            price=float(data['price']) if data.get('price') else None,
            status=cls._map_ccxt_status(data.get('status', 'pending')),
            filled=float(data.get('filled', 0)),
            remaining=float(data.get('remaining', 0)),
            timestamp=int(data.get('timestamp', 0)),
            fee=float(data.get('fee', {}).get('cost', 0)),
            fee_currency=data.get('fee', {}).get('currency', ''),
        )

    @staticmethod
    def _map_ccxt_status(ccxt_status: str) -> OrderStatus:
        """Map ccxt status to OrderStatus enum."""
        status_map = {
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELED,
            'rejected': OrderStatus.REJECTED,
            'expired': OrderStatus.CANCELED,
        }
        return status_map.get(ccxt_status, OrderStatus.PENDING)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "amount": self.amount,
            "price": self.price,
            "status": self.status.value,
            "filled": self.filled,
            "remaining": self.remaining,
            "timestamp": self.timestamp,
            "fee": self.fee,
        }


@dataclass
class Position:
    """Unified position structure.

    Represents an open trading position across all exchanges.
    """

    symbol: str
    side: PositionSide
    size: float  # Position size (always positive)
    entry_price: float
    current_price: float
    unrealized_pnl: float  # Unrealized profit/loss
    realized_pnl: float = 0.0  # Realized profit/loss (from partial closes)
    timestamp: int = 0  # Position open timestamp
    leverage: float = 1.0  # Leverage multiplier
    liquidation_price: Optional[float] = None  # Liquidation price (for leveraged positions)

    @property
    def pnl_pct(self) -> float:
        """Profit/loss percentage."""
        if self.side == PositionSide.LONG:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            return ((self.entry_price - self.current_price) / self.entry_price) * 100

    @property
    def position_value(self) -> float:
        """Current position value in quote currency."""
        return self.size * self.current_price

    @property
    def is_profitable(self) -> bool:
        """Whether position is currently profitable."""
        return self.unrealized_pnl > 0

    @property
    def is_long(self) -> bool:
        """Whether position is long."""
        return self.side == PositionSide.LONG

    @property
    def is_short(self) -> bool:
        """Whether position is short."""
        return self.side == PositionSide.SHORT

    @classmethod
    def from_ccxt(cls, data: dict) -> "Position":
        """Create from ccxt position format.

        Args:
            data: ccxt position dict

        Returns:
            Position instance
        """
        contracts = float(data.get('contracts', 0))
        side = PositionSide.LONG if contracts > 0 else PositionSide.SHORT

        return cls(
            symbol=data.get('symbol', ''),
            side=side,
            size=abs(contracts),
            entry_price=float(data.get('entryPrice', 0)),
            current_price=float(data.get('markPrice', 0)),
            unrealized_pnl=float(data.get('unrealizedPnl', 0)),
            realized_pnl=float(data.get('realizedPnl', 0)),
            timestamp=int(data.get('timestamp', 0)),
            leverage=float(data.get('leverage', 1)),
            liquidation_price=float(data['liquidationPrice']) if data.get('liquidationPrice') else None,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "size": self.size,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "pnl_pct": self.pnl_pct,
            "timestamp": self.timestamp,
            "leverage": self.leverage,
            "liquidation_price": self.liquidation_price,
        }


@dataclass
class Trade:
    """Unified completed trade structure.

    Represents a closed trade with full lifecycle information.
    """

    id: str
    symbol: str
    side: OrderSide
    entry_price: float
    exit_price: float
    size: float
    pnl: float  # Profit/loss in quote currency
    pnl_pct: float  # Profit/loss percentage
    entry_timestamp: int
    exit_timestamp: int
    hold_time_seconds: int = 0
    fee_total: float = 0.0
    decision_data: dict = field(default_factory=dict)  # Agent decision context

    def __post_init__(self):
        """Calculate hold time if not provided."""
        if self.hold_time_seconds == 0:
            self.hold_time_seconds = (self.exit_timestamp - self.entry_timestamp) // 1000

    @property
    def is_win(self) -> bool:
        """Whether trade was profitable."""
        return self.pnl > 0

    @property
    def is_loss(self) -> bool:
        """Whether trade was a loss."""
        return self.pnl < 0

    @property
    def hold_time_hours(self) -> float:
        """Hold time in hours."""
        return self.hold_time_seconds / 3600

    @property
    def entry_datetime(self) -> datetime:
        """Entry timestamp as datetime."""
        return datetime.fromtimestamp(self.entry_timestamp / 1000)

    @property
    def exit_datetime(self) -> datetime:
        """Exit timestamp as datetime."""
        return datetime.fromtimestamp(self.exit_timestamp / 1000)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "entry_timestamp": self.entry_timestamp,
            "exit_timestamp": self.exit_timestamp,
            "hold_time_seconds": self.hold_time_seconds,
            "fee_total": self.fee_total,
            "decision_data": self.decision_data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trade":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            entry_price=data["entry_price"],
            exit_price=data["exit_price"],
            size=data["size"],
            pnl=data["pnl"],
            pnl_pct=data["pnl_pct"],
            entry_timestamp=data["entry_timestamp"],
            exit_timestamp=data["exit_timestamp"],
            hold_time_seconds=data.get("hold_time_seconds", 0),
            fee_total=data.get("fee_total", 0.0),
            decision_data=data.get("decision_data", {}),
        )
