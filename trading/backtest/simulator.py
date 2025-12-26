"""Market Simulator - Virtual order execution for backtesting.

Simulates realistic order execution with slippage, fees, and market impact.
Tracks positions, balance, and P&L during backtest runs.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import uuid

from ..config import TradingConfig
from ..models.market_data import OHLCV
from ..models.positions import Order, OrderSide, OrderStatus, OrderType
from ..memory.trade_history import TradeRecord


@dataclass
class SimulatedPosition:
    """A simulated position held during backtest."""

    position_id: str
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    entry_timestamp: int
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


class MarketSimulator:
    """Simulates market execution for backtesting.

    Handles:
    - Virtual order execution with slippage
    - Position tracking
    - Balance management
    - Fee calculation
    - Realistic market impact modeling

    Example:
        >>> simulator = MarketSimulator(config)
        >>> simulator.reset(initial_balance=10000.0)
        >>> order = simulator.execute_order(
        ...     symbol="BTC/USDT",
        ...     side="buy",
        ...     amount=0.1,
        ...     current_candle=candle
        ... )
    """

    def __init__(self, config: TradingConfig):
        """Initialize market simulator.

        Args:
            config: Trading configuration for fee rates and slippage
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Simulation parameters
        self.slippage_pct = 0.05  # 0.05% slippage
        self.maker_fee_pct = 0.02  # 0.02% maker fee
        self.taker_fee_pct = 0.04  # 0.04% taker fee (default for market orders)

        # State
        self.balance_usd = 0.0
        self.initial_balance = 0.0
        self.positions: Dict[str, SimulatedPosition] = {}
        self.equity_curve: List[float] = []
        self.current_candle: Optional[OHLCV] = None

    def reset(self, initial_balance: float):
        """Reset simulator state for new backtest.

        Args:
            initial_balance: Starting capital in USD
        """
        self.balance_usd = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.equity_curve = [initial_balance]
        self.current_candle = None

        self.logger.info(f"Simulator reset with ${initial_balance:,.2f}")

    def update_candle(self, candle: OHLCV):
        """Update current candle and recalculate positions.

        Args:
            candle: Current market candle
        """
        self.current_candle = candle

        # Update positions with current price
        for position in self.positions.values():
            position.current_price = candle.close

            # Calculate unrealized P&L
            if position.side == "long":
                position.unrealized_pnl = (candle.close - position.entry_price) * position.size
            else:  # short
                position.unrealized_pnl = (position.entry_price - candle.close) * position.size

        # Update equity curve
        total_equity = self.balance_usd + sum(p.unrealized_pnl for p in self.positions.values())
        self.equity_curve.append(total_equity)

    def execute_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        current_candle: OHLCV,
        order_type: str = "market",
    ) -> Order:
        """Execute a virtual order with slippage and fees.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            amount: Order size in base currency
            current_candle: Current market candle
            order_type: "market" or "limit"

        Returns:
            Simulated Order object

        Example:
            >>> order = simulator.execute_order(
            ...     "BTC/USDT", "buy", 0.1, candle
            ... )
            >>> order.filled
            0.1
        """
        # Calculate execution price with slippage
        if order_type == "market":
            if side == "buy":
                # Market buy: execute at ask + slippage
                execution_price = current_candle.close * (1 + self.slippage_pct / 100)
                fee_pct = self.taker_fee_pct
            else:  # sell
                # Market sell: execute at bid - slippage
                execution_price = current_candle.close * (1 - self.slippage_pct / 100)
                fee_pct = self.taker_fee_pct
        else:  # limit order
            execution_price = current_candle.close  # Assume filled at limit price
            fee_pct = self.maker_fee_pct

        # Calculate total cost including fees
        notional_value = amount * execution_price
        fee = notional_value * (fee_pct / 100)
        total_cost = notional_value + fee

        # Check if sufficient balance
        if side == "buy" and total_cost > self.balance_usd:
            self.logger.warning(
                f"Insufficient balance: ${self.balance_usd:.2f} < ${total_cost:.2f}"
            )
            # Execute with available balance
            available_amount = (self.balance_usd * 0.99) / execution_price
            amount = available_amount
            notional_value = amount * execution_price
            fee = notional_value * (fee_pct / 100)
            total_cost = notional_value + fee

        # Update balance
        if side == "buy":
            self.balance_usd -= total_cost
            position_side = "long"
        else:  # sell
            self.balance_usd += (notional_value - fee)
            position_side = "short"

        # Create or update position
        position_id = f"{symbol}_{position_side}"

        if position_id in self.positions:
            # Add to existing position (average price)
            existing = self.positions[position_id]
            total_size = existing.size + amount
            avg_price = (
                (existing.entry_price * existing.size) + (execution_price * amount)
            ) / total_size

            existing.size = total_size
            existing.entry_price = avg_price
        else:
            # Create new position
            self.positions[position_id] = SimulatedPosition(
                position_id=position_id,
                symbol=symbol,
                side=position_side,
                size=amount,
                entry_price=execution_price,
                entry_timestamp=current_candle.timestamp,
                current_price=execution_price,
            )

        # Create order object
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            order_type=OrderType.MARKET if order_type == "market" else OrderType.LIMIT,
            amount=amount,
            price=execution_price,
            status=OrderStatus.FILLED,
            filled=amount,
            remaining=0.0,
            timestamp=current_candle.timestamp,
            fee=fee,
            fee_currency="USDT",
        )

        self.logger.debug(
            f"Executed {side} {amount:.4f} {symbol} @ ${execution_price:.2f} "
            f"(fee: ${fee:.2f}, balance: ${self.balance_usd:.2f})"
        )

        return order

    def close_position(
        self,
        position_id: str,
        current_candle: OHLCV,
    ) -> Optional[TradeRecord]:
        """Close a position and realize P&L.

        Args:
            position_id: Position ID to close
            current_candle: Current market candle

        Returns:
            TradeRecord if position was closed, None otherwise
        """
        if position_id not in self.positions:
            return None

        position = self.positions[position_id]

        # Execute closing order (opposite side)
        close_side = "sell" if position.side == "long" else "buy"
        close_order = self.execute_order(
            symbol=position.symbol,
            side=close_side,
            amount=position.size,
            current_candle=current_candle,
        )

        # Calculate realized P&L
        if position.side == "long":
            realized_pnl = (close_order.price - position.entry_price) * position.size
        else:  # short
            realized_pnl = (position.entry_price - close_order.price) * position.size

        # Deduct fees
        realized_pnl -= close_order.fee

        # Create trade record
        trade = TradeRecord(
            trade_id=close_order.id,
            symbol=position.symbol,
            timestamp=position.entry_timestamp,
            side=close_side,
            order_type="market",
            amount=position.size,
            entry_price=position.entry_price,
            exit_price=close_order.price,
            realized_pnl=realized_pnl,
            pnl_pct=(realized_pnl / (position.entry_price * position.size)) * 100,
            fees=close_order.fee,
            closed=True,
            close_timestamp=current_candle.timestamp,
            won=realized_pnl > 0,
        )

        # Remove position
        del self.positions[position_id]

        self.logger.info(
            f"Closed {position.side} position: "
            f"P&L ${realized_pnl:.2f} ({trade.pnl_pct:.2f}%)"
        )

        return trade

    def close_all_positions(self, current_candle: OHLCV) -> List[TradeRecord]:
        """Close all open positions (called at end of backtest).

        Args:
            current_candle: Final market candle

        Returns:
            List of trade records for closed positions
        """
        trades = []

        position_ids = list(self.positions.keys())
        for position_id in position_ids:
            trade = self.close_position(position_id, current_candle)
            if trade:
                trades.append(trade)

        return trades

    def get_total_equity(self) -> float:
        """Get total equity (balance + unrealized P&L)."""
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        return self.balance_usd + unrealized_pnl

    def get_position_value(self) -> float:
        """Get total value of open positions."""
        return sum(p.size * p.current_price for p in self.positions.values())

    def get_leverage(self) -> float:
        """Calculate current leverage."""
        position_value = self.get_position_value()
        total_equity = self.get_total_equity()

        if total_equity == 0:
            return 0.0

        return position_value / total_equity

    def check_margin_call(self) -> bool:
        """Check if account has margin call (negative equity)."""
        total_equity = self.get_total_equity()

        if total_equity <= 0:
            self.logger.error("MARGIN CALL: Account equity depleted!")
            return True

        # Check if leverage too high
        leverage = self.get_leverage()
        if leverage > 10.0:  # 10x leverage limit
            self.logger.warning(f"High leverage: {leverage:.1f}x")

        return False

    def apply_stop_loss(
        self,
        position_id: str,
        stop_price: float,
        current_candle: OHLCV,
    ) -> Optional[TradeRecord]:
        """Check and apply stop-loss if triggered.

        Args:
            position_id: Position to check
            stop_price: Stop-loss trigger price
            current_candle: Current market candle

        Returns:
            TradeRecord if stop was hit, None otherwise
        """
        if position_id not in self.positions:
            return None

        position = self.positions[position_id]

        # Check if stop-loss triggered
        triggered = False
        if position.side == "long" and current_candle.low <= stop_price:
            triggered = True
        elif position.side == "short" and current_candle.high >= stop_price:
            triggered = True

        if triggered:
            self.logger.info(f"Stop-loss triggered for {position_id} at ${stop_price:.2f}")
            return self.close_position(position_id, current_candle)

        return None

    def apply_take_profit(
        self,
        position_id: str,
        take_profit_price: float,
        current_candle: OHLCV,
    ) -> Optional[TradeRecord]:
        """Check and apply take-profit if triggered.

        Args:
            position_id: Position to check
            take_profit_price: Take-profit trigger price
            current_candle: Current market candle

        Returns:
            TradeRecord if take-profit was hit, None otherwise
        """
        if position_id not in self.positions:
            return None

        position = self.positions[position_id]

        # Check if take-profit triggered
        triggered = False
        if position.side == "long" and current_candle.high >= take_profit_price:
            triggered = True
        elif position.side == "short" and current_candle.low <= take_profit_price:
            triggered = True

        if triggered:
            self.logger.info(
                f"Take-profit triggered for {position_id} at ${take_profit_price:.2f}"
            )
            return self.close_position(position_id, current_candle)

        return None

    def get_performance_summary(self) -> Dict[str, float]:
        """Get current performance summary.

        Returns:
            Dictionary with balance, equity, P&L, and return metrics
        """
        total_equity = self.get_total_equity()
        total_pnl = total_equity - self.initial_balance
        return_pct = (total_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0.0

        return {
            "balance": self.balance_usd,
            "total_equity": total_equity,
            "total_pnl": total_pnl,
            "return_pct": return_pct,
            "open_positions": len(self.positions),
            "position_value": self.get_position_value(),
            "leverage": self.get_leverage(),
        }
