"""Portfolio Manager - Multi-symbol position tracking and management.

Centralized management for portfolios containing multiple cryptocurrency positions.
Tracks positions, calculates metrics, and manages risk across the entire portfolio.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class Position:
    """Individual position within portfolio."""
    symbol: str
    amount: float  # Number of coins
    entry_price: float
    current_price: float
    entry_timestamp: datetime
    side: str = "long"  # or "short"

    @property
    def value(self) -> float:
        """Current position value in USD."""
        return self.amount * self.current_price

    @property
    def cost_basis(self) -> float:
        """Original cost basis in USD."""
        return self.amount * self.entry_price

    @property
    def pnl(self) -> float:
        """Unrealized profit/loss in USD."""
        if self.side == "long":
            return self.value - self.cost_basis
        else:  # short
            return self.cost_basis - self.value

    @property
    def pnl_pct(self) -> float:
        """Unrealized profit/loss percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.pnl / self.cost_basis) * 100


class PortfolioManager:
    """Manage multi-symbol cryptocurrency portfolio.

    Tracks positions, calculates portfolio-level metrics,
    and manages risk across multiple assets.

    Example:
        >>> portfolio = PortfolioManager(symbols=["BTC/USDT", "ETH/USDT"])
        >>> await portfolio.add_position("BTC/USDT", 0.5, 42000)
        >>> metrics = portfolio.get_metrics()
    """

    def __init__(
        self,
        symbols: List[str],
        max_total_capital: float = 100000,
        max_position_pct: float = 0.5,
        max_symbols: int = 10,
    ):
        """Initialize portfolio manager.

        Args:
            symbols: List of symbols to track
            max_total_capital: Maximum total portfolio capital
            max_position_pct: Maximum percentage per position (0-1)
            max_symbols: Maximum number of concurrent positions
        """
        self.logger = logging.getLogger(__name__)

        self.symbols = symbols
        self.max_total_capital = max_total_capital
        self.max_position_pct = max_position_pct
        self.max_symbols = max_symbols

        # Positions: symbol -> Position
        self.positions: Dict[str, Position] = {}

        # Cash balance
        self.cash_balance = max_total_capital

        # Historical equity curve
        self.equity_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.total_realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0

    async def add_position(
        self,
        symbol: str,
        amount: float,
        entry_price: float,
        side: str = "long",
    ) -> bool:
        """Add or increase position.

        Args:
            symbol: Trading symbol
            amount: Amount to add
            entry_price: Entry price
            side: Position side (long/short)

        Returns:
            True if position added successfully
        """
        # Check if symbol is allowed
        if symbol not in self.symbols:
            self.logger.warning(f"Symbol {symbol} not in allowed list")
            return False

        # Check max symbols limit
        if symbol not in self.positions and len(self.positions) >= self.max_symbols:
            self.logger.warning(f"Max symbols limit ({self.max_symbols}) reached")
            return False

        # Check capital requirement
        cost = amount * entry_price
        if cost > self.cash_balance:
            self.logger.warning(f"Insufficient cash: need ${cost:,.2f}, have ${self.cash_balance:,.2f}")
            return False

        # Check position size limit
        total_value = self.get_total_value()
        if total_value > 0:
            position_pct = cost / (total_value + self.cash_balance)
            if position_pct > self.max_position_pct:
                self.logger.warning(f"Position size {position_pct:.1%} exceeds limit {self.max_position_pct:.1%}")
                return False

        # Add or update position
        if symbol in self.positions:
            # Average up/down existing position
            existing = self.positions[symbol]
            total_amount = existing.amount + amount
            avg_price = ((existing.amount * existing.entry_price) + (amount * entry_price)) / total_amount

            existing.amount = total_amount
            existing.entry_price = avg_price

            self.logger.info(f"Increased {symbol} position to {total_amount:.4f} @ ${avg_price:,.2f}")

        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                amount=amount,
                entry_price=entry_price,
                current_price=entry_price,
                entry_timestamp=datetime.now(),
                side=side,
            )

            self.logger.info(f"Added {symbol} position: {amount:.4f} @ ${entry_price:,.2f}")

        # Update cash balance
        self.cash_balance -= cost

        return True

    async def remove_position(
        self,
        symbol: str,
        amount: Optional[float] = None,
        exit_price: Optional[float] = None,
    ) -> Optional[float]:
        """Remove or reduce position.

        Args:
            symbol: Trading symbol
            amount: Amount to remove (None = close entire position)
            exit_price: Exit price (None = use current price)

        Returns:
            Realized PnL or None if position doesn't exist
        """
        if symbol not in self.positions:
            self.logger.warning(f"No position for {symbol}")
            return None

        position = self.positions[symbol]

        # Default to full position closure
        if amount is None:
            amount = position.amount

        # Validate amount
        if amount > position.amount:
            self.logger.warning(f"Cannot remove {amount}, only have {position.amount}")
            return None

        # Default to current price
        if exit_price is None:
            exit_price = position.current_price

        # Calculate realized PnL
        realized_pnl = ((exit_price - position.entry_price) / position.entry_price) * (amount * position.entry_price)

        if position.side == "short":
            realized_pnl = -realized_pnl

        # Update position
        if amount >= position.amount:
            # Close entire position
            del self.positions[symbol]
            self.logger.info(f"Closed {symbol} position: PnL = ${realized_pnl:+,.2f}")
        else:
            # Partial closure
            position.amount -= amount
            self.logger.info(f"Reduced {symbol} position by {amount:.4f}: PnL = ${realized_pnl:+,.2f}")

        # Update cash balance
        proceeds = amount * exit_price
        self.cash_balance += proceeds

        # Update performance tracking
        self.total_realized_pnl += realized_pnl
        self.total_trades += 1
        if realized_pnl > 0:
            self.winning_trades += 1

        return realized_pnl

    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions.

        Args:
            prices: Dictionary mapping symbols to current prices
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]

        # Record equity snapshot
        self._record_equity_snapshot()

    def _record_equity_snapshot(self):
        """Record current equity for historical tracking."""
        self.equity_history.append({
            'timestamp': datetime.now().isoformat(),
            'total_value': self.get_total_value(),
            'cash': self.cash_balance,
            'positions_value': self.get_positions_value(),
            'unrealized_pnl': self.get_unrealized_pnl(),
        })

        # Keep last 10,000 snapshots
        if len(self.equity_history) > 10000:
            self.equity_history = self.equity_history[-10000:]

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position object or None
        """
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all active positions.

        Returns:
            Dictionary of positions
        """
        return self.positions.copy()

    def get_positions_value(self) -> float:
        """Get total value of all positions.

        Returns:
            Total positions value in USD
        """
        return sum(pos.value for pos in self.positions.values())

    def get_total_value(self) -> float:
        """Get total portfolio value (cash + positions).

        Returns:
            Total portfolio value in USD
        """
        return self.cash_balance + self.get_positions_value()

    def get_unrealized_pnl(self) -> float:
        """Get total unrealized PnL.

        Returns:
            Unrealized PnL in USD
        """
        return sum(pos.pnl for pos in self.positions.values())

    def get_allocation(self) -> Dict[str, float]:
        """Get current portfolio allocation percentages.

        Returns:
            Dictionary mapping symbols to allocation percentages (0-1)
        """
        total_value = self.get_total_value()

        if total_value == 0:
            return {}

        allocation = {}
        for symbol, position in self.positions.items():
            allocation[symbol] = position.value / total_value

        # Add cash allocation
        allocation['CASH'] = self.cash_balance / total_value

        return allocation

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio metrics.

        Returns:
            Dictionary of portfolio metrics
        """
        total_value = self.get_total_value()
        positions_value = self.get_positions_value()
        unrealized_pnl = self.get_unrealized_pnl()

        # Calculate returns
        initial_capital = self.max_total_capital
        total_return = ((total_value - initial_capital) / initial_capital) * 100

        # Win rate
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        # Concentration (Herfindahl index)
        allocation = self.get_allocation()
        allocation_values = [v for k, v in allocation.items() if k != 'CASH']
        concentration = sum(v ** 2 for v in allocation_values) if allocation_values else 0

        # Diversification ratio (inverse of concentration)
        diversification = 1 / concentration if concentration > 0 else 0

        return {
            'timestamp': datetime.now().isoformat(),
            'total_value': total_value,
            'cash_balance': self.cash_balance,
            'positions_value': positions_value,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': self.total_realized_pnl,
            'total_pnl': unrealized_pnl + self.total_realized_pnl,
            'total_return_pct': total_return,
            'num_positions': len(self.positions),
            'allocation': allocation,
            'concentration': concentration,
            'diversification_ratio': diversification,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
        }

    def get_equity_curve(self) -> pd.DataFrame:
        """Get historical equity curve.

        Returns:
            DataFrame with equity history
        """
        if not self.equity_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.equity_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        return df

    def calculate_var(self, confidence: float = 0.95, window_days: int = 30) -> float:
        """Calculate Value at Risk.

        Args:
            confidence: Confidence level (0-1)
            window_days: Historical window in days

        Returns:
            VaR in USD (negative = loss)
        """
        equity_curve = self.get_equity_curve()

        if len(equity_curve) < window_days:
            return 0.0

        # Calculate daily returns
        returns = equity_curve['total_value'].pct_change().dropna()
        recent_returns = returns.tail(window_days)

        # Calculate VaR
        var_percentile = (1 - confidence) * 100
        var = np.percentile(recent_returns, var_percentile)

        # Convert to USD
        current_value = self.get_total_value()
        var_usd = var * current_value

        return var_usd

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02, window_days: int = 365) -> float:
        """Calculate portfolio Sharpe ratio.

        Args:
            risk_free_rate: Annual risk-free rate
            window_days: Historical window in days

        Returns:
            Sharpe ratio
        """
        equity_curve = self.get_equity_curve()

        if len(equity_curve) < window_days:
            return 0.0

        # Calculate daily returns
        returns = equity_curve['total_value'].pct_change().dropna()
        recent_returns = returns.tail(window_days)

        # Annualized metrics
        mean_return = recent_returns.mean() * 252  # Trading days per year
        std_return = recent_returns.std() * np.sqrt(252)

        if std_return == 0:
            return 0.0

        # Sharpe ratio
        sharpe = (mean_return - risk_free_rate) / std_return

        return sharpe

    def __repr__(self) -> str:
        """String representation."""
        return f"PortfolioManager(positions={len(self.positions)}, value=${self.get_total_value():,.2f})"
