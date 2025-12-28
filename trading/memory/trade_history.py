"""Trade history and journaling system.

File-based trade journal for persistent storage and advanced performance analysis.
Includes Sharpe ratio, Sortino ratio, maximum drawdown, and rolling metrics.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict, field
import math

from ..models.positions import Order, Position, OrderSide, OrderStatus
from ..models.decision import TradingDecision
from ..models.regime import MarketRegime


@dataclass
class TradeRecord:
    """Record of a completed trade with full context."""

    # Trade identification
    trade_id: str
    symbol: str
    timestamp: int

    # Order details
    side: str  # buy/sell
    order_type: str  # market/limit
    amount: float
    entry_price: float
    exit_price: Optional[float] = None

    # Performance
    realized_pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0

    # Context
    market_regime: Optional[str] = None
    bull_confidence: Optional[float] = None
    bear_confidence: Optional[float] = None
    decision_confidence: Optional[float] = None

    # Outcome
    won: bool = False
    closed: bool = False
    close_timestamp: Optional[int] = None

    # Agent insights (for learning)
    agent_votes: Dict[str, Any] = field(default_factory=dict)
    signals: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TradeRecord":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_order(
        cls,
        order: Order,
        decision: Optional[TradingDecision] = None,
    ) -> "TradeRecord":
        """Create trade record from order and decision context."""
        record = cls(
            trade_id=order.id,
            symbol=order.symbol,
            timestamp=order.timestamp,
            side=order.side.value,
            order_type=order.order_type.value,
            amount=order.amount,
            entry_price=order.price or 0.0,
        )

        # Add decision context if available
        if decision:
            record.market_regime = decision.regime.value
            record.bull_confidence = decision.bull_vote.confidence
            record.bear_confidence = decision.bear_vote.confidence
            record.decision_confidence = decision.confidence
            record.agent_votes = {
                "bull": {
                    "confidence": decision.bull_vote.confidence,
                    "reasoning": decision.bull_vote.reasoning,
                },
                "bear": {
                    "confidence": decision.bear_vote.confidence,
                    "reasoning": decision.bear_vote.reasoning,
                },
            }

        return record

    def close_trade(self, exit_price: float, fees: float = 0.0):
        """Mark trade as closed and calculate PnL."""
        self.exit_price = exit_price
        self.fees = fees
        self.closed = True
        self.close_timestamp = int(datetime.now().timestamp() * 1000)

        # Calculate PnL
        if self.side == "buy":  # Long position
            self.realized_pnl = (exit_price - self.entry_price) * self.amount - fees
        else:  # Short position
            self.realized_pnl = (self.entry_price - exit_price) * self.amount - fees

        self.pnl_pct = (self.realized_pnl / (self.entry_price * self.amount)) * 100
        self.won = self.realized_pnl > 0


class TradeJournal:
    """Hybrid trade journal with database and file-based storage.

    Stores trades in PostgreSQL database with TimescaleDB optimization.
    Falls back to file-based storage if database is unavailable.
    """

    def __init__(self, spec_dir: Path, use_database: bool = True):
        """Initialize trade journal.

        Args:
            spec_dir: Spec directory (e.g., specs/001-feature/)
            use_database: Try to use database if available (default: True)
        """
        self.spec_dir = spec_dir
        self.memory_dir = spec_dir / "memory" / "trades"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Database connection (with fallback)
        self.use_database = use_database
        self.db = None
        self.repo = None

        if use_database:
            try:
                from ..database.connection import get_db
                from ..database.repositories import TradeRepository

                self.db = next(get_db())
                self.repo = TradeRepository(self.db)
                self.logger.info("TradeJournal using database storage")
            except Exception as e:
                self.logger.warning(f"Database unavailable, falling back to file storage: {e}")
                self.use_database = False

        if not self.use_database:
            self.logger.info("TradeJournal using file-based storage")

        # Index file for file-based storage
        self.index_file = self.memory_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, str]:
        """Load trade index (maps trade_id to filename)."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load trade index: {e}")
        return {}

    def _save_index(self):
        """Save trade index to disk."""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save trade index: {e}")

    def log_trade(self, trade: TradeRecord):
        """Log a trade to the journal (database or file).

        Args:
            trade: Trade record to log
        """
        if self.use_database and self.repo:
            try:
                # Convert timestamp from milliseconds to datetime
                trade_data = trade.to_dict()
                trade_data['timestamp'] = datetime.fromtimestamp(trade.timestamp / 1000)
                if trade.close_timestamp:
                    trade_data['close_timestamp'] = datetime.fromtimestamp(trade.close_timestamp / 1000)

                self.repo.create_trade(trade_data)
                self.logger.info(f"Trade logged to database: {trade.trade_id} ({trade.symbol})")
                return
            except Exception as e:
                self.logger.error(f"Failed to log trade to database: {e}, falling back to file")
                self.use_database = False  # Disable database for future calls

        # File-based fallback
        timestamp_str = datetime.fromtimestamp(trade.timestamp / 1000).strftime(
            "%Y%m%d_%H%M%S"
        )
        filename = f"{timestamp_str}_{trade.trade_id}.json"
        filepath = self.memory_dir / filename

        try:
            with open(filepath, "w") as f:
                json.dump(trade.to_dict(), f, indent=2)

            # Update index
            self.index[trade.trade_id] = filename
            self._save_index()

            self.logger.info(f"Trade logged to file: {trade.trade_id} ({trade.symbol})")

        except Exception as e:
            self.logger.error(f"Failed to log trade to file: {e}")

    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """Get a specific trade by ID.

        Args:
            trade_id: Trade ID to lookup

        Returns:
            Trade record or None if not found
        """
        filename = self.index.get(trade_id)
        if not filename:
            return None

        filepath = self.memory_dir / filename
        if not filepath.exists():
            return None

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return TradeRecord.from_dict(data)
        except Exception as e:
            self.logger.error(f"Failed to load trade {trade_id}: {e}")
            return None

    def get_recent_trades(self, limit: int = 50) -> List[TradeRecord]:
        """Get recent trades sorted by timestamp.

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of trade records
        """
        trades = []

        # Get all trade files sorted by timestamp (filename)
        trade_files = sorted(self.memory_dir.glob("*.json"), reverse=True)

        # Exclude index file
        trade_files = [f for f in trade_files if f.name != "index.json"]

        for filepath in trade_files[:limit]:
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                trades.append(TradeRecord.from_dict(data))
            except Exception as e:
                self.logger.warning(f"Failed to load trade from {filepath}: {e}")

        return trades

    def get_all_trades(self) -> List[TradeRecord]:
        """Get all trades from the journal.

        Returns:
            List of all trade records
        """
        return self.get_recent_trades(limit=999999)

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from trade history.

        Returns:
            Dictionary with metrics (win_rate, total_pnl, avg_win, etc.)
        """
        trades = self.get_all_trades()

        if not trades:
            return {
                "total_trades": 0,
                "closed_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "profit_factor": 0.0,
            }

        closed_trades = [t for t in trades if t.closed]
        winning_trades = [t for t in closed_trades if t.won]
        losing_trades = [t for t in closed_trades if not t.won]

        total_pnl = sum(t.realized_pnl for t in closed_trades)
        total_wins = sum(t.realized_pnl for t in winning_trades)
        total_losses = abs(sum(t.realized_pnl for t in losing_trades))

        return {
            "total_trades": len(trades),
            "closed_trades": len(closed_trades),
            "win_rate": (len(winning_trades) / len(closed_trades) * 100)
            if closed_trades
            else 0.0,
            "total_pnl": total_pnl,
            "avg_win": (total_wins / len(winning_trades)) if winning_trades else 0.0,
            "avg_loss": (total_losses / len(losing_trades)) if losing_trades else 0.0,
            "largest_win": max((t.realized_pnl for t in winning_trades), default=0.0),
            "largest_loss": min((t.realized_pnl for t in losing_trades), default=0.0),
            "profit_factor": (total_wins / total_losses) if total_losses > 0 else 0.0,
        }

    def get_trades_by_regime(self, regime: MarketRegime) -> List[TradeRecord]:
        """Get all trades executed in a specific market regime.

        Args:
            regime: Market regime to filter by

        Returns:
            List of trades in that regime
        """
        all_trades = self.get_all_trades()
        return [t for t in all_trades if t.market_regime == regime.value]

    def get_trades_by_symbol(self, symbol: str) -> List[TradeRecord]:
        """Get all trades for a specific symbol.

        Args:
            symbol: Trading symbol to filter by

        Returns:
            List of trades for that symbol
        """
        all_trades = self.get_all_trades()
        return [t for t in all_trades if t.symbol == symbol]

    def calculate_sharpe_ratio(
        self,
        trades: Optional[List[TradeRecord]] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate Sharpe ratio from trade returns.

        Sharpe ratio = (Average Return - Risk-Free Rate) / Standard Deviation of Returns

        Args:
            trades: List of trades (None = all trades)
            risk_free_rate: Annual risk-free rate (default 2%)
            periods_per_year: Trading periods per year (default 252 for daily)

        Returns:
            Sharpe ratio (annualized)

        Example:
            >>> journal = TradeJournal(spec_dir)
            >>> sharpe = journal.calculate_sharpe_ratio()
            >>> print(f"Sharpe Ratio: {sharpe:.2f}")
        """
        if trades is None:
            trades = self.get_all_trades()

        closed_trades = [t for t in trades if t.closed]

        if not closed_trades:
            return 0.0

        # Calculate returns (P&L percentage for each trade)
        returns = [t.pnl_pct / 100 for t in closed_trades]

        # Calculate mean return
        mean_return = sum(returns) / len(returns)

        # Calculate standard deviation
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return 0.0

        # Annualize (assuming trades are independent events)
        # For trading, we typically annualize by sqrt(periods)
        annualized_return = mean_return * periods_per_year
        annualized_std = std_dev * math.sqrt(periods_per_year)

        # Sharpe ratio
        sharpe = (annualized_return - risk_free_rate) / annualized_std

        return sharpe

    def calculate_sortino_ratio(
        self,
        trades: Optional[List[TradeRecord]] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate Sortino ratio from trade returns.

        Sortino ratio = (Average Return - Risk-Free Rate) / Downside Deviation
        Similar to Sharpe but only penalizes downside volatility.

        Args:
            trades: List of trades (None = all trades)
            risk_free_rate: Annual risk-free rate (default 2%)
            periods_per_year: Trading periods per year

        Returns:
            Sortino ratio (annualized)
        """
        if trades is None:
            trades = self.get_all_trades()

        closed_trades = [t for t in trades if t.closed]

        if not closed_trades:
            return 0.0

        # Calculate returns
        returns = [t.pnl_pct / 100 for t in closed_trades]
        mean_return = sum(returns) / len(returns)

        # Calculate downside deviation (only negative returns)
        downside_returns = [r for r in returns if r < 0]

        if not downside_returns:
            return 0.0  # No downside = undefined Sortino

        downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
        downside_std = math.sqrt(downside_variance)

        if downside_std == 0:
            return 0.0

        # Annualize
        annualized_return = mean_return * periods_per_year
        annualized_downside_std = downside_std * math.sqrt(periods_per_year)

        # Sortino ratio
        sortino = (annualized_return - risk_free_rate) / annualized_downside_std

        return sortino

    def calculate_max_drawdown(
        self,
        trades: Optional[List[TradeRecord]] = None,
        initial_balance: float = 10000.0,
    ) -> tuple[float, float]:
        """Calculate maximum drawdown from trade equity curve.

        Max drawdown = largest peak-to-trough decline in equity.

        Args:
            trades: List of trades (None = all trades)
            initial_balance: Starting balance in USD

        Returns:
            Tuple of (max_drawdown_usd, max_drawdown_pct)

        Example:
            >>> dd_usd, dd_pct = journal.calculate_max_drawdown()
            >>> print(f"Max Drawdown: ${dd_usd:.2f} ({dd_pct:.1f}%)")
        """
        if trades is None:
            trades = self.get_all_trades()

        closed_trades = sorted(
            [t for t in trades if t.closed],
            key=lambda t: t.close_timestamp or 0,
        )

        if not closed_trades:
            return 0.0, 0.0

        # Build equity curve
        equity = initial_balance
        equity_curve = [initial_balance]

        for trade in closed_trades:
            equity += trade.realized_pnl
            equity_curve.append(equity)

        # Calculate drawdowns
        max_drawdown_usd = 0.0
        max_drawdown_pct = 0.0
        peak = equity_curve[0]

        for value in equity_curve:
            # Update peak
            if value > peak:
                peak = value

            # Calculate drawdown from peak
            drawdown = peak - value
            drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0

            # Update max
            if drawdown > max_drawdown_usd:
                max_drawdown_usd = drawdown
                max_drawdown_pct = drawdown_pct

        return max_drawdown_usd, max_drawdown_pct

    def calculate_rolling_metrics(
        self,
        window_days: int = 30,
        trades: Optional[List[TradeRecord]] = None,
    ) -> List[Dict[str, Any]]:
        """Calculate rolling performance metrics over a time window.

        Args:
            window_days: Size of rolling window in days
            trades: List of trades (None = all trades)

        Returns:
            List of metric snapshots at each window

        Example:
            >>> metrics = journal.calculate_rolling_metrics(window_days=7)
            >>> for m in metrics[-5:]:
            ...     print(f"{m['date']}: Win Rate {m['win_rate']:.1f}%")
        """
        if trades is None:
            trades = self.get_all_trades()

        closed_trades = sorted(
            [t for t in trades if t.closed and t.close_timestamp],
            key=lambda t: t.close_timestamp or 0,
        )

        if not closed_trades:
            return []

        # Get date range
        start_timestamp = closed_trades[0].close_timestamp or 0
        end_timestamp = closed_trades[-1].close_timestamp or 0

        window_ms = window_days * 24 * 3600 * 1000  # Convert days to milliseconds
        rolling_metrics = []

        # Iterate through windows
        current_timestamp = start_timestamp + window_ms

        while current_timestamp <= end_timestamp:
            # Get trades in window
            window_start = current_timestamp - window_ms
            window_trades = [
                t for t in closed_trades
                if window_start <= (t.close_timestamp or 0) <= current_timestamp
            ]

            if window_trades:
                # Calculate metrics for window
                winning = [t for t in window_trades if t.won]
                total_pnl = sum(t.realized_pnl for t in window_trades)

                rolling_metrics.append({
                    "date": datetime.fromtimestamp(current_timestamp / 1000).strftime("%Y-%m-%d"),
                    "timestamp": current_timestamp,
                    "total_trades": len(window_trades),
                    "winning_trades": len(winning),
                    "win_rate": (len(winning) / len(window_trades) * 100),
                    "total_pnl": total_pnl,
                    "avg_pnl": total_pnl / len(window_trades),
                })

            # Move window forward by 1 day
            current_timestamp += (24 * 3600 * 1000)

        return rolling_metrics

    def log_backtest_result(self, backtest_id: str, result: Dict[str, Any]):
        """Store backtest result for later analysis.

        Args:
            backtest_id: Unique backtest identifier
            result: Backtest result dictionary

        Example:
            >>> journal.log_backtest_result("backtest_20240101", {
            ...     "sharpe_ratio": 1.5,
            ...     "max_drawdown": -15.2,
            ...     "total_pnl": 1250.0,
            ... })
        """
        backtest_dir = self.memory_dir.parent / "backtests"
        backtest_dir.mkdir(parents=True, exist_ok=True)

        filepath = backtest_dir / f"{backtest_id}.json"

        try:
            with open(filepath, "w") as f:
                json.dump(result, f, indent=2)

            self.logger.info(f"Backtest result logged: {backtest_id}")

        except Exception as e:
            self.logger.error(f"Failed to log backtest result: {e}")

    def get_backtest_results(self) -> List[Dict[str, Any]]:
        """Get all stored backtest results.

        Returns:
            List of backtest result dictionaries
        """
        backtest_dir = self.memory_dir.parent / "backtests"

        if not backtest_dir.exists():
            return []

        results = []

        for filepath in backtest_dir.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    data["backtest_id"] = filepath.stem
                    results.append(data)
            except Exception as e:
                self.logger.warning(f"Failed to load backtest from {filepath}: {e}")

        return results

    def calculate_comprehensive_metrics(
        self,
        trades: Optional[List[TradeRecord]] = None,
        initial_balance: float = 10000.0,
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics including risk-adjusted returns.

        Args:
            trades: List of trades (None = all trades)
            initial_balance: Starting balance

        Returns:
            Dictionary with all performance metrics

        Example:
            >>> metrics = journal.calculate_comprehensive_metrics()
            >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
            >>> print(f"Max DD: {metrics['max_drawdown_pct']:.1f}%")
        """
        basic_metrics = self.calculate_metrics()

        # Add advanced metrics
        sharpe = self.calculate_sharpe_ratio(trades)
        sortino = self.calculate_sortino_ratio(trades)
        max_dd_usd, max_dd_pct = self.calculate_max_drawdown(trades, initial_balance)

        # Combine all metrics
        comprehensive = {
            **basic_metrics,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd_usd,
            "max_drawdown_pct": max_dd_pct,
            "initial_balance": initial_balance,
            "final_balance": initial_balance + basic_metrics["total_pnl"],
            "return_pct": (basic_metrics["total_pnl"] / initial_balance * 100)
            if initial_balance > 0
            else 0.0,
        }

        return comprehensive

    def get_performance_summary(self) -> str:
        """Generate human-readable performance summary.

        Returns:
            Formatted performance summary string

        Example:
            >>> print(journal.get_performance_summary())
            === Performance Summary ===
            Total Trades: 45
            Win Rate: 62.2%
            ...
        """
        metrics = self.calculate_comprehensive_metrics()

        summary = "=== Performance Summary ===\n\n"
        summary += f"Total Trades: {metrics['total_trades']}\n"
        summary += f"Closed Trades: {metrics['closed_trades']}\n"
        summary += f"Win Rate: {metrics['win_rate']:.1f}%\n\n"

        summary += f"Total P&L: ${metrics['total_pnl']:,.2f}\n"
        summary += f"Return: {metrics['return_pct']:+.2f}%\n\n"

        summary += f"Average Win: ${metrics['avg_win']:.2f}\n"
        summary += f"Average Loss: ${metrics['avg_loss']:.2f}\n"
        summary += f"Profit Factor: {metrics['profit_factor']:.2f}\n\n"

        summary += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        summary += f"Sortino Ratio: {metrics['sortino_ratio']:.2f}\n"
        summary += f"Max Drawdown: ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.1f}%)\n"

        return summary
