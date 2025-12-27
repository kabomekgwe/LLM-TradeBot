"""Real-time performance metrics tracker.

Maintains rolling window of trades and calculates live performance metrics:
- Sharpe ratio (annualized returns / std dev)
- Drawdown (peak equity - current equity) / peak equity
- Win rate (winning trades / total trades)
- Total P&L and daily P&L
- Consecutive losses

Updates after each trade execution and broadcasts to dashboard.
"""

import math
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics snapshot."""

    # Core metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0

    # P&L metrics
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    consecutive_losses: int = 0

    # Risk metrics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Equity tracking
    current_equity: float = 10000.0
    peak_equity: float = 10000.0

    # Timestamp
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetrics":
        """Create from dictionary."""
        return cls(**data)


class MetricsTracker:
    """Real-time performance metrics tracker.

    Maintains rolling window of trades and calculates live metrics
    after each trade execution. Used for dashboard updates and
    alert trigger checks.

    Features:
    - Rolling window calculations (configurable window size)
    - Real-time Sharpe ratio, Sortino ratio, drawdown
    - Win rate and consecutive loss tracking
    - Daily/weekly P&L aggregation
    - Equity curve tracking

    Example:
        >>> tracker = MetricsTracker(initial_equity=10000.0)
        >>> tracker.update_trade(trade_record)
        >>> metrics = tracker.get_current_metrics()
        >>> print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
    """

    def __init__(
        self,
        initial_equity: float = 10000.0,
        rolling_window_trades: int = 100,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ):
        """Initialize metrics tracker.

        Args:
            initial_equity: Starting account equity
            rolling_window_trades: Number of recent trades to consider
            risk_free_rate: Annual risk-free rate (default 2%)
            periods_per_year: Trading periods per year (default 252)
        """
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.initial_equity = initial_equity
        self.rolling_window_trades = rolling_window_trades
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        # Trade history (rolling window)
        self.trades: deque = deque(maxlen=rolling_window_trades)

        # Equity tracking
        self.current_equity = initial_equity
        self.peak_equity = initial_equity

        # Daily/weekly tracking
        self.daily_trades: List[Dict[str, Any]] = []
        self.weekly_trades: List[Dict[str, Any]] = []
        self.last_reset_date = datetime.now().date()
        self.last_weekly_reset = datetime.now()

        # Current metrics cache
        self._metrics_cache: Optional[PerformanceMetrics] = None
        self._cache_valid = False

        self.logger.info(
            "MetricsTracker initialized",
            extra={
                "initial_equity": initial_equity,
                "rolling_window": rolling_window_trades,
            }
        )

    def update_trade(self, trade_data: Dict[str, Any]) -> PerformanceMetrics:
        """Update metrics with new trade.

        Args:
            trade_data: Trade information dict with keys:
                - realized_pnl: float (required)
                - pnl_pct: float (optional)
                - won: bool (optional)
                - closed: bool (optional, default True)
                - timestamp: int (optional, milliseconds)
                - symbol: str (optional)
                - side: str (optional)

        Returns:
            Updated performance metrics
        """
        # Validate required fields
        if "realized_pnl" not in trade_data:
            self.logger.error("Trade data missing required field: realized_pnl")
            return self.get_current_metrics()

        # Only process closed trades
        if not trade_data.get("closed", True):
            self.logger.debug("Skipping open trade in metrics update")
            return self.get_current_metrics()

        # Add timestamp if missing
        if "timestamp" not in trade_data:
            trade_data["timestamp"] = int(datetime.now().timestamp() * 1000)

        # Add to rolling window
        self.trades.append(trade_data)

        # Update equity
        pnl = trade_data["realized_pnl"]
        self.current_equity += pnl

        # Update peak equity
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        # Add to daily/weekly tracking
        self._update_period_tracking(trade_data)

        # Invalidate cache
        self._cache_valid = False

        # Calculate and cache new metrics
        metrics = self._calculate_metrics()
        self._metrics_cache = metrics
        self._cache_valid = True

        self.logger.debug(
            "Metrics updated",
            extra={
                "pnl": pnl,
                "current_equity": self.current_equity,
                "sharpe": metrics.sharpe_ratio,
                "drawdown": metrics.current_drawdown,
            }
        )

        return metrics

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics.

        Returns:
            Current metrics snapshot
        """
        if self._cache_valid and self._metrics_cache:
            return self._metrics_cache

        # Calculate fresh metrics
        metrics = self._calculate_metrics()
        self._metrics_cache = metrics
        self._cache_valid = True

        return metrics

    def _calculate_metrics(self) -> PerformanceMetrics:
        """Calculate all performance metrics from current trade history.

        Returns:
            PerformanceMetrics object with all calculated values
        """
        if not self.trades:
            # Return default metrics if no trades
            return PerformanceMetrics(
                current_equity=self.current_equity,
                peak_equity=self.peak_equity,
                timestamp=datetime.now().isoformat(),
            )

        # Convert to list for easier processing
        trades_list = list(self.trades)

        # Basic statistics
        total_trades = len(trades_list)
        winning_trades = [t for t in trades_list if t.get("won", t.get("realized_pnl", 0) > 0)]
        losing_trades = [t for t in trades_list if not t.get("won", t.get("realized_pnl", 0) <= 0)]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0

        # P&L calculations
        total_pnl = sum(t.get("realized_pnl", 0) for t in trades_list)
        daily_pnl = sum(t.get("realized_pnl", 0) for t in self.daily_trades)
        weekly_pnl = sum(t.get("realized_pnl", 0) for t in self.weekly_trades)

        total_wins = sum(t.get("realized_pnl", 0) for t in winning_trades)
        total_losses = abs(sum(t.get("realized_pnl", 0) for t in losing_trades))

        avg_win = (total_wins / win_count) if win_count > 0 else 0.0
        avg_loss = (total_losses / loss_count) if loss_count > 0 else 0.0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0.0

        largest_win = max((t.get("realized_pnl", 0) for t in winning_trades), default=0.0)
        largest_loss = min((t.get("realized_pnl", 0) for t in losing_trades), default=0.0)

        # Consecutive losses
        consecutive_losses = self._calculate_consecutive_losses(trades_list)

        # Drawdown
        current_drawdown = self._calculate_current_drawdown()
        max_drawdown = self._calculate_max_drawdown()

        # Sharpe and Sortino ratios
        sharpe_ratio = self._calculate_sharpe_ratio(trades_list)
        sortino_ratio = self._calculate_sortino_ratio(trades_list)

        return PerformanceMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            consecutive_losses=consecutive_losses,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            largest_win=largest_win,
            largest_loss=largest_loss,
            current_equity=self.current_equity,
            peak_equity=self.peak_equity,
            timestamp=datetime.now().isoformat(),
        )

    def _calculate_sharpe_ratio(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate Sharpe ratio from trade returns.

        Sharpe ratio = (Average Return - Risk-Free Rate) / Standard Deviation

        Args:
            trades: List of trade dicts

        Returns:
            Sharpe ratio (annualized)
        """
        if len(trades) < 2:
            return 0.0

        # Calculate returns (percentage)
        returns = []
        for trade in trades:
            pnl_pct = trade.get("pnl_pct")
            if pnl_pct is not None:
                returns.append(pnl_pct / 100)
            elif "realized_pnl" in trade and "entry_price" in trade and "amount" in trade:
                # Calculate pnl_pct if not provided
                entry_value = trade["entry_price"] * trade["amount"]
                pnl_pct = (trade["realized_pnl"] / entry_value) * 100 if entry_value > 0 else 0
                returns.append(pnl_pct / 100)
            else:
                # Skip trades without enough info
                continue

        if len(returns) < 2:
            return 0.0

        # Calculate mean and std dev
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return 0.0

        # Annualize
        annualized_return = mean_return * self.periods_per_year
        annualized_std = std_dev * math.sqrt(self.periods_per_year)

        # Sharpe ratio
        sharpe = (annualized_return - self.risk_free_rate) / annualized_std

        return sharpe

    def _calculate_sortino_ratio(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate Sortino ratio (only penalizes downside volatility).

        Args:
            trades: List of trade dicts

        Returns:
            Sortino ratio (annualized)
        """
        if len(trades) < 2:
            return 0.0

        # Calculate returns
        returns = []
        for trade in trades:
            pnl_pct = trade.get("pnl_pct")
            if pnl_pct is not None:
                returns.append(pnl_pct / 100)
            elif "realized_pnl" in trade and "entry_price" in trade and "amount" in trade:
                entry_value = trade["entry_price"] * trade["amount"]
                pnl_pct = (trade["realized_pnl"] / entry_value) * 100 if entry_value > 0 else 0
                returns.append(pnl_pct / 100)

        if len(returns) < 2:
            return 0.0

        mean_return = sum(returns) / len(returns)

        # Downside deviation (only negative returns)
        downside_returns = [r for r in returns if r < 0]

        if not downside_returns:
            # No losses - return high Sortino
            return 10.0

        downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
        downside_std = math.sqrt(downside_variance)

        if downside_std == 0:
            return 0.0

        # Annualize
        annualized_return = mean_return * self.periods_per_year
        annualized_downside_std = downside_std * math.sqrt(self.periods_per_year)

        # Sortino ratio
        sortino = (annualized_return - self.risk_free_rate) / annualized_downside_std

        return sortino

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown percentage.

        Drawdown = (Peak Equity - Current Equity) / Peak Equity * 100

        Returns:
            Current drawdown percentage (0-100)
        """
        if self.peak_equity == 0:
            return 0.0

        drawdown = ((self.peak_equity - self.current_equity) / self.peak_equity) * 100
        return max(0.0, drawdown)  # Drawdown can't be negative

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from trade history.

        Returns:
            Maximum drawdown percentage (0-100)
        """
        if not self.trades:
            return 0.0

        # Reconstruct equity curve
        equity = self.initial_equity
        peak = equity
        max_dd = 0.0

        for trade in self.trades:
            equity += trade.get("realized_pnl", 0)

            if equity > peak:
                peak = equity

            # Calculate drawdown at this point
            dd = ((peak - equity) / peak) * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        return max_dd

    def _calculate_consecutive_losses(self, trades: List[Dict[str, Any]]) -> int:
        """Calculate current consecutive losing trades.

        Args:
            trades: List of trade dicts

        Returns:
            Number of consecutive losses
        """
        if not trades:
            return 0

        consecutive = 0

        # Count from most recent backwards
        for trade in reversed(trades):
            is_loss = not trade.get("won", trade.get("realized_pnl", 0) <= 0)

            if is_loss:
                consecutive += 1
            else:
                break  # Stop at first win

        return consecutive

    def _update_period_tracking(self, trade_data: Dict[str, Any]):
        """Update daily and weekly trade tracking.

        Args:
            trade_data: Trade information dict
        """
        # Check if we need to reset daily tracking
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_trades = []
            self.last_reset_date = current_date

        # Check if we need to reset weekly tracking (Monday reset)
        current_time = datetime.now()
        days_since_weekly_reset = (current_time - self.last_weekly_reset).days

        if days_since_weekly_reset >= 7:
            self.weekly_trades = []
            self.last_weekly_reset = current_time

        # Add to tracking lists
        self.daily_trades.append(trade_data)
        self.weekly_trades.append(trade_data)

    def reset(self):
        """Reset all metrics to initial state."""
        self.trades.clear()
        self.current_equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.daily_trades.clear()
        self.weekly_trades.clear()
        self.last_reset_date = datetime.now().date()
        self.last_weekly_reset = datetime.now()
        self._cache_valid = False
        self._metrics_cache = None

        self.logger.info("Metrics tracker reset to initial state")

    def get_equity_curve(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get equity curve data points for charting.

        Args:
            limit: Maximum number of data points

        Returns:
            List of {timestamp, equity, pnl} dicts
        """
        equity_points = []
        equity = self.initial_equity

        # Get recent trades
        trades_list = list(self.trades)[-limit:] if len(self.trades) > limit else list(self.trades)

        for trade in trades_list:
            equity += trade.get("realized_pnl", 0)

            equity_points.append({
                "timestamp": trade.get("timestamp", int(datetime.now().timestamp() * 1000)),
                "equity": equity,
                "pnl": trade.get("realized_pnl", 0),
            })

        return equity_points
