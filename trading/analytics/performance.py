"""Performance Tracker - Real-time and historical performance analytics.

Tracks trading performance metrics over time with time-series analysis,
benchmark comparison, and performance attribution.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

from ..memory.trade_history import TradeRecord, TradeJournal


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance metrics."""

    timestamp: datetime
    total_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown_pct: float
    equity: float

    # Period-specific metrics
    period_trades: int = 0
    period_pnl: float = 0.0
    period_return_pct: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "equity": self.equity,
            "period_trades": self.period_trades,
            "period_pnl": self.period_pnl,
            "period_return_pct": self.period_return_pct,
        }


@dataclass
class BenchmarkComparison:
    """Comparison against a benchmark (e.g., buy-and-hold)."""

    strategy_return: float
    benchmark_return: float
    alpha: float  # Excess return over benchmark
    beta: float  # Correlation with benchmark
    information_ratio: float  # Alpha / tracking error

    outperformance_pct: float
    correlation: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "strategy_return": self.strategy_return,
            "benchmark_return": self.benchmark_return,
            "alpha": self.alpha,
            "beta": self.beta,
            "information_ratio": self.information_ratio,
            "outperformance_pct": self.outperformance_pct,
            "correlation": self.correlation,
        }


class PerformanceTracker:
    """Real-time performance tracking and analytics.

    Features:
    - Time-series performance snapshots
    - Period-over-period comparisons (daily, weekly, monthly)
    - Benchmark comparison (vs buy-and-hold)
    - Performance attribution by strategy/regime
    - Equity curve generation

    Example:
        >>> tracker = PerformanceTracker(journal)
        >>> tracker.take_snapshot()
        >>> daily_perf = tracker.get_daily_performance()
        >>> print(f"Today's P&L: ${daily_perf['pnl']:.2f}")
    """

    def __init__(self, journal: TradeJournal, snapshot_dir: Optional[Path] = None):
        """Initialize performance tracker.

        Args:
            journal: TradeJournal instance
            snapshot_dir: Directory to store performance snapshots
        """
        self.journal = journal
        self.logger = logging.getLogger(__name__)

        # Snapshot storage
        if snapshot_dir is None:
            snapshot_dir = journal.spec_dir / "memory" / "performance"
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.snapshots: List[PerformanceSnapshot] = []
        self._load_snapshots()

    def _load_snapshots(self):
        """Load existing snapshots from disk."""
        snapshot_file = self.snapshot_dir / "snapshots.json"

        if snapshot_file.exists():
            try:
                with open(snapshot_file, "r") as f:
                    data = json.load(f)

                self.snapshots = [
                    PerformanceSnapshot(
                        timestamp=datetime.fromisoformat(s["timestamp"]),
                        total_trades=s["total_trades"],
                        win_rate=s["win_rate"],
                        total_pnl=s["total_pnl"],
                        sharpe_ratio=s["sharpe_ratio"],
                        max_drawdown_pct=s["max_drawdown_pct"],
                        equity=s["equity"],
                        period_trades=s.get("period_trades", 0),
                        period_pnl=s.get("period_pnl", 0.0),
                        period_return_pct=s.get("period_return_pct", 0.0),
                    )
                    for s in data
                ]

                self.logger.info(f"Loaded {len(self.snapshots)} performance snapshots")

            except Exception as e:
                self.logger.warning(f"Failed to load snapshots: {e}")

    def _save_snapshots(self):
        """Save snapshots to disk."""
        snapshot_file = self.snapshot_dir / "snapshots.json"

        try:
            with open(snapshot_file, "w") as f:
                json.dump([s.to_dict() for s in self.snapshots], f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save snapshots: {e}")

    def take_snapshot(self, initial_balance: float = 10000.0) -> PerformanceSnapshot:
        """Take a performance snapshot at current time.

        Args:
            initial_balance: Starting capital

        Returns:
            PerformanceSnapshot with current metrics

        Example:
            >>> snapshot = tracker.take_snapshot()
            >>> print(f"Equity: ${snapshot.equity:,.2f}")
        """
        # Get comprehensive metrics
        metrics = self.journal.calculate_comprehensive_metrics(
            trades=None, initial_balance=initial_balance
        )

        # Calculate period metrics (since last snapshot)
        period_trades = 0
        period_pnl = 0.0

        if self.snapshots:
            last_snapshot = self.snapshots[-1]
            period_trades = metrics["total_trades"] - last_snapshot.total_trades
            period_pnl = metrics["total_pnl"] - last_snapshot.total_pnl

        # Create snapshot
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            total_trades=metrics["total_trades"],
            win_rate=metrics["win_rate"],
            total_pnl=metrics["total_pnl"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown_pct=metrics["max_drawdown_pct"],
            equity=metrics["final_balance"],
            period_trades=period_trades,
            period_pnl=period_pnl,
            period_return_pct=metrics["return_pct"],
        )

        # Add to snapshots
        self.snapshots.append(snapshot)
        self._save_snapshots()

        self.logger.info(f"Performance snapshot taken: Equity ${snapshot.equity:,.2f}")

        return snapshot

    def get_daily_performance(self) -> Dict[str, Any]:
        """Get performance for the current day.

        Returns:
            Dictionary with daily metrics

        Example:
            >>> daily = tracker.get_daily_performance()
            >>> print(f"Today: {daily['trades']} trades, ${daily['pnl']:.2f} P&L")
        """
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Get today's trades
        all_trades = self.journal.get_all_trades()
        today_trades = [
            t for t in all_trades
            if t.close_timestamp
            and datetime.fromtimestamp(t.close_timestamp / 1000) >= today_start
        ]

        if not today_trades:
            return {
                "date": today_start.strftime("%Y-%m-%d"),
                "trades": 0,
                "pnl": 0.0,
                "win_rate": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
            }

        winning = [t for t in today_trades if t.won]
        total_pnl = sum(t.realized_pnl for t in today_trades)

        return {
            "date": today_start.strftime("%Y-%m-%d"),
            "trades": len(today_trades),
            "winning_trades": len(winning),
            "pnl": total_pnl,
            "win_rate": (len(winning) / len(today_trades) * 100),
            "best_trade": max((t.realized_pnl for t in today_trades), default=0.0),
            "worst_trade": min((t.realized_pnl for t in today_trades), default=0.0),
        }

    def get_weekly_performance(self) -> Dict[str, Any]:
        """Get performance for the current week."""
        week_start = datetime.now() - timedelta(days=datetime.now().weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

        all_trades = self.journal.get_all_trades()
        week_trades = [
            t for t in all_trades
            if t.close_timestamp
            and datetime.fromtimestamp(t.close_timestamp / 1000) >= week_start
        ]

        if not week_trades:
            return {
                "week_start": week_start.strftime("%Y-%m-%d"),
                "trades": 0,
                "pnl": 0.0,
            }

        return {
            "week_start": week_start.strftime("%Y-%m-%d"),
            "trades": len(week_trades),
            "pnl": sum(t.realized_pnl for t in week_trades),
            "win_rate": (len([t for t in week_trades if t.won]) / len(week_trades) * 100),
        }

    def get_monthly_performance(self) -> Dict[str, Any]:
        """Get performance for the current month."""
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        all_trades = self.journal.get_all_trades()
        month_trades = [
            t for t in all_trades
            if t.close_timestamp
            and datetime.fromtimestamp(t.close_timestamp / 1000) >= month_start
        ]

        if not month_trades:
            return {
                "month": month_start.strftime("%Y-%m"),
                "trades": 0,
                "pnl": 0.0,
            }

        return {
            "month": month_start.strftime("%Y-%m"),
            "trades": len(month_trades),
            "pnl": sum(t.realized_pnl for t in month_trades),
            "win_rate": (len([t for t in month_trades if t.won]) / len(month_trades) * 100),
        }

    def compare_to_benchmark(
        self,
        buy_and_hold_return: float,
        trades: Optional[List[TradeRecord]] = None,
    ) -> BenchmarkComparison:
        """Compare strategy performance to a benchmark (e.g., buy-and-hold).

        Args:
            buy_and_hold_return: Benchmark return percentage
            trades: Trades to analyze (None = all trades)

        Returns:
            BenchmarkComparison with alpha, beta, information ratio

        Example:
            >>> # BTC buy-and-hold returned 50% over the period
            >>> comparison = tracker.compare_to_benchmark(50.0)
            >>> print(f"Alpha: {comparison.alpha:.2f}%")
            >>> print(f"Outperformance: {comparison.outperformance_pct:.2f}%")
        """
        if trades is None:
            trades = self.journal.get_all_trades()

        closed_trades = [t for t in trades if t.closed]

        if not closed_trades:
            return BenchmarkComparison(
                strategy_return=0.0,
                benchmark_return=buy_and_hold_return,
                alpha=0.0,
                beta=0.0,
                information_ratio=0.0,
                outperformance_pct=0.0,
                correlation=0.0,
            )

        # Calculate strategy return
        total_pnl = sum(t.realized_pnl for t in closed_trades)
        initial_balance = 10000.0  # Assume
        strategy_return = (total_pnl / initial_balance) * 100

        # Alpha = Strategy Return - Benchmark Return
        alpha = strategy_return - buy_and_hold_return

        # Simplified beta calculation (correlation with market)
        # In reality, would need market returns for each trade period
        # Here we use 1.0 (perfect correlation) as simplification
        beta = 1.0

        # Information Ratio = Alpha / Tracking Error
        # Simplified: use standard deviation of returns as tracking error
        returns = [t.pnl_pct for t in closed_trades]
        import math
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        tracking_error = math.sqrt(variance)

        information_ratio = alpha / tracking_error if tracking_error > 0 else 0.0

        # Outperformance
        outperformance_pct = ((strategy_return / buy_and_hold_return) - 1) * 100 if buy_and_hold_return != 0 else 0.0

        # Correlation (simplified)
        correlation = 0.5  # Placeholder

        return BenchmarkComparison(
            strategy_return=strategy_return,
            benchmark_return=buy_and_hold_return,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            outperformance_pct=outperformance_pct,
            correlation=correlation,
        )

    def get_equity_curve(
        self,
        initial_balance: float = 10000.0,
        trades: Optional[List[TradeRecord]] = None,
    ) -> List[Tuple[datetime, float]]:
        """Generate equity curve over time.

        Args:
            initial_balance: Starting capital
            trades: Trades to include (None = all trades)

        Returns:
            List of (datetime, equity) tuples

        Example:
            >>> curve = tracker.get_equity_curve()
            >>> for date, equity in curve[-10:]:
            ...     print(f"{date.strftime('%Y-%m-%d')}: ${equity:,.2f}")
        """
        if trades is None:
            trades = self.journal.get_all_trades()

        closed_trades = sorted(
            [t for t in trades if t.closed and t.close_timestamp],
            key=lambda t: t.close_timestamp or 0,
        )

        if not closed_trades:
            return [(datetime.now(), initial_balance)]

        equity_curve = [(datetime.fromtimestamp(closed_trades[0].timestamp / 1000), initial_balance)]

        equity = initial_balance
        for trade in closed_trades:
            equity += trade.realized_pnl
            equity_curve.append((datetime.fromtimestamp(trade.close_timestamp / 1000), equity))

        return equity_curve

    def get_performance_by_time_of_day(
        self,
        trades: Optional[List[TradeRecord]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Analyze performance by hour of day.

        Args:
            trades: Trades to analyze

        Returns:
            Dictionary mapping hour (0-23) to performance metrics

        Example:
            >>> by_hour = tracker.get_performance_by_time_of_day()
            >>> print(f"Best hour: {max(by_hour, key=lambda h: by_hour[h]['pnl'])}")
        """
        if trades is None:
            trades = self.journal.get_all_trades()

        closed_trades = [t for t in trades if t.closed and t.close_timestamp]

        by_hour: Dict[int, List[TradeRecord]] = {i: [] for i in range(24)}

        for trade in closed_trades:
            hour = datetime.fromtimestamp(trade.close_timestamp / 1000).hour
            by_hour[hour].append(trade)

        results = {}
        for hour, hour_trades in by_hour.items():
            if not hour_trades:
                continue

            winning = [t for t in hour_trades if t.won]
            results[hour] = {
                "trades": len(hour_trades),
                "winning": len(winning),
                "win_rate": (len(winning) / len(hour_trades) * 100),
                "pnl": sum(t.realized_pnl for t in hour_trades),
            }

        return results

    def get_performance_summary(self) -> str:
        """Generate human-readable performance summary.

        Returns:
            Formatted summary string

        Example:
            >>> print(tracker.get_performance_summary())
        """
        daily = self.get_daily_performance()
        weekly = self.get_weekly_performance()
        monthly = self.get_monthly_performance()

        summary = "=== Performance Summary ===\n\n"
        summary += f"📅 Today ({daily['date']}):\n"
        summary += f"  Trades: {daily['trades']}\n"
        summary += f"  P&L: ${daily['pnl']:+,.2f}\n"
        summary += f"  Win Rate: {daily['win_rate']:.1f}%\n\n"

        summary += f"📊 This Week:\n"
        summary += f"  Trades: {weekly['trades']}\n"
        summary += f"  P&L: ${weekly['pnl']:+,.2f}\n"
        summary += f"  Win Rate: {weekly.get('win_rate', 0):.1f}%\n\n"

        summary += f"📈 This Month ({monthly['month']}):\n"
        summary += f"  Trades: {monthly['trades']}\n"
        summary += f"  P&L: ${monthly['pnl']:+,.2f}\n"
        summary += f"  Win Rate: {monthly.get('win_rate', 0):.1f}%\n"

        return summary
