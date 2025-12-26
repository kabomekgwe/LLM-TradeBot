"""Backtesting Engine - Historical strategy testing with 8-agent simulation.

This module provides the BacktestEngine class that enables testing trading strategies
on historical data by replaying market conditions through the existing 8-agent system.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import pandas as pd

from ..config import TradingConfig
from ..state import TradingState
from ..manager import TradingManager
from ..models.market_data import OHLCV
from ..memory.trade_history import TradeJournal, TradeRecord

from .simulator import MarketSimulator
from .reports import BacktestReport
from .data_loader import HistoricalDataLoader


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    # Identification
    backtest_id: str
    symbol: str
    start_date: datetime
    end_date: datetime

    # Configuration
    config: Dict[str, Any]

    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # Execution stats
    avg_trade_duration_hours: float = 0.0
    candles_processed: int = 0
    execution_time_seconds: float = 0.0

    # Trade records
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "backtest_id": self.backtest_id,
            "symbol": self.symbol,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "config": self.config,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "avg_trade_duration_hours": self.avg_trade_duration_hours,
            "candles_processed": self.candles_processed,
            "execution_time_seconds": self.execution_time_seconds,
            "trades": [t.to_dict() for t in self.trades],
            "equity_curve": self.equity_curve,
        }


class BacktestEngine:
    """Backtesting engine for historical strategy validation.

    Wraps the existing TradingManager and replays historical data through
    the 8-agent decision pipeline. Simulates order execution with slippage
    and fees for realistic performance estimation.

    Example:
        >>> engine = BacktestEngine(spec_dir, config)
        >>> result = await engine.run_backtest(
        ...     symbol="BTC/USDT",
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 6, 30)
        ... )
        >>> print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    """

    def __init__(self, spec_dir: Path, config: Optional[TradingConfig] = None):
        """Initialize backtest engine.

        Args:
            spec_dir: Specification directory for state isolation
            config: Trading configuration (if None, loads from env)
        """
        self.spec_dir = spec_dir
        self.config = config or TradingConfig.from_env("paper")  # Default to paper for safety
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.data_loader = HistoricalDataLoader()
        self.simulator = MarketSimulator(self.config)
        self.report_generator = BacktestReport()

    async def run_backtest(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 10000.0,
    ) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            start_date: Backtest start date
            end_date: Backtest end date
            initial_balance: Starting capital in USD

        Returns:
            BacktestResult with performance metrics and trade history

        Example:
            >>> result = await engine.run_backtest(
            ...     "BTC/USDT",
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 6, 30),
            ...     initial_balance=10000.0
            ... )
        """
        start_time = datetime.now()
        backtest_id = f"backtest_{int(start_time.timestamp())}"

        self.logger.info(
            f"Starting backtest: {symbol} from {start_date} to {end_date}"
        )

        # Load historical data
        self.logger.info("Loading historical data...")
        historical_data = await self.data_loader.load_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframes=["5m", "15m", "1h"],
        )

        if not historical_data:
            raise ValueError("No historical data available for backtest period")

        # Initialize backtest state
        backtest_state = self._initialize_backtest_state(initial_balance)

        # Initialize market simulator
        self.simulator.reset(initial_balance)

        # Run simulation through historical data
        trades = await self._simulate_trading(
            symbol=symbol,
            historical_data=historical_data,
            backtest_state=backtest_state,
        )

        # Calculate performance metrics
        result = self._calculate_metrics(
            backtest_id=backtest_id,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            trades=trades,
            equity_curve=self.simulator.equity_curve,
            initial_balance=initial_balance,
            candles_processed=len(historical_data["1h"]),
        )

        # Add execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        result.execution_time_seconds = execution_time

        # Store result
        self._save_backtest_result(result)

        self.logger.info(
            f"Backtest complete: {result.total_trades} trades, "
            f"Win Rate: {result.win_rate:.1f}%, "
            f"Sharpe: {result.sharpe_ratio:.2f}, "
            f"Max DD: {result.max_drawdown_pct:.1f}%"
        )

        return result

    def _initialize_backtest_state(self, initial_balance: float) -> TradingState:
        """Initialize isolated state for backtest."""
        state = TradingState()
        state.initialized = True
        state.provider = "backtest"
        state.created_at = datetime.now().isoformat()
        return state

    async def _simulate_trading(
        self,
        symbol: str,
        historical_data: Dict[str, List[OHLCV]],
        backtest_state: TradingState,
    ) -> List[TradeRecord]:
        """Simulate trading by replaying historical data through 8-agent pipeline.

        This is the core of the backtesting engine. It:
        1. Iterates through historical candles
        2. Feeds data to the 8-agent decision system
        3. Simulates order execution with slippage/fees
        4. Tracks positions and P&L
        """
        trades: List[TradeRecord] = []

        # Get 1h candles as the primary timeframe
        candles_1h = historical_data["1h"]

        self.logger.info(f"Simulating {len(candles_1h)} candles...")

        # Create a paper trading manager for backtest
        # We'll override its provider with our simulator
        manager = TradingManager(self.spec_dir, provider="paper")
        manager.state = backtest_state

        for i, candle in enumerate(candles_1h):
            # Skip initial warmup period (need enough data for indicators)
            if i < 50:
                continue

            # Prepare market data context (last 200 candles for each timeframe)
            market_data = self._prepare_market_data_context(
                historical_data, current_index=i
            )

            # Update simulator with current candle
            self.simulator.update_candle(candle)

            try:
                # Run 8-agent decision loop
                # Note: We're reusing the existing TradingManager pipeline
                # The simulator intercepts create_order() calls
                decision_context = await self._run_agent_pipeline(
                    manager=manager,
                    symbol=symbol,
                    market_data=market_data,
                )

                # Extract decision
                decision = decision_context.get("decision", {})
                execution = decision_context.get("execution", {})

                # Check if trade was executed
                if execution.get("success"):
                    # Simulate order execution
                    order = execution.get("order")
                    simulated_order = self.simulator.execute_order(
                        symbol=symbol,
                        side=order.side,
                        amount=order.amount,
                        current_candle=candle,
                    )

                    # Create trade record
                    trade = TradeRecord(
                        trade_id=simulated_order.id,
                        symbol=symbol,
                        timestamp=candle.timestamp,
                        side=decision.get("action"),
                        order_type="market",
                        amount=simulated_order.amount,
                        entry_price=simulated_order.price,
                        market_regime=decision_context.get("regime"),
                        bull_confidence=decision_context.get("bull_vote", {}).get("confidence"),
                        bear_confidence=decision_context.get("bear_vote", {}).get("confidence"),
                        decision_confidence=decision.get("confidence"),
                    )

                    trades.append(trade)

            except Exception as e:
                self.logger.warning(f"Error at candle {i}: {e}")
                continue

        # Close any remaining positions
        final_trades = self.simulator.close_all_positions(candles_1h[-1])
        trades.extend(final_trades)

        return trades

    def _prepare_market_data_context(
        self,
        historical_data: Dict[str, List[OHLCV]],
        current_index: int,
    ) -> Dict[str, Any]:
        """Prepare market data context for agents (last 200 candles per timeframe)."""
        context = {}

        # 1h timeframe (primary)
        candles_1h = historical_data["1h"]
        start_idx = max(0, current_index - 200)
        context["1h"] = candles_1h[start_idx:current_index]

        # 15m timeframe (4x more candles)
        if "15m" in historical_data:
            candles_15m = historical_data["15m"]
            start_idx_15m = max(0, current_index * 4 - 200)
            context["15m"] = candles_15m[start_idx_15m:current_index * 4]

        # 5m timeframe (12x more candles)
        if "5m" in historical_data:
            candles_5m = historical_data["5m"]
            start_idx_5m = max(0, current_index * 12 - 200)
            context["5m"] = candles_5m[start_idx_5m:current_index * 12]

        # Add ticker (synthesized from current candle)
        current_candle = candles_1h[current_index]
        context["ticker"] = {
            "symbol": "BTC/USDT",
            "last": current_candle.close,
            "bid": current_candle.close * 0.9995,  # 0.05% spread
            "ask": current_candle.close * 1.0005,
        }

        return context

    async def _run_agent_pipeline(
        self,
        manager: TradingManager,
        symbol: str,
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the 8-agent pipeline with historical data.

        This reuses the existing TradingManager's agent coordination logic
        but feeds it historical data instead of live data.
        """
        context = {
            "symbol": symbol,
            "state": manager.state,
            "market_data": market_data,
        }

        # Run all agents sequentially (same as TradingManager.run_trading_loop)
        context.update(await manager.agents["data_sync"].execute(context))
        context.update(await manager.agents["quant_analyst"].execute(context))
        context.update(await manager.agents["predict"].execute(context))

        # Bull + Bear
        bull_result = await manager.agents["bull"].execute(context)
        context.update(bull_result)
        bear_result = await manager.agents["bear"].execute(context)
        context.update(bear_result)

        # Decision aggregation
        context.update(await manager.agents["decision_core"].execute(context))

        # Risk veto
        context.update(await manager.agents["risk_audit"].execute(context))

        # Check veto
        risk_audit = context.get("risk_audit", {})
        if not risk_audit.get("veto", False):
            # Execute (will be simulated by MarketSimulator)
            context.update(await manager.agents["execution"].execute(context))

        return context

    def _calculate_metrics(
        self,
        backtest_id: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        trades: List[TradeRecord],
        equity_curve: List[float],
        initial_balance: float,
        candles_processed: int,
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics."""
        result = BacktestResult(
            backtest_id=backtest_id,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            config=self.config.__dict__,
            trades=trades,
            equity_curve=equity_curve,
            candles_processed=candles_processed,
        )

        if not trades:
            return result

        # Basic trade stats
        closed_trades = [t for t in trades if t.closed]
        winning_trades = [t for t in closed_trades if t.won]
        losing_trades = [t for t in closed_trades if not t.won]

        result.total_trades = len(trades)
        result.winning_trades = len(winning_trades)
        result.losing_trades = len(losing_trades)
        result.win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0.0

        # P&L metrics
        total_pnl = sum(t.realized_pnl for t in closed_trades)
        result.total_pnl = total_pnl
        result.total_pnl_pct = (total_pnl / initial_balance) * 100

        total_wins = sum(t.realized_pnl for t in winning_trades)
        total_losses = abs(sum(t.realized_pnl for t in losing_trades))

        result.avg_win = (total_wins / len(winning_trades)) if winning_trades else 0.0
        result.avg_loss = (total_losses / len(losing_trades)) if losing_trades else 0.0
        result.largest_win = max((t.realized_pnl for t in winning_trades), default=0.0)
        result.largest_loss = min((t.realized_pnl for t in losing_trades), default=0.0)
        result.profit_factor = (total_wins / total_losses) if total_losses > 0 else 0.0

        # Risk metrics
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()

            # Sharpe ratio (annualized)
            if returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5)

            # Sortino ratio (annualized, using downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                result.sortino_ratio = (returns.mean() / downside_returns.std()) * (252 ** 0.5)

            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            result.max_drawdown_pct = abs(drawdown.min() * 100)
            result.max_drawdown = abs(drawdown.min() * initial_balance)

        # Trade duration
        completed_trades = [t for t in trades if t.closed and t.close_timestamp]
        if completed_trades:
            durations = [
                (t.close_timestamp - t.timestamp) / (1000 * 3600)  # Convert ms to hours
                for t in completed_trades
            ]
            result.avg_trade_duration_hours = sum(durations) / len(durations)

        return result

    def _save_backtest_result(self, result: BacktestResult):
        """Save backtest result to file."""
        # Store in spec's memory directory
        backtest_dir = self.spec_dir / "memory" / "backtests"
        backtest_dir.mkdir(parents=True, exist_ok=True)

        import json
        filepath = backtest_dir / f"{result.backtest_id}.json"

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        self.logger.info(f"Backtest result saved to {filepath}")
