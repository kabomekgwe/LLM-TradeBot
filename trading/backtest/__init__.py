"""Backtesting Module - Historical strategy validation and performance analysis.

This module provides comprehensive backtesting capabilities for the LLM-TradeBot
trading system. It enables testing strategies on historical data by replaying
market conditions through the 8-agent decision pipeline.

Components:
- BacktestEngine: Main orchestrator for running backtests
- MarketSimulator: Virtual order execution with slippage and fees
- BacktestReport: Report generation (markdown, HTML, JSON)
- HistoricalDataLoader: Historical data fetching and caching
- BacktestResult: Performance metrics and trade history

Example Usage:
    ```python
    from trading.backtest import BacktestEngine
    from datetime import datetime

    # Initialize backtest engine
    engine = BacktestEngine(spec_dir=Path("specs/001/"))

    # Run backtest
    result = await engine.run_backtest(
        symbol="BTC/USDT",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        initial_balance=10000.0
    )

    # Print results
    print(f"Total P&L: ${result.total_pnl:.2f}")
    print(f"Win Rate: {result.win_rate:.1f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown_pct:.1f}%")

    # Generate report
    from trading.backtest import BacktestReport
    report = BacktestReport()
    report.generate_markdown(result, Path("backtest_report.md"))
    ```

Features:
- Full 8-agent simulation on historical data
- Realistic order execution with slippage and fees
- Comprehensive performance metrics (Sharpe, Sortino, Drawdown)
- Beautiful markdown and HTML reports
- Efficient data caching to avoid repeated API calls
- Support for multiple timeframes (5m, 15m, 1h)

Performance Metrics:
- Returns: Total P&L, Total P&L %, Average Win, Average Loss
- Win Rate: Percentage of profitable trades
- Sharpe Ratio: Risk-adjusted returns (annualized)
- Sortino Ratio: Downside risk-adjusted returns
- Maximum Drawdown: Largest peak-to-trough decline
- Profit Factor: Ratio of gross profit to gross loss

Architecture:
The backtest engine wraps the existing TradingManager and replays historical
data through the unchanged 8-agent pipeline. This ensures backtest results
accurately reflect live trading behavior.

Data Flow:
1. HistoricalDataLoader fetches OHLCV data from exchange or cache
2. BacktestEngine feeds data to 8-agent pipeline candle-by-candle
3. MarketSimulator executes orders with realistic slippage/fees
4. BacktestResult aggregates metrics and trade history
5. BacktestReport generates human-readable summaries

Note:
- Backtests use paper trading mode by default for safety
- All configuration from TradingConfig is respected
- State isolation ensures backtests don't interfere with live trading
"""

from .engine import BacktestEngine, BacktestResult
from .simulator import MarketSimulator, SimulatedPosition
from .reports import BacktestReport
from .data_loader import HistoricalDataLoader

__all__ = [
    # Core classes
    "BacktestEngine",
    "BacktestResult",
    "MarketSimulator",
    "SimulatedPosition",
    "BacktestReport",
    "HistoricalDataLoader",
]

__version__ = "1.0.0"

# Module metadata
__author__ = "LLM-TradeBot Contributors"
__description__ = "Historical backtesting engine for trading strategy validation"
