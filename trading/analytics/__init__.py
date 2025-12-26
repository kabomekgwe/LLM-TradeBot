"""Advanced Analytics - Risk metrics, performance tracking, and attribution.

This module provides professional-grade analytics for trading performance:
- Risk Calculator: VaR, CVaR, Monte Carlo simulation
- Performance Tracker: Time-series metrics, benchmarking
- Attribution Analyzer: Agent and strategy performance breakdown

Components:
- RiskCalculator: Value-at-Risk, stress testing, risk of ruin
- PerformanceTracker: Daily/weekly/monthly tracking, equity curves
- AttributionAnalyzer: Regime, confidence, agent performance analysis

Example Usage:
    ```python
    from trading.analytics import RiskCalculator, PerformanceTracker, AttributionAnalyzer
    from trading.memory.trade_history import TradeJournal
    from pathlib import Path

    # Initialize
    journal = TradeJournal(Path("specs/001"))
    trades = journal.get_all_trades()

    # Risk analysis
    risk_calc = RiskCalculator()
    risk_metrics = risk_calc.calculate_risk_metrics(trades)
    print(f"95% VaR: ${risk_metrics.var_95:.2f}")
    print(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
    print(f"Risk of Ruin: {risk_metrics.risk_of_ruin_pct:.1f}%")

    # Monte Carlo simulation
    mc_result = risk_calc.run_monte_carlo_simulation(
        trades=trades,
        initial_balance=10000,
        num_simulations=10000,
        time_horizon_days=365,
    )
    print(f"1-year projection:")
    print(f"  Mean: ${mc_result.mean_final_equity:,.2f}")
    print(f"  5th percentile: ${mc_result.percentile_5:,.2f}")
    print(f"  95th percentile: ${mc_result.percentile_95:,.2f}")
    print(f"  Probability of loss: {mc_result.probability_of_loss:.1f}%")

    # Performance tracking
    tracker = PerformanceTracker(journal)
    tracker.take_snapshot()  # Save current metrics

    daily = tracker.get_daily_performance()
    print(f"Today: {daily['trades']} trades, ${daily['pnl']:.2f} P&L")

    # Benchmark comparison
    comparison = tracker.compare_to_benchmark(buy_and_hold_return=50.0)
    print(f"Alpha: {comparison.alpha:.2f}%")
    print(f"Information Ratio: {comparison.information_ratio:.2f}")

    # Attribution analysis
    analyzer = AttributionAnalyzer()

    # By market regime
    by_regime = analyzer.analyze_by_regime(trades)
    for regime_attr in by_regime:
        print(f"{regime_attr.regime}: {regime_attr.win_rate:.1f}% win rate")

    # Bull vs Bear
    bull_bear = analyzer.analyze_bull_vs_bear(trades)
    print(f"Bull agent: {bull_bear['bull'].total_pnl:+.2f}")
    print(f"Bear agent: {bull_bear['bear'].total_pnl:+.2f}")

    # Best patterns
    patterns = analyzer.get_best_performing_patterns(trades)
    print(f"Best regime: {patterns['best_regime']['regime']}")
    print(f"Optimal confidence: {patterns['best_confidence']['name']}")

    # Full report
    print(analyzer.generate_attribution_report(trades))
    ```

Features:
- **Risk Metrics**: VaR (95%, 99%), CVaR, skewness, kurtosis
- **Monte Carlo**: 10,000 simulations for future projections
- **Performance Tracking**: Daily/weekly/monthly snapshots
- **Benchmarking**: Alpha, beta, information ratio
- **Attribution**: Regime, agent, confidence analysis

Risk Metrics Explained:
- **VaR (Value-at-Risk)**: Worst expected loss at confidence level
- **CVaR (Conditional VaR)**: Average loss beyond VaR threshold
- **Sharpe Ratio**: Risk-adjusted returns (> 1.0 is good, > 2.0 is excellent)
- **Sortino Ratio**: Like Sharpe but only penalizes downside volatility
- **Calmar Ratio**: Annual return / max drawdown
- **Skewness**: Asymmetry of returns (positive = more upside, negative = more downside)
- **Kurtosis**: Tail risk (> 0 = fatter tails than normal, more extreme events)
- **Risk of Ruin**: Probability of losing 50% of capital

Performance Patterns:
- Best/worst market regimes
- Optimal decision confidence ranges
- Bull vs bear agent effectiveness
- Time-of-day performance
- Trade duration sweet spots

Note:
- Monte Carlo simulations assume trade returns follow historical distribution
- Risk metrics require at least 30 trades for statistical significance
- Attribution analysis helps identify edge and areas to improve
"""

from .risk_calculator import (
    RiskCalculator,
    RiskMetrics,
    MonteCarloResult,
)
from .performance import (
    PerformanceTracker,
    PerformanceSnapshot,
    BenchmarkComparison,
)
from .attribution import (
    AttributionAnalyzer,
    AgentAttribution,
    RegimeAttribution,
)

__all__ = [
    # Risk analysis
    "RiskCalculator",
    "RiskMetrics",
    "MonteCarloResult",

    # Performance tracking
    "PerformanceTracker",
    "PerformanceSnapshot",
    "BenchmarkComparison",

    # Attribution analysis
    "AttributionAnalyzer",
    "AgentAttribution",
    "RegimeAttribution",
]

__version__ = "1.0.0"

# Module metadata
__author__ = "LLM-TradeBot Contributors"
__description__ = "Advanced analytics: risk metrics, performance tracking, attribution analysis"
