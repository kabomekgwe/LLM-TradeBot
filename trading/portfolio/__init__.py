"""Portfolio Management - Multi-symbol trading and optimization.

Manages portfolios of multiple cryptocurrency positions with:
- Multi-symbol position tracking
- Portfolio rebalancing strategies
- Asset correlation analysis
- Risk-based allocation (Risk Parity, Kelly Criterion)
- Diversification optimization

Components:
- PortfolioManager: Centralized portfolio tracking and management
- PortfolioRebalancer: Automated rebalancing strategies
- CorrelationAnalyzer: Asset correlation and dependency tracking
- AllocationOptimizer: Risk-based position sizing

Example Usage:
    ```python
    from trading.portfolio import PortfolioManager, PortfolioRebalancer

    # Initialize portfolio manager
    portfolio = PortfolioManager(
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        max_total_capital=10000,
        max_position_pct=0.4,  # Max 40% per asset
    )

    # Add positions
    await portfolio.add_position("BTC/USDT", amount=0.1, entry_price=42000)
    await portfolio.add_position("ETH/USDT", amount=2.0, entry_price=2200)

    # Get portfolio metrics
    metrics = portfolio.get_metrics()
    print(f"Total Value: ${metrics['total_value']:,.2f}")
    print(f"Diversification: {metrics['diversification_ratio']:.2f}")
    print(f"Portfolio Beta: {metrics['portfolio_beta']:.2f}")

    # Rebalance to target allocation
    rebalancer = PortfolioRebalancer(portfolio)
    rebalance_actions = await rebalancer.rebalance_to_target({
        "BTC/USDT": 0.5,  # 50%
        "ETH/USDT": 0.3,  # 30%
        "SOL/USDT": 0.2,  # 20%
    })

    # Execute rebalance
    for action in rebalance_actions:
        print(f"{action['action']} {action['amount']} {action['symbol']}")
    ```

Configuration:
    Set in TradingConfig or environment:
    ```bash
    # Portfolio settings
    PORTFOLIO_ENABLED=true
    PORTFOLIO_MAX_SYMBOLS=10
    PORTFOLIO_MAX_POSITION_PCT=0.4
    PORTFOLIO_REBALANCE_THRESHOLD=0.05  # 5% drift triggers rebalance

    # Allocation strategy
    PORTFOLIO_ALLOCATION_METHOD=risk_parity  # or 'equal_weight', 'kelly'
    PORTFOLIO_CORRELATION_WINDOW=30  # Days for correlation calculation
    ```

Allocation Methods:
    - **Equal Weight**: Simple 1/N allocation
    - **Risk Parity**: Equal risk contribution from each asset
    - **Kelly Criterion**: Optimal sizing based on edge and volatility
    - **Minimum Variance**: Minimize portfolio volatility
    - **Maximum Sharpe**: Maximize risk-adjusted returns

Risk Metrics:
    - Portfolio Value at Risk (VaR)
    - Conditional VaR (CVaR)
    - Correlation matrix
    - Diversification ratio
    - Portfolio beta
    - Concentration risk
"""

from .manager import PortfolioManager
from .rebalancer import PortfolioRebalancer, RebalanceStrategy
from .correlation import CorrelationAnalyzer
from .allocation import AllocationOptimizer, AllocationMethod

__all__ = [
    "PortfolioManager",
    "PortfolioRebalancer",
    "RebalanceStrategy",
    "CorrelationAnalyzer",
    "AllocationOptimizer",
    "AllocationMethod",
]

__version__ = "1.0.0"
