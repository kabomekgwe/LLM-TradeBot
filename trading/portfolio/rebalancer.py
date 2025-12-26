"""Portfolio Rebalancer - Automated portfolio rebalancing strategies.

Implements various rebalancing strategies:
- Periodic rebalancing (time-based)
- Threshold rebalancing (drift-based)
- Volatility-based rebalancing
- Target allocation rebalancing
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

from .manager import PortfolioManager


class RebalanceStrategy(str, Enum):
    """Portfolio rebalancing strategies."""
    PERIODIC = "periodic"  # Rebalance on fixed schedule
    THRESHOLD = "threshold"  # Rebalance when drift exceeds threshold
    VOLATILITY = "volatility"  # Rebalance based on volatility changes
    TARGET = "target"  # Rebalance to specific target allocation


class PortfolioRebalancer:
    """Automated portfolio rebalancing.

    Manages portfolio rebalancing to maintain target allocations
    and risk levels across multiple assets.

    Example:
        >>> rebalancer = PortfolioRebalancer(portfolio)
        >>> actions = await rebalancer.rebalance_to_target({
        ...     "BTC/USDT": 0.6,
        ...     "ETH/USDT": 0.4,
        ... })
    """

    def __init__(
        self,
        portfolio: PortfolioManager,
        strategy: RebalanceStrategy = RebalanceStrategy.THRESHOLD,
        rebalance_threshold: float = 0.05,  # 5% drift
        rebalance_interval_days: int = 7,
        min_trade_size_usd: float = 100,
    ):
        """Initialize portfolio rebalancer.

        Args:
            portfolio: PortfolioManager instance
            strategy: Rebalancing strategy
            rebalance_threshold: Drift threshold for threshold strategy (0-1)
            rebalance_interval_days: Days between rebalances for periodic strategy
            min_trade_size_usd: Minimum trade size in USD
        """
        self.logger = logging.getLogger(__name__)

        self.portfolio = portfolio
        self.strategy = strategy
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_interval = timedelta(days=rebalance_interval_days)
        self.min_trade_size_usd = min_trade_size_usd

        # Last rebalance timestamp
        self.last_rebalance: Optional[datetime] = None

    async def should_rebalance(self, target_allocation: Dict[str, float]) -> bool:
        """Check if portfolio should be rebalanced.

        Args:
            target_allocation: Target allocation percentages

        Returns:
            True if rebalancing is recommended
        """
        if self.strategy == RebalanceStrategy.PERIODIC:
            return self._should_rebalance_periodic()

        elif self.strategy == RebalanceStrategy.THRESHOLD:
            return self._should_rebalance_threshold(target_allocation)

        elif self.strategy == RebalanceStrategy.VOLATILITY:
            return self._should_rebalance_volatility()

        elif self.strategy == RebalanceStrategy.TARGET:
            # Always rebalance to target
            return True

        return False

    def _should_rebalance_periodic(self) -> bool:
        """Check if periodic rebalance is due.

        Returns:
            True if rebalance interval has passed
        """
        if self.last_rebalance is None:
            return True

        time_since_rebalance = datetime.now() - self.last_rebalance
        return time_since_rebalance >= self.rebalance_interval

    def _should_rebalance_threshold(self, target_allocation: Dict[str, float]) -> bool:
        """Check if allocation drift exceeds threshold.

        Args:
            target_allocation: Target allocation percentages

        Returns:
            True if drift exceeds threshold
        """
        current_allocation = self.portfolio.get_allocation()

        # Calculate maximum drift
        max_drift = 0.0

        for symbol, target_pct in target_allocation.items():
            current_pct = current_allocation.get(symbol, 0.0)
            drift = abs(current_pct - target_pct)

            if drift > max_drift:
                max_drift = drift

        self.logger.debug(f"Maximum allocation drift: {max_drift:.2%}")

        return max_drift > self.rebalance_threshold

    def _should_rebalance_volatility(self) -> bool:
        """Check if volatility changes warrant rebalancing.

        Returns:
            True if significant volatility changes detected
        """
        # Placeholder - would require volatility tracking
        self.logger.warning("Volatility-based rebalancing not yet implemented")
        return False

    async def rebalance_to_target(
        self,
        target_allocation: Dict[str, float],
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        """Rebalance portfolio to target allocation.

        Args:
            target_allocation: Target allocation percentages (should sum to ≤1)
            dry_run: If True, only return actions without executing

        Returns:
            List of rebalancing actions to take
        """
        # Validate target allocation
        total_target = sum(target_allocation.values())
        if total_target > 1.0:
            raise ValueError(f"Target allocation sums to {total_target:.2%}, must be ≤100%")

        self.logger.info(f"Rebalancing portfolio to target allocation...")

        # Get current state
        current_allocation = self.portfolio.get_allocation()
        total_value = self.portfolio.get_total_value()

        # Calculate required trades
        actions = []

        for symbol, target_pct in target_allocation.items():
            current_pct = current_allocation.get(symbol, 0.0)
            pct_diff = target_pct - current_pct

            # Calculate USD value difference
            target_value = total_value * target_pct
            current_position = self.portfolio.get_position(symbol)
            current_value = current_position.value if current_position else 0.0
            value_diff = target_value - current_value

            # Skip small adjustments
            if abs(value_diff) < self.min_trade_size_usd:
                continue

            # Determine action
            if value_diff > 0:
                # Need to buy
                action = {
                    'symbol': symbol,
                    'action': 'buy',
                    'current_pct': current_pct,
                    'target_pct': target_pct,
                    'value_diff': value_diff,
                    'amount_usd': value_diff,
                }
            else:
                # Need to sell
                action = {
                    'symbol': symbol,
                    'action': 'sell',
                    'current_pct': current_pct,
                    'target_pct': target_pct,
                    'value_diff': value_diff,
                    'amount_usd': abs(value_diff),
                }

            actions.append(action)

            self.logger.info(
                f"{action['action'].upper()} {symbol}: "
                f"{current_pct:.1%} → {target_pct:.1%} "
                f"(${abs(value_diff):,.2f})"
            )

        # Execute rebalancing (if not dry run)
        if not dry_run and actions:
            self.last_rebalance = datetime.now()

        return actions

    async def rebalance_equal_weight(self) -> List[Dict[str, Any]]:
        """Rebalance to equal weight allocation.

        Returns:
            List of rebalancing actions
        """
        # Get all symbols (excluding cash)
        symbols = list(self.portfolio.positions.keys())

        if not symbols:
            self.logger.warning("No positions to rebalance")
            return []

        # Equal weight allocation
        weight_per_symbol = 0.95 / len(symbols)  # Leave 5% in cash
        target_allocation = {symbol: weight_per_symbol for symbol in symbols}

        return await self.rebalance_to_target(target_allocation)

    def calculate_rebalance_cost(self, actions: List[Dict[str, Any]]) -> float:
        """Calculate estimated cost of rebalancing (fees + slippage).

        Args:
            actions: List of rebalancing actions

        Returns:
            Estimated total cost in USD
        """
        total_cost = 0.0

        # Assume 0.1% trading fee + 0.05% slippage
        fee_rate = 0.001
        slippage_rate = 0.0005
        total_rate = fee_rate + slippage_rate

        for action in actions:
            trade_value = action['amount_usd']
            cost = trade_value * total_rate
            total_cost += cost

        return total_cost

    def get_rebalance_benefit(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
    ) -> float:
        """Estimate benefit of rebalancing.

        Args:
            current_allocation: Current allocation
            target_allocation: Target allocation

        Returns:
            Estimated benefit score (higher = more beneficial)
        """
        # Calculate allocation distance (sum of squared differences)
        distance = 0.0

        all_symbols = set(current_allocation.keys()) | set(target_allocation.keys())

        for symbol in all_symbols:
            current = current_allocation.get(symbol, 0.0)
            target = target_allocation.get(symbol, 0.0)
            distance += (current - target) ** 2

        # Benefit is proportional to distance from target
        benefit = distance * 100  # Scale for readability

        return benefit

    def should_rebalance_cost_benefit(
        self,
        target_allocation: Dict[str, float],
        cost_threshold: float = 0.01,  # 1% of portfolio
    ) -> bool:
        """Check if rebalancing is worthwhile considering costs.

        Args:
            target_allocation: Target allocation
            cost_threshold: Maximum acceptable cost as fraction of portfolio

        Returns:
            True if benefit outweighs cost
        """
        # Get rebalancing actions
        actions = self.rebalance_to_target(target_allocation, dry_run=True)

        if not actions:
            return False

        # Calculate costs and benefits
        cost = self.calculate_rebalance_cost(actions)
        current_allocation = self.portfolio.get_allocation()
        benefit = self.get_rebalance_benefit(current_allocation, target_allocation)

        # Cost as percentage of portfolio
        total_value = self.portfolio.get_total_value()
        cost_pct = cost / total_value if total_value > 0 else 0

        self.logger.debug(
            f"Rebalance analysis: Benefit={benefit:.2f}, "
            f"Cost=${cost:,.2f} ({cost_pct:.2%})"
        )

        # Rebalance if benefit is high and cost is acceptable
        return benefit > 1.0 and cost_pct < cost_threshold

    def __repr__(self) -> str:
        """String representation."""
        return f"PortfolioRebalancer(strategy={self.strategy.value}, threshold={self.rebalance_threshold:.1%})"
