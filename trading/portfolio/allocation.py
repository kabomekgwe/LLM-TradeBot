"""Allocation Optimizer - Risk-based position sizing strategies.

Implements advanced allocation methods:
- Equal Weight: Simple 1/N allocation
- Risk Parity: Equal risk contribution from each asset
- Kelly Criterion: Optimal sizing based on edge and volatility
- Minimum Variance: Minimize portfolio volatility
- Maximum Sharpe: Maximize risk-adjusted returns
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from enum import Enum
from scipy.optimize import minimize


class AllocationMethod(str, Enum):
    """Portfolio allocation methods."""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    KELLY_CRITERION = "kelly"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"


class AllocationOptimizer:
    """Optimize portfolio allocation using various strategies.

    Calculates optimal position sizes based on risk metrics,
    correlations, and expected returns.

    Example:
        >>> optimizer = AllocationOptimizer()
        >>> allocation = optimizer.optimize(
        ...     symbols=["BTC/USDT", "ETH/USDT"],
        ...     expected_returns=[0.15, 0.12],
        ...     volatilities=[0.80, 0.70],
        ...     correlation_matrix=corr_matrix,
        ...     method=AllocationMethod.RISK_PARITY
        ... )
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize allocation optimizer.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.logger = logging.getLogger(__name__)
        self.risk_free_rate = risk_free_rate

    def optimize(
        self,
        symbols: List[str],
        expected_returns: Optional[List[float]] = None,
        volatilities: Optional[List[float]] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT,
    ) -> Dict[str, float]:
        """Optimize portfolio allocation.

        Args:
            symbols: List of symbols
            expected_returns: Expected annual returns for each symbol
            volatilities: Annual volatilities for each symbol
            correlation_matrix: Correlation matrix
            method: Allocation method to use

        Returns:
            Dictionary mapping symbols to allocation weights (0-1)
        """
        n = len(symbols)

        if method == AllocationMethod.EQUAL_WEIGHT:
            return self._equal_weight(symbols)

        elif method == AllocationMethod.RISK_PARITY:
            if volatilities is None:
                raise ValueError("Volatilities required for risk parity")
            return self._risk_parity(symbols, volatilities, correlation_matrix)

        elif method == AllocationMethod.KELLY_CRITERION:
            if expected_returns is None or volatilities is None:
                raise ValueError("Expected returns and volatilities required for Kelly")
            return self._kelly_criterion(symbols, expected_returns, volatilities)

        elif method == AllocationMethod.MIN_VARIANCE:
            if volatilities is None or correlation_matrix is None:
                raise ValueError("Volatilities and correlation matrix required for min variance")
            return self._min_variance(symbols, volatilities, correlation_matrix)

        elif method == AllocationMethod.MAX_SHARPE:
            if expected_returns is None or volatilities is None or correlation_matrix is None:
                raise ValueError("All metrics required for max Sharpe")
            return self._max_sharpe(symbols, expected_returns, volatilities, correlation_matrix)

        else:
            raise ValueError(f"Unknown allocation method: {method}")

    def _equal_weight(self, symbols: List[str]) -> Dict[str, float]:
        """Equal weight allocation (1/N).

        Args:
            symbols: List of symbols

        Returns:
            Equal weight allocation
        """
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}

    def _risk_parity(
        self,
        symbols: List[str],
        volatilities: List[float],
        correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Risk parity allocation (equal risk contribution).

        Args:
            symbols: List of symbols
            volatilities: Annual volatilities
            correlation_matrix: Correlation matrix (optional)

        Returns:
            Risk parity allocation
        """
        n = len(symbols)

        if correlation_matrix is None:
            # Simple risk parity (no correlation)
            inv_vol = [1.0 / vol for vol in volatilities]
            total = sum(inv_vol)
            weights = [w / total for w in inv_vol]

        else:
            # Risk parity with correlation
            # Use optimization to find equal risk contribution

            def risk_contribution(weights, cov_matrix):
                """Calculate risk contribution for each asset."""
                portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
                marginal_contrib = np.dot(cov_matrix, weights)
                risk_contrib = weights * marginal_contrib / np.sqrt(portfolio_var)
                return risk_contrib

            def objective(weights, cov_matrix):
                """Minimize difference in risk contributions."""
                contrib = risk_contribution(weights, cov_matrix)
                # Minimize variance of risk contributions
                return np.var(contrib)

            # Build covariance matrix
            cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix.values

            # Optimize
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = tuple((0, 1) for _ in range(n))
            initial_weights = np.array([1.0 / n] * n)

            result = minimize(
                objective,
                initial_weights,
                args=(cov_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
            )

            weights = result.x

        return {symbol: float(weight) for symbol, weight in zip(symbols, weights)}

    def _kelly_criterion(
        self,
        symbols: List[str],
        expected_returns: List[float],
        volatilities: List[float],
    ) -> Dict[str, float]:
        """Kelly Criterion allocation.

        Args:
            symbols: List of symbols
            expected_returns: Expected annual returns
            volatilities: Annual volatilities

        Returns:
            Kelly allocation
        """
        # Kelly fraction: f = (expected return - risk free rate) / variance
        kelly_fractions = []

        for ret, vol in zip(expected_returns, volatilities):
            edge = ret - self.risk_free_rate
            variance = vol ** 2
            kelly_f = edge / variance if variance > 0 else 0

            # Apply fractional Kelly (half Kelly for safety)
            kelly_f = max(0, min(kelly_f * 0.5, 1.0))

            kelly_fractions.append(kelly_f)

        # Normalize to sum to 1
        total = sum(kelly_fractions)

        if total == 0:
            # Fall back to equal weight
            return self._equal_weight(symbols)

        weights = [f / total for f in kelly_fractions]

        return {symbol: float(weight) for symbol, weight in zip(symbols, weights)}

    def _min_variance(
        self,
        symbols: List[str],
        volatilities: List[float],
        correlation_matrix: pd.DataFrame,
    ) -> Dict[str, float]:
        """Minimum variance allocation.

        Args:
            symbols: List of symbols
            volatilities: Annual volatilities
            correlation_matrix: Correlation matrix

        Returns:
            Minimum variance allocation
        """
        n = len(symbols)

        # Build covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix.values

        def portfolio_variance(weights):
            """Calculate portfolio variance."""
            return np.dot(weights, np.dot(cov_matrix, weights))

        # Optimize
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        initial_weights = np.array([1.0 / n] * n)

        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )

        weights = result.x

        return {symbol: float(weight) for symbol, weight in zip(symbols, weights)}

    def _max_sharpe(
        self,
        symbols: List[str],
        expected_returns: List[float],
        volatilities: List[float],
        correlation_matrix: pd.DataFrame,
    ) -> Dict[str, float]:
        """Maximum Sharpe ratio allocation.

        Args:
            symbols: List of symbols
            expected_returns: Expected annual returns
            volatilities: Annual volatilities
            correlation_matrix: Correlation matrix

        Returns:
            Maximum Sharpe allocation
        """
        n = len(symbols)

        # Build covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix.values

        def negative_sharpe(weights):
            """Calculate negative Sharpe ratio (for minimization)."""
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

            if portfolio_vol == 0:
                return 0

            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe  # Negative for minimization

        # Optimize
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        initial_weights = np.array([1.0 / n] * n)

        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )

        weights = result.x

        return {symbol: float(weight) for symbol, weight in zip(symbols, weights)}

    def calculate_portfolio_metrics(
        self,
        allocation: Dict[str, float],
        expected_returns: List[float],
        volatilities: List[float],
        correlation_matrix: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate portfolio metrics for given allocation.

        Args:
            allocation: Portfolio allocation
            expected_returns: Expected returns
            volatilities: Volatilities
            correlation_matrix: Correlation matrix

        Returns:
            Dictionary with portfolio metrics
        """
        symbols = list(allocation.keys())
        weights = np.array([allocation[s] for s in symbols])

        # Expected return
        portfolio_return = np.dot(weights, expected_returns)

        # Volatility
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix.values
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

        # Sharpe ratio
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"AllocationOptimizer(risk_free_rate={self.risk_free_rate:.2%})"
