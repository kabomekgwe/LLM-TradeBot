"""Risk Calculator - Advanced risk metrics and Monte Carlo simulation.

Provides professional risk assessment tools including Value-at-Risk (VaR),
Conditional Value-at-Risk (CVaR), Monte Carlo simulation, and stress testing.
"""

import logging
import random
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import math
from datetime import datetime

from ..memory.trade_history import TradeRecord


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a trading strategy."""

    # Value-at-Risk (VaR)
    var_95: float  # 95% confidence VaR
    var_99: float  # 99% confidence VaR
    cvar_95: float  # Conditional VaR (expected shortfall) at 95%
    cvar_99: float  # Conditional VaR at 99%

    # Risk ratios
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float  # Return / Max Drawdown

    # Drawdown metrics
    max_drawdown: float
    max_drawdown_pct: float
    avg_drawdown: float
    drawdown_duration_days: int

    # Volatility
    daily_volatility: float
    annualized_volatility: float

    # Distribution metrics
    skewness: float  # Asymmetry of returns
    kurtosis: float  # Tail risk / fat tails

    # Risk of ruin
    risk_of_ruin_pct: float  # Probability of losing X% of capital

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "avg_drawdown": self.avg_drawdown,
            "drawdown_duration_days": self.drawdown_duration_days,
            "daily_volatility": self.daily_volatility,
            "annualized_volatility": self.annualized_volatility,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "risk_of_ruin_pct": self.risk_of_ruin_pct,
        }


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""

    num_simulations: int
    time_horizon_days: int

    # Equity distribution
    mean_final_equity: float
    median_final_equity: float
    percentile_5: float  # 5th percentile (worst 5%)
    percentile_25: float
    percentile_75: float
    percentile_95: float  # 95th percentile (best 5%)

    # Risk metrics
    probability_of_loss: float  # % of simulations ending below initial capital
    probability_of_ruin: float  # % hitting zero
    max_drawdown_distribution: List[float]

    # Full simulation results
    all_final_equities: List[float]

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding full simulation data)."""
        return {
            "num_simulations": self.num_simulations,
            "time_horizon_days": self.time_horizon_days,
            "mean_final_equity": self.mean_final_equity,
            "median_final_equity": self.median_final_equity,
            "percentile_5": self.percentile_5,
            "percentile_25": self.percentile_25,
            "percentile_75": self.percentile_75,
            "percentile_95": self.percentile_95,
            "probability_of_loss": self.probability_of_loss,
            "probability_of_ruin": self.probability_of_ruin,
        }


class RiskCalculator:
    """Advanced risk calculation and Monte Carlo simulation.

    Provides professional risk management tools for quantitative analysis.

    Features:
    - Value-at-Risk (VaR) calculation
    - Conditional VaR (CVaR) / Expected Shortfall
    - Monte Carlo simulation
    - Risk of ruin calculation
    - Stress testing

    Example:
        >>> calculator = RiskCalculator()
        >>> risk_metrics = calculator.calculate_risk_metrics(trades)
        >>> print(f"95% VaR: ${risk_metrics.var_95:.2f}")
        >>> print(f"Risk of Ruin: {risk_metrics.risk_of_ruin_pct:.1f}%")
    """

    def __init__(self):
        """Initialize risk calculator."""
        self.logger = logging.getLogger(__name__)

    def calculate_var(
        self,
        returns: List[float],
        confidence_level: float = 0.95,
    ) -> float:
        """Calculate Value-at-Risk (VaR).

        VaR = worst expected loss at a given confidence level

        Args:
            returns: List of return percentages
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR as a positive number (loss)

        Example:
            >>> returns = [-0.02, 0.01, -0.03, 0.02, -0.01]
            >>> var_95 = calculator.calculate_var(returns, 0.95)
            >>> print(f"95% VaR: {var_95:.2f}%")
        """
        if not returns:
            return 0.0

        # Sort returns from worst to best
        sorted_returns = sorted(returns)

        # Find the index for the confidence level
        index = int((1 - confidence_level) * len(sorted_returns))
        index = max(0, min(index, len(sorted_returns) - 1))

        # VaR is the negative of the return at this percentile
        var = -sorted_returns[index]

        return var

    def calculate_cvar(
        self,
        returns: List[float],
        confidence_level: float = 0.95,
    ) -> float:
        """Calculate Conditional VaR (CVaR) / Expected Shortfall.

        CVaR = average loss beyond the VaR threshold

        Args:
            returns: List of return percentages
            confidence_level: Confidence level

        Returns:
            CVaR as a positive number (average loss in tail)
        """
        if not returns:
            return 0.0

        sorted_returns = sorted(returns)

        # Get all returns worse than VaR
        var_threshold = -self.calculate_var(returns, confidence_level)
        tail_returns = [r for r in sorted_returns if r <= var_threshold]

        if not tail_returns:
            return 0.0

        # CVaR is average of tail losses
        cvar = -sum(tail_returns) / len(tail_returns)

        return cvar

    def calculate_skewness(self, returns: List[float]) -> float:
        """Calculate skewness of return distribution.

        Skewness > 0: More extreme positive returns (good)
        Skewness < 0: More extreme negative returns (bad)
        Skewness = 0: Symmetric distribution

        Args:
            returns: List of returns

        Returns:
            Skewness coefficient
        """
        if len(returns) < 3:
            return 0.0

        n = len(returns)
        mean = sum(returns) / n

        # Calculate variance
        variance = sum((r - mean) ** 2 for r in returns) / n
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return 0.0

        # Calculate skewness
        skewness = sum((r - mean) ** 3 for r in returns) / (n * std_dev ** 3)

        return skewness

    def calculate_kurtosis(self, returns: List[float]) -> float:
        """Calculate kurtosis of return distribution.

        Kurtosis > 3: Fat tails (more extreme events than normal)
        Kurtosis = 3: Normal distribution
        Kurtosis < 3: Thin tails

        Args:
            returns: List of returns

        Returns:
            Excess kurtosis (kurtosis - 3)
        """
        if len(returns) < 4:
            return 0.0

        n = len(returns)
        mean = sum(returns) / n

        # Calculate variance
        variance = sum((r - mean) ** 2 for r in returns) / n
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return 0.0

        # Calculate kurtosis
        kurtosis = sum((r - mean) ** 4 for r in returns) / (n * std_dev ** 4)

        # Return excess kurtosis
        return kurtosis - 3

    def calculate_risk_of_ruin(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        initial_balance: float,
        ruin_threshold: float = 0.5,
    ) -> float:
        """Calculate probability of losing X% of capital (risk of ruin).

        Uses gambler's ruin formula for trading.

        Args:
            win_rate: Win rate (e.g., 0.55 for 55%)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive)
            initial_balance: Starting capital
            ruin_threshold: Fraction of capital loss considered ruin (0.5 = 50%)

        Returns:
            Probability of ruin as percentage
        """
        if win_rate >= 1.0 or win_rate <= 0.0:
            return 0.0 if win_rate >= 1.0 else 100.0

        loss_rate = 1 - win_rate

        if avg_win == 0 or avg_loss == 0:
            return 50.0

        # Expected value per trade
        expected_value = (win_rate * avg_win) - (loss_rate * avg_loss)

        if expected_value <= 0:
            return 100.0  # Negative expectancy = eventual ruin

        # Risk/reward ratio
        risk_reward = avg_win / avg_loss

        # Calculate ruin probability (simplified formula)
        # P(ruin) = (loss/win)^N where N is number of trades to ruin
        trades_to_ruin = (initial_balance * ruin_threshold) / avg_loss

        if risk_reward >= 1:
            # Positive edge
            ruin_prob = (1 / risk_reward) ** trades_to_ruin
        else:
            # Negative edge
            ruin_prob = 1.0

        return min(ruin_prob * 100, 100.0)

    def calculate_risk_metrics(
        self,
        trades: List[TradeRecord],
        initial_balance: float = 10000.0,
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics from trade history.

        Args:
            trades: List of completed trades
            initial_balance: Starting capital

        Returns:
            RiskMetrics object with all risk calculations

        Example:
            >>> metrics = calculator.calculate_risk_metrics(trades)
            >>> print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
            >>> print(f"VaR (95%): ${metrics.var_95:.2f}")
        """
        closed_trades = [t for t in trades if t.closed]

        if not closed_trades:
            return RiskMetrics(
                var_95=0.0,
                var_99=0.0,
                cvar_95=0.0,
                cvar_99=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                avg_drawdown=0.0,
                drawdown_duration_days=0,
                daily_volatility=0.0,
                annualized_volatility=0.0,
                skewness=0.0,
                kurtosis=0.0,
                risk_of_ruin_pct=0.0,
            )

        # Extract returns
        returns = [t.pnl_pct / 100 for t in closed_trades]

        # Calculate VaR and CVaR
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        cvar_99 = self.calculate_cvar(returns, 0.99)

        # Calculate ratios (use formulas from trade_history.py)
        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)

        # Calculate drawdown metrics
        max_dd, max_dd_pct, avg_dd, dd_duration = self._calculate_drawdown_metrics(
            closed_trades, initial_balance
        )

        # Calmar ratio = Annual Return / Max Drawdown
        total_return = sum(t.realized_pnl for t in closed_trades)
        calmar = (total_return / initial_balance * 100 / max_dd_pct) if max_dd_pct > 0 else 0.0

        # Calculate volatility
        daily_vol = math.sqrt(sum(r ** 2 for r in returns) / len(returns))
        annual_vol = daily_vol * math.sqrt(252)

        # Distribution metrics
        skewness = self.calculate_skewness(returns)
        kurtosis = self.calculate_kurtosis(returns)

        # Risk of ruin
        winning = [t for t in closed_trades if t.won]
        losing = [t for t in closed_trades if not t.won]
        win_rate = len(winning) / len(closed_trades)
        avg_win = (sum(t.realized_pnl for t in winning) / len(winning)) if winning else 0.0
        avg_loss = abs((sum(t.realized_pnl for t in losing) / len(losing))) if losing else 0.0

        risk_of_ruin = self.calculate_risk_of_ruin(
            win_rate, avg_win, avg_loss, initial_balance, ruin_threshold=0.5
        )

        return RiskMetrics(
            var_95=var_95 * 100,  # Convert to percentage
            var_99=var_99 * 100,
            cvar_95=cvar_95 * 100,
            cvar_99=cvar_99 * 100,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            avg_drawdown=avg_dd,
            drawdown_duration_days=dd_duration,
            daily_volatility=daily_vol * 100,
            annualized_volatility=annual_vol * 100,
            skewness=skewness,
            kurtosis=kurtosis,
            risk_of_ruin_pct=risk_of_ruin,
        )

    def run_monte_carlo_simulation(
        self,
        trades: List[TradeRecord],
        initial_balance: float = 10000.0,
        num_simulations: int = 10000,
        time_horizon_days: int = 365,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation to project future performance.

        Simulates future trading by randomly sampling from historical trade distribution.

        Args:
            trades: Historical trades to sample from
            initial_balance: Starting capital for each simulation
            num_simulations: Number of simulations to run
            time_horizon_days: Simulation period in days

        Returns:
            MonteCarloResult with simulation statistics

        Example:
            >>> mc_result = calculator.run_monte_carlo_simulation(trades, 10000, 1000, 365)
            >>> print(f"Mean final equity: ${mc_result.mean_final_equity:,.2f}")
            >>> print(f"Probability of loss: {mc_result.probability_of_loss:.1f}%")
        """
        closed_trades = [t for t in trades if t.closed]

        if not closed_trades:
            raise ValueError("No closed trades available for simulation")

        # Calculate trades per day
        if len(closed_trades) > 1:
            time_span_days = (
                (closed_trades[-1].close_timestamp - closed_trades[0].timestamp)
                / (1000 * 86400)
            )
            trades_per_day = len(closed_trades) / time_span_days if time_span_days > 0 else 1
        else:
            trades_per_day = 1

        num_trades_to_simulate = int(trades_per_day * time_horizon_days)

        self.logger.info(
            f"Running Monte Carlo: {num_simulations} simulations, "
            f"{num_trades_to_simulate} trades each"
        )

        # Run simulations
        final_equities = []
        max_drawdowns = []
        num_ruined = 0

        for _ in range(num_simulations):
            equity = initial_balance
            peak = initial_balance
            max_dd = 0.0

            # Simulate trades by random sampling
            for _ in range(num_trades_to_simulate):
                # Randomly sample a trade from history
                sampled_trade = random.choice(closed_trades)

                # Apply the trade's return
                equity += (equity * sampled_trade.pnl_pct / 100)

                # Track drawdown
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, drawdown)

                # Check for ruin
                if equity <= 0:
                    num_ruined += 1
                    break

            final_equities.append(max(equity, 0))
            max_drawdowns.append(max_dd)

        # Calculate statistics
        sorted_equities = sorted(final_equities)
        mean_equity = sum(final_equities) / len(final_equities)
        median_equity = sorted_equities[len(sorted_equities) // 2]

        percentile_5 = sorted_equities[int(0.05 * len(sorted_equities))]
        percentile_25 = sorted_equities[int(0.25 * len(sorted_equities))]
        percentile_75 = sorted_equities[int(0.75 * len(sorted_equities))]
        percentile_95 = sorted_equities[int(0.95 * len(sorted_equities))]

        probability_of_loss = sum(1 for e in final_equities if e < initial_balance) / num_simulations * 100
        probability_of_ruin = (num_ruined / num_simulations) * 100

        return MonteCarloResult(
            num_simulations=num_simulations,
            time_horizon_days=time_horizon_days,
            mean_final_equity=mean_equity,
            median_final_equity=median_equity,
            percentile_5=percentile_5,
            percentile_25=percentile_25,
            percentile_75=percentile_75,
            percentile_95=percentile_95,
            probability_of_loss=probability_of_loss,
            probability_of_ruin=probability_of_ruin,
            max_drawdown_distribution=max_drawdowns,
            all_final_equities=final_equities,
        )

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio (simplified)."""
        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return 0.0

        # Annualize
        sharpe = (mean_return / std_dev) * math.sqrt(252)
        return sharpe

    def _calculate_sortino(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (simplified)."""
        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]

        if not downside:
            return 0.0

        downside_var = sum(r ** 2 for r in downside) / len(downside)
        downside_std = math.sqrt(downside_var)

        if downside_std == 0:
            return 0.0

        sortino = (mean_return / downside_std) * math.sqrt(252)
        return sortino

    def _calculate_drawdown_metrics(
        self,
        trades: List[TradeRecord],
        initial_balance: float,
    ) -> Tuple[float, float, float, int]:
        """Calculate drawdown metrics.

        Returns:
            (max_drawdown_usd, max_drawdown_pct, avg_drawdown_pct, duration_days)
        """
        sorted_trades = sorted(trades, key=lambda t: t.close_timestamp or 0)

        equity = initial_balance
        peak = initial_balance
        max_dd_usd = 0.0
        max_dd_pct = 0.0
        all_drawdowns = []
        dd_duration = 0
        current_dd_start = None

        for trade in sorted_trades:
            equity += trade.realized_pnl

            if equity > peak:
                peak = equity
                if current_dd_start:
                    # Drawdown recovered
                    current_dd_start = None
            else:
                # In drawdown
                if current_dd_start is None:
                    current_dd_start = trade.close_timestamp

                dd = peak - equity
                dd_pct = (dd / peak * 100) if peak > 0 else 0

                all_drawdowns.append(dd_pct)

                if dd > max_dd_usd:
                    max_dd_usd = dd
                    max_dd_pct = dd_pct

                # Calculate duration
                if current_dd_start and trade.close_timestamp:
                    duration = (trade.close_timestamp - current_dd_start) / (1000 * 86400)
                    dd_duration = max(dd_duration, int(duration))

        avg_dd = sum(all_drawdowns) / len(all_drawdowns) if all_drawdowns else 0.0

        return max_dd_usd, max_dd_pct, avg_dd, dd_duration
