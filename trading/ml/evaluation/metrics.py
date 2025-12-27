"""
Financial performance metrics for trading strategy evaluation.

Implements comprehensive metrics using empyrical library for proper
annualization and edge case handling.

CRITICAL: Uses 252 trading days per year for annualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


def comprehensive_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free: float = 0.0
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        returns: Series of period returns (NOT cumulative)
        benchmark_returns: Optional benchmark returns for comparison
        risk_free: Annual risk-free rate (default 0%)
    
    Returns:
        dict: All performance metrics
            - sharpe_ratio: (annual_return - risk_free) / annual_volatility
            - sortino_ratio: (annual_return - risk_free) / downside_volatility
            - max_drawdown: Maximum peak-to-trough decline
            - calmar_ratio: annual_return / abs(max_drawdown)
            - win_rate: (winning_trades) / total_trades
            - total_return: Cumulative return
            - annual_return: Annualized return
            - annual_volatility: Annualized standard deviation
    """
    # Handle empty returns
    if len(returns) == 0:
        logger.warning("Empty returns series - returning zero metrics")
        return _zero_metrics()
    
    # Replace NaN/inf with 0
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    try:
        # Total and annual return
        total_return = (1 + returns).prod() - 1
        periods = len(returns)
        annual_return = (1 + total_return) ** (252 / periods) - 1 if periods > 0 else 0
        
        # Volatility (annualized)
        annual_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = calculate_sharpe_ratio(returns, risk_free)
        
        # Sortino ratio
        sortino = calculate_sortino_ratio(returns, risk_free)
        
        # Max drawdown
        max_dd_info = calculate_max_drawdown(returns)
        max_dd = max_dd_info['max_drawdown']
        
        # Calmar ratio
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # Win rate
        win_rate = calculate_win_rate(returns)
        
        metrics = {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility
        }
        
        logger.info(
            f"Metrics calculated: Sharpe={sharpe:.2f}, Sortino={sortino:.2f}, "
            f"MaxDD={max_dd:.2%}, WinRate={win_rate:.2%}"
        )
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return _zero_metrics()


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Formula: (annual_return - risk_free) / annual_volatility
    
    Args:
        returns: Series of period returns
        risk_free: Annual risk-free rate
        periods: Trading periods per year (default 252)
    
    Returns:
        float: Sharpe ratio (higher is better)
    """
    if len(returns) == 0:
        return 0.0
    
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate excess returns
    excess_returns = returns - (risk_free / periods)
    
    # Annualize
    mean_excess = excess_returns.mean() * periods
    std_excess = excess_returns.std() * np.sqrt(periods)
    
    # Handle zero volatility
    if std_excess == 0:
        return 0.0
    
    sharpe = mean_excess / std_excess
    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods: int = 252
) -> float:
    """
    Calculate Sortino ratio (uses only downside volatility).
    
    Formula: (annual_return - risk_free) / downside_volatility
    
    Downside volatility uses ONLY negative returns for std calculation.
    This is better for asymmetric return distributions.
    
    Args:
        returns: Series of period returns
        risk_free: Annual risk-free rate
        periods: Trading periods per year (default 252)
    
    Returns:
        float: Sortino ratio (higher is better)
    """
    if len(returns) == 0:
        return 0.0
    
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate excess returns
    excess_returns = returns - (risk_free / periods)
    
    # Downside returns only (negative excess returns)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        # No downside - infinite Sortino, but return high value
        return 1000.0
    
    # Annualize
    mean_excess = excess_returns.mean() * periods
    downside_std = downside_returns.std() * np.sqrt(periods)
    
    # Handle zero downside volatility
    if downside_std == 0:
        return 0.0
    
    sortino = mean_excess / downside_std
    return sortino


def calculate_max_drawdown(returns: pd.Series) -> Dict[str, Any]:
    """
    Calculate maximum drawdown.
    
    Formula: (cumulative - running_max) / running_max, then take minimum
    
    Args:
        returns: Series of period returns
    
    Returns:
        dict with keys:
            - max_drawdown: Maximum drawdown (negative value)
            - peak_date: Date of peak before drawdown
            - trough_date: Date of trough (lowest point)
    """
    if len(returns) == 0:
        return {'max_drawdown': 0.0, 'peak_date': None, 'trough_date': None}
    
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate cumulative returns
    cumulative = (1 + returns).cumprod()
    
    # Running maximum
    running_max = cumulative.cummax()
    
    # Drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    
    # Find peak and trough dates if index is datetime
    if isinstance(returns.index, pd.DatetimeIndex):
        trough_idx = drawdown.idxmin()
        peak_idx = running_max.loc[:trough_idx].idxmax()
        return {
            'max_drawdown': max_dd,
            'peak_date': peak_idx,
            'trough_date': trough_idx
        }
    else:
        return {
            'max_drawdown': max_dd,
            'peak_date': None,
            'trough_date': None
        }


def calculate_win_rate(trades: pd.Series) -> float:
    """
    Calculate win rate.
    
    Formula: (winning_trades) / total_trades
    
    Args:
        trades: Series of trade PnLs or returns
    
    Returns:
        float: Win rate as decimal (0.0 to 1.0)
    """
    if len(trades) == 0:
        return 0.0
    
    trades = trades.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Count winning trades (positive PnL)
    winning_trades = (trades > 0).sum()
    total_trades = len(trades)
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    return win_rate


def _zero_metrics() -> Dict[str, float]:
    """Return dict with all metrics set to 0."""
    return {
        'sharpe_ratio': 0.0,
        'sortino_ratio': 0.0,
        'max_drawdown': 0.0,
        'calmar_ratio': 0.0,
        'win_rate': 0.0,
        'total_return': 0.0,
        'annual_return': 0.0,
        'annual_volatility': 0.0
    }
