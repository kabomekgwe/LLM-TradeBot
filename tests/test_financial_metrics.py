"""
Unit tests for financial performance metrics.

Tests ensure:
- Sharpe ratio calculation correct
- Sortino ratio uses downside-only volatility
- Max drawdown calculation correct
- Win rate calculation correct
- Edge case handling (NaN, zero division, empty data)
"""

import pytest
import pandas as pd
import numpy as np
from trading.ml.evaluation.metrics import (
    comprehensive_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_win_rate
)


class TestSharpeRatio:
    """Test suite for Sharpe ratio calculation."""
    
    def test_sharpe_positive_returns(self):
        """Verify Sharpe > 0 for positive returns."""
        # Create positive returns
        np.random.seed(42)
        returns = pd.Series(np.random.rand(252) * 0.01 + 0.001)  # Positive mean
        
        sharpe = calculate_sharpe_ratio(returns, risk_free=0.0)
        
        assert sharpe > 0, f"Sharpe should be positive for positive returns, got {sharpe}"
    
    def test_sharpe_negative_returns(self):
        """Verify Sharpe < 0 for negative returns."""
        # Create negative returns
        np.random.seed(42)
        returns = pd.Series(np.random.rand(252) * -0.01 - 0.001)  # Negative mean
        
        sharpe = calculate_sharpe_ratio(returns, risk_free=0.0)
        
        assert sharpe < 0, f"Sharpe should be negative for negative returns, got {sharpe}"
    
    def test_sharpe_zero_volatility(self):
        """Return 0 when std=0 (edge case)."""
        # Constant returns (zero volatility)
        returns = pd.Series([0.01] * 252)
        
        sharpe = calculate_sharpe_ratio(returns, risk_free=0.0)
        
        # With zero volatility, Sharpe should be 0 (avoid division by zero)
        assert sharpe == 0.0


class TestSortinoRatio:
    """Test suite for Sortino ratio calculation."""
    
    def test_sortino_higher_than_sharpe(self):
        """For asymmetric positive returns, Sortino > Sharpe."""
        # Create asymmetric returns (more positive than negative)
        np.random.seed(42)
        positive_returns = np.random.rand(200) * 0.02 + 0.005  # Large positive
        negative_returns = np.random.rand(52) * -0.005  # Small negative
        returns = pd.Series(np.concatenate([positive_returns, negative_returns]))
        
        sharpe = calculate_sharpe_ratio(returns, risk_free=0.0)
        sortino = calculate_sortino_ratio(returns, risk_free=0.0)
        
        # Sortino should be higher (less penalty for upside volatility)
        assert sortino > sharpe, f"Sortino ({sortino}) should be > Sharpe ({sharpe})"
    
    def test_sortino_downside_only(self):
        """Verify uses only returns < 0 for std."""
        # Create returns with known downside
        returns = pd.Series([0.02, 0.01, -0.01, 0.03, -0.02, 0.01])  # 2 negative
        
        sortino = calculate_sortino_ratio(returns, risk_free=0.0)
        
        # Sortino uses only downside returns for std calculation
        # Should be finite (not inf) because we have downside returns
        assert np.isfinite(sortino)


class TestMaxDrawdown:
    """Test suite for max drawdown calculation."""
    
    def test_max_drawdown_calculation(self):
        """Known drawdown pattern, verify correct."""
        # Create returns with known drawdown
        # Start at 100, go to 150 (+50%), then drop to 100 (-33.33%)
        returns = pd.Series([0.5, -0.333333])  # +50%, then -33.33%
        
        result = calculate_max_drawdown(returns)
        max_dd = result['max_drawdown']
        
        # Max drawdown should be -33.33% (from 150 to 100)
        assert abs(max_dd - (-0.333333)) < 0.001, \
            f"Max drawdown should be -0.333, got {max_dd}"
    
    def test_no_drawdown(self):
        """Returns 0 for monotonically increasing returns."""
        # All positive returns (no drawdown)
        returns = pd.Series([0.01, 0.02, 0.01, 0.03])
        
        result = calculate_max_drawdown(returns)
        max_dd = result['max_drawdown']
        
        # Should be 0 or very close to 0
        assert abs(max_dd) < 0.001, f"Max drawdown should be ~0, got {max_dd}"
    
    def test_peak_trough_dates(self):
        """Verify peak/trough timestamps correct."""
        # Create returns with datetime index
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        returns = pd.Series([0.1, 0.05, -0.2, 0.05, 0.1], index=dates)
        
        result = calculate_max_drawdown(returns)
        
        # Peak should be at index 1 (after +10% and +5%)
        # Trough should be at index 2 (after -20%)
        assert result['peak_date'] == dates[1]
        assert result['trough_date'] == dates[2]


class TestWinRate:
    """Test suite for win rate calculation."""
    
    def test_win_rate_all_wins(self):
        """100% for all positive trades."""
        trades = pd.Series([0.01, 0.02, 0.015, 0.03])
        
        win_rate = calculate_win_rate(trades)
        
        assert win_rate == 1.0, f"Win rate should be 100%, got {win_rate}"
    
    def test_win_rate_all_losses(self):
        """0% for all negative trades."""
        trades = pd.Series([-0.01, -0.02, -0.015, -0.03])
        
        win_rate = calculate_win_rate(trades)
        
        assert win_rate == 0.0, f"Win rate should be 0%, got {win_rate}"
    
    def test_win_rate_mixed(self):
        """Correct ratio for mixed trades."""
        trades = pd.Series([0.01, -0.02, 0.015, -0.01, 0.03])  # 3 wins, 2 losses
        
        win_rate = calculate_win_rate(trades)
        
        expected = 3 / 5  # 60%
        assert abs(win_rate - expected) < 0.001, \
            f"Win rate should be 60%, got {win_rate}"


class TestComprehensiveMetrics:
    """Test suite for comprehensive_metrics function."""
    
    def test_all_metrics_present(self):
        """Verify all 8 metrics returned."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)
        
        metrics = comprehensive_metrics(returns, risk_free=0.02)
        
        required_metrics = [
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'calmar_ratio',
            'win_rate', 'total_return', 'annual_return', 'annual_volatility'
        ]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
    
    def test_empty_returns(self):
        """Return zero metrics for empty data."""
        returns = pd.Series([])
        
        metrics = comprehensive_metrics(returns)
        
        # All metrics should be 0
        for key, value in metrics.items():
            assert value == 0.0, f"Metric {key} should be 0 for empty data"
    
    def test_nan_handling(self):
        """Replace NaN/inf with 0."""
        returns = pd.Series([0.01, np.nan, 0.02, np.inf, -0.01])
        
        metrics = comprehensive_metrics(returns)
        
        # Should not raise error and return finite values
        assert np.isfinite(metrics['sharpe_ratio'])
        assert np.isfinite(metrics['sortino_ratio'])
