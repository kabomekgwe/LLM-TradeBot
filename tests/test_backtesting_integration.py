"""
Integration tests for backtesting infrastructure.

Tests the complete backtesting pipeline:
- Realistic cost modeling
- Precomputed predictions (no look-ahead bias)
- Performance report generation
- Model comparison
"""

import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from trading.ml.deep_learning.backtesting import (
    BacktestConfig,
    BacktestRunner,
    DeepLearningStrategy,
    precompute_predictions
)
from trading.ml.evaluation import PerformanceReportGenerator


class TestBacktestingIntegration:
    """Integration tests for backtesting components."""

    @pytest.fixture
    def dummy_ohlcv_data(self):
        """Create dummy OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
        
        close_prices = 50000 + np.cumsum(np.random.randn(1000) * 100)
        
        data = pd.DataFrame({
            'Open': close_prices + np.random.randn(1000) * 50,
            'High': close_prices + np.abs(np.random.randn(1000) * 100),
            'Low': close_prices - np.abs(np.random.randn(1000) * 100),
            'Close': close_prices,
            'Volume': np.random.rand(1000) * 1000000
        }, index=dates)
        
        return data

    @pytest.fixture
    def dummy_predictions(self, dummy_ohlcv_data):
        """Create dummy predictions aligned to OHLCV data."""
        # Predictions start after sequence_length (60)
        predictions = pd.Series(
            np.random.rand(len(dummy_ohlcv_data) - 60),
            index=dummy_ohlcv_data.index[60:]
        )
        return predictions

    def test_backtest_config_realistic_costs(self):
        """Test that BacktestConfig has realistic cost parameters."""
        config = BacktestConfig()
        
        # Verify commission is 0.1% (0.001)
        assert config.commission == 0.001, "Commission should be 0.001 (0.1%)"
        
        # Verify slippage is 15 bps
        assert config.slippage_bps == 15, "Slippage should be 15 basis points"
        
        # Verify trade-on-close is False (realistic execution)
        assert config.trade_on_close is False, "trade_on_close should be False for realism"
        
        # Verify effective commission includes slippage
        effective_commission = config.get_effective_commission()
        expected = 0.001 + (15 / 10000)  # commission + slippage in decimal
        assert abs(effective_commission - expected) < 1e-6, \
            f"Effective commission should be {expected}, got {effective_commission}"

    def test_precompute_predictions_no_look_ahead(self, dummy_ohlcv_data):
        """Test that predictions are precomputed Series (no look-ahead bias)."""
        # Create dummy model (simple linear layer for testing)
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 2)
            
            def forward(self, x):
                # x shape: (batch, seq_len, features)
                return self.linear(x[:, -1, :])  # Use last timestep
        
        model = DummyModel()
        
        # Create dummy features
        feature_cols = [f'feature_{i}' for i in range(10)]
        features_df = dummy_ohlcv_data.copy()
        for col in feature_cols:
            features_df[col] = np.random.randn(len(dummy_ohlcv_data))
        
        # Precompute predictions
        predictions = precompute_predictions(
            model=model,
            data=features_df,
            feature_columns=feature_cols,
            sequence_length=60,
            device='cpu'
        )
        
        # Verify predictions are a Series
        assert isinstance(predictions, pd.Series), "Predictions should be pd.Series"
        
        # Verify predictions don't include first sequence_length samples
        assert len(predictions) == len(dummy_ohlcv_data) - 60, \
            "Predictions should exclude first sequence_length samples"
        
        # Verify index alignment
        assert predictions.index[0] == dummy_ohlcv_data.index[60], \
            "First prediction should align to index 60"

    def test_backtest_runner_single_backtest(self, dummy_ohlcv_data, dummy_predictions):
        """Test BacktestRunner.run_single_backtest()."""
        config = BacktestConfig(initial_cash=10000, commission=0.001)
        runner = BacktestRunner(config)
        
        # Create dummy features DataFrame
        features_df = dummy_ohlcv_data.copy()
        
        # Run single backtest
        result = runner.run_single_backtest(
            model=None,  # Not used in single backtest
            data=features_df,
            predictions=dummy_predictions,
            ohlcv_data=dummy_ohlcv_data
        )
        
        # Verify result structure
        assert 'backtest_stats' in result, "Result should contain backtest_stats"
        assert 'financial_metrics' in result, "Result should contain financial_metrics"
        assert 'trades' in result, "Result should contain trades"
        assert 'returns' in result, "Result should contain returns"
        
        # Verify financial metrics are present
        metrics = result['financial_metrics']
        required_metrics = [
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'calmar_ratio', 'win_rate', 'total_return',
            'annual_return', 'annual_volatility'
        ]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

    def test_report_generator_creates_files(self, tmp_path):
        """Test that PerformanceReportGenerator creates output files."""
        generator = PerformanceReportGenerator(output_dir=tmp_path)
        
        # Create dummy backtest results
        backtest_results = []
        for i in range(3):
            result = {
                'fold': i + 1,
                'test_start': i * 100,
                'test_end': (i + 1) * 100,
                'financial_metrics': {
                    'sharpe_ratio': 1.5 + np.random.randn() * 0.3,
                    'sortino_ratio': 2.0 + np.random.randn() * 0.3,
                    'max_drawdown': -0.15 + np.random.randn() * 0.05,
                    'calmar_ratio': 1.2 + np.random.randn() * 0.2,
                    'win_rate': 0.55 + np.random.randn() * 0.05,
                    'total_return': 0.25 + np.random.randn() * 0.1,
                    'annual_return': 0.30 + np.random.randn() * 0.1,
                    'annual_volatility': 0.20 + np.random.randn() * 0.05
                },
                'returns': pd.Series(np.random.randn(100) * 0.01)
            }
            backtest_results.append(result)
        
        # Generate report
        summary = generator.generate_single_model_report(
            model_name='TestModel',
            backtest_results=backtest_results
        )
        
        # Verify summary structure
        assert 'mean_metrics' in summary, "Summary should contain mean_metrics"
        assert 'std_metrics' in summary, "Summary should contain std_metrics"
        assert 'best_fold' in summary, "Summary should contain best_fold"
        assert 'worst_fold' in summary, "Summary should contain worst_fold"
        
        # Verify files were created
        model_dir = tmp_path / 'testmodel'
        assert model_dir.exists(), "Model directory should be created"
        assert (model_dir / 'equity_curves.png').exists(), "Equity curves plot should exist"
        assert (model_dir / 'drawdown_distribution.png').exists(), "Drawdown plot should exist"
        assert (model_dir / 'metrics_evolution.png').exists(), "Metrics evolution plot should exist"
        assert (model_dir / 'report.txt').exists(), "Text report should exist"

    def test_comparison_report_multiple_models(self, tmp_path):
        """Test comparison report generation for multiple models."""
        generator = PerformanceReportGenerator(output_dir=tmp_path)
        
        # Create dummy results for two models
        model_results = {}
        for model_name in ['BiLSTM', 'Transformer']:
            results = []
            for i in range(2):
                result = {
                    'fold': i + 1,
                    'financial_metrics': {
                        'sharpe_ratio': 1.5 + np.random.randn() * 0.3,
                        'sortino_ratio': 2.0 + np.random.randn() * 0.3,
                        'max_drawdown': -0.15 + np.random.randn() * 0.05,
                        'calmar_ratio': 1.2 + np.random.randn() * 0.2,
                        'win_rate': 0.55 + np.random.randn() * 0.05,
                        'total_return': 0.25 + np.random.randn() * 0.1,
                        'annual_return': 0.30 + np.random.randn() * 0.1,
                        'annual_volatility': 0.20 + np.random.randn() * 0.05
                    },
                    'returns': pd.Series(np.random.randn(100) * 0.01)
                }
                results.append(result)
            model_results[model_name] = results
        
        # Generate comparison report
        comparison_df = generator.generate_comparison_report(model_results)
        
        # Verify comparison DataFrame
        assert len(comparison_df) == 2, "Should have 2 models in comparison"
        assert 'Model' in comparison_df.columns, "Should have Model column"
        assert 'Sharpe Ratio' in comparison_df.columns, "Should have Sharpe Ratio column"
        
        # Verify comparison files created
        comparison_dir = tmp_path / 'comparison'
        assert comparison_dir.exists(), "Comparison directory should exist"
        assert (comparison_dir / 'comparison_table.csv').exists(), "Comparison table CSV should exist"
        assert (comparison_dir / 'model_comparison.png').exists(), "Comparison plot should exist"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
