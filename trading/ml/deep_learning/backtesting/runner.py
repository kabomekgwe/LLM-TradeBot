"""
Backtest runner with realistic cost modeling and walk-forward validation.

Integrates backtesting.py framework with deep learning models and
comprehensive financial metrics from Plan 08-01.
"""

import pandas as pd
import numpy as np
from backtesting import Backtest
from typing import Any, Dict, List, Optional
import logging

from trading.ml.deep_learning.backtesting.strategy import (
    DeepLearningStrategy,
    precompute_predictions_with_scaler
)
from trading.ml.deep_learning.backtesting.config import BacktestConfig
from trading.ml.evaluation.metrics import comprehensive_metrics
from trading.ml.evaluation.walk_forward import WalkForwardValidator

logger = logging.getLogger(__name__)


class BacktestRunner:
    """
    Run backtests with realistic costs and walk-forward validation.

    Integrates:
    - backtesting.py framework for execution
    - Plan 08-01 walk-forward validation for temporal integrity
    - Plan 08-01 comprehensive metrics for financial assessment
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtest runner.

        Args:
            config: Backtesting configuration (uses defaults if None)
        """
        self.config = config or BacktestConfig()
        logger.info(f"BacktestRunner initialized with config: {self.config.to_dict()}")

    def run_single_backtest(
        self,
        model: Any,
        data: pd.DataFrame,
        predictions: pd.Series,
        ohlcv_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run single backtest with precomputed predictions.

        Args:
            model: Trained model (for reference/logging only)
            data: DataFrame with features (for metrics calculation)
            predictions: Precomputed predictions (pd.Series with DatetimeIndex)
            ohlcv_data: OHLCV DataFrame for backtesting.py
                        Must have columns: Open, High, Low, Close, Volume

        Returns:
            dict with keys:
                - backtest_stats: backtesting.py Stats object
                - financial_metrics: Comprehensive metrics from Plan 08-01
                - trades: DataFrame of all trades
                - returns: Series of period returns
        """
        logger.info(
            f"Running single backtest: {len(ohlcv_data)} bars, "
            f"{len(predictions)} predictions"
        )

        # Verify OHLCV data format
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = set(required_cols) - set(ohlcv_data.columns)
        if missing:
            raise ValueError(f"OHLCV data missing columns: {missing}")

        # Align predictions to OHLCV data
        predictions_aligned = predictions.reindex(ohlcv_data.index, fill_value=0.5)

        # Create strategy class with predictions
        class ConfiguredStrategy(DeepLearningStrategy):
            threshold = self.config.prediction_threshold

            def __init__(self):
                super().__init__()
                self._predictions = predictions_aligned

        # Run backtest
        bt = Backtest(
            data=ohlcv_data,
            strategy=ConfiguredStrategy,
            cash=self.config.initial_cash,
            commission=self.config.commission,  # Note: includes slippage estimate
            margin=1.0,  # No leverage
            trade_on_close=self.config.trade_on_close,
            hedging=False,
            exclusive_orders=True
        )

        # Execute backtest
        stats = bt.run()

        # Extract trades
        trades_df = stats._trades if hasattr(stats, '_trades') else pd.DataFrame()

        # Calculate returns
        if 'Equity' in stats._equity_curve.columns:
            equity = stats._equity_curve['Equity']
            returns = equity.pct_change().dropna()
        else:
            returns = pd.Series()

        # Calculate comprehensive metrics (from Plan 08-01)
        if len(returns) > 0:
            financial_metrics = comprehensive_metrics(
                returns=returns,
                risk_free=0.0  # Default 0% risk-free rate
            )
        else:
            financial_metrics = {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'annual_return': 0.0,
                'annual_volatility': 0.0
            }

        logger.info(
            f"Backtest complete: "
            f"Return={financial_metrics['total_return']:.2%}, "
            f"Sharpe={financial_metrics['sharpe_ratio']:.2f}, "
            f"MaxDD={financial_metrics['max_drawdown']:.2%}"
        )

        return {
            'backtest_stats': stats,
            'financial_metrics': financial_metrics,
            'trades': trades_df,
            'returns': returns
        }

    def run_walk_forward_backtest(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        ohlcv_data: pd.DataFrame,
        feature_columns: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Run walk-forward backtesting.

        Uses WalkForwardValidator from Plan 08-01 to create chronological folds,
        then runs backtest on each test period with predictions from that fold.

        Args:
            model: Model class (NOT fitted) - will be trained in each fold
            X: Features DataFrame
            y: Labels Series
            ohlcv_data: OHLCV DataFrame (must align with X/y)
            feature_columns: List of feature column names

        Returns:
            List of backtest results (one dict per fold)
        """
        logger.info(
            f"Starting walk-forward backtest: {len(X)} samples, "
            f"{len(feature_columns)} features"
        )

        # Initialize walk-forward validator (from Plan 08-01)
        validator = WalkForwardValidator(
            initial_train_size=self.config.initial_train_size,
            test_size=self.config.test_size,
            step_size=self.config.step_size
        )

        # Get fold count
        fold_count = validator.get_split_count(len(X))
        logger.info(f"Walk-forward will create {fold_count} folds")

        # Perform walk-forward validation
        fold_results = validator.validate(model, X, y)

        # Run backtest for each fold
        backtest_results = []

        for idx, fold_row in fold_results.iterrows():
            fold = fold_row['fold']
            test_start = fold_row['test_start']
            test_end = fold_row['test_end']

            logger.info(
                f"Processing fold {fold}/{fold_count}: "
                f"test=[{test_start}:{test_end}]"
            )

            # Get predictions for this fold (already computed by validator)
            predictions = pd.Series(
                fold_row['predictions'],
                index=X.index[test_start:test_end+1]
            )

            # Get corresponding OHLCV data
            ohlcv_fold = ohlcv_data.iloc[test_start:test_end+1]

            # Run backtest for this fold
            try:
                backtest_result = self.run_single_backtest(
                    model=model,
                    data=X.iloc[test_start:test_end+1],
                    predictions=predictions,
                    ohlcv_data=ohlcv_fold
                )

                backtest_result['fold'] = fold
                backtest_result['test_start'] = test_start
                backtest_result['test_end'] = test_end

                backtest_results.append(backtest_result)

            except Exception as e:
                logger.error(f"Fold {fold} backtest failed: {e}")
                continue

        logger.info(
            f"Walk-forward backtest complete: {len(backtest_results)} successful folds"
        )

        return backtest_results
