"""
Backtesting module for deep learning trading models.

Integrates BiLSTM and Transformer models with backtesting.py framework
for realistic trading simulation and performance evaluation.
"""

from trading.ml.deep_learning.backtesting.strategy import (
    DeepLearningStrategy,
    precompute_predictions
)
from trading.ml.deep_learning.backtesting.config import BacktestConfig
from trading.ml.deep_learning.backtesting.runner import BacktestRunner

__all__ = [
    'DeepLearningStrategy',
    'precompute_predictions',
    'BacktestConfig',
    'BacktestRunner'
]
