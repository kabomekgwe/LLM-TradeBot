"""
Model evaluation module for LLM-TradeBot.

Provides walk-forward validation and financial performance metrics
for evaluating machine learning models under realistic conditions.
"""

from trading.ml.evaluation.walk_forward import WalkForwardValidator
from trading.ml.evaluation.metrics import (
    comprehensive_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_win_rate
)
from trading.ml.evaluation.report_generator import PerformanceReportGenerator

__all__ = [
    'WalkForwardValidator',
    'comprehensive_metrics',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'calculate_win_rate',
    'PerformanceReportGenerator'
]
