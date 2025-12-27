"""
Backtesting configuration with realistic trading costs.

Based on 08-RESEARCH.md findings:
- Commission: 0.001 (0.1% per trade, realistic for crypto)
- Slippage: 10-20 basis points (0.001-0.002)
- Trade-on-close: False for realistic execution
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting with realistic costs.

    Attributes:
        initial_cash: Starting capital (default $10,000)
        commission: Commission per trade as decimal (default 0.001 = 0.1%)
        slippage_bps: Slippage in basis points (default 15 bps = 0.15%)
        trade_on_close: Whether to trade on bar close (False = realistic)
        max_position_size: Maximum position size as fraction of capital (default 0.95)
        sequence_length: Sequence length for model predictions (default 60)
        prediction_threshold: Confidence threshold for trades (default 0.5)
        initial_train_size: Initial training window for walk-forward (default 252)
        test_size: Test window size for walk-forward (default 60)
        step_size: Step size for walk-forward validation (default 30)
    """

    # Capital settings
    initial_cash: float = 10000.0
    max_position_size: float = 0.95  # 95% of capital

    # Cost modeling (from 08-RESEARCH.md)
    commission: float = 0.001  # 0.1% per trade (realistic for crypto)
    slippage_bps: int = 15  # 15 basis points slippage
    trade_on_close: bool = False  # Realistic execution (no perfect fills)

    # Model parameters
    sequence_length: int = 60  # Timesteps per sequence
    prediction_threshold: float = 0.5  # Confidence threshold for trades

    # Walk-forward validation parameters (from Plan 08-01)
    initial_train_size: int = 252  # 1 year initial training
    test_size: int = 60  # 3 months test window
    step_size: int = 30  # 1 month advance per fold

    # Risk management
    stop_loss: Optional[float] = None  # Optional stop loss (e.g., 0.05 = 5%)
    take_profit: Optional[float] = None  # Optional take profit (e.g., 0.10 = 10%)

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.commission < 0 or self.commission > 0.1:
            raise ValueError(f"Commission must be between 0 and 0.1, got {self.commission}")

        if self.slippage_bps < 0 or self.slippage_bps > 100:
            raise ValueError(f"Slippage must be between 0 and 100 bps, got {self.slippage_bps}")

        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise ValueError(
                f"Max position size must be between 0 and 1, got {self.max_position_size}"
            )

        if self.prediction_threshold < 0 or self.prediction_threshold > 1:
            raise ValueError(
                f"Prediction threshold must be between 0 and 1, got {self.prediction_threshold}"
            )

    def get_effective_commission(self) -> float:
        """
        Get effective commission including slippage estimate.

        Note: backtesting.py doesn't have built-in slippage modeling,
        so we include slippage in the commission estimate.

        Returns:
            float: Effective commission including slippage
        """
        slippage_decimal = self.slippage_bps / 10000  # Convert bps to decimal
        return self.commission + slippage_decimal

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging/serialization."""
        return {
            'initial_cash': self.initial_cash,
            'commission': self.commission,
            'slippage_bps': self.slippage_bps,
            'trade_on_close': self.trade_on_close,
            'max_position_size': self.max_position_size,
            'sequence_length': self.sequence_length,
            'prediction_threshold': self.prediction_threshold,
            'initial_train_size': self.initial_train_size,
            'test_size': self.test_size,
            'step_size': self.step_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }
