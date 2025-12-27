"""Volatility regime detection using Hidden Markov Models.

Classifies market into low/high volatility regimes using HMM
with switching variance on returns.
"""

import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

logger = logging.getLogger(__name__)


class VolatilityRegimeDetector:
    """Detect volatility regimes using Hidden Markov Models."""

    def __init__(self, n_regimes: int = 2, min_periods: int = 100):
        """Initialize regime detector.

        Args:
            n_regimes: Number of volatility regimes (default: 2 for low/high)
            min_periods: Minimum data points required for HMM (default: 100)
        """
        self.n_regimes = n_regimes
        self.min_periods = min_periods
        self.model = None
        self.results = None

    def fit(self, returns: pd.Series) -> None:
        """Fit HMM to returns data.

        Args:
            returns: Series of returns (close.pct_change())

        Raises:
            ValueError: If insufficient data
        """
        # Remove NaN and check length
        returns_clean = returns.dropna()

        if len(returns_clean) < self.min_periods:
            raise ValueError(f"Insufficient data: need {self.min_periods}, got {len(returns_clean)}")

        try:
            # Fit Markov Regime Switching model
            self.model = MarkovRegression(
                returns_clean,
                k_regimes=self.n_regimes,
                switching_variance=True,  # Key: variance switches between regimes
            )

            # Fit with error handling (HMM can fail to converge)
            self.results = self.model.fit(maxiter=100, disp=False)

            logger.info(f"HMM fitted with {self.n_regimes} regimes on {len(returns_clean)} returns")

        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            raise

    def detect_regimes(self, df: pd.DataFrame, return_col: str = 'close') -> pd.DataFrame:
        """Detect volatility regimes in DataFrame.

        Args:
            df: DataFrame with OHLCV data
            return_col: Column to calculate returns from (default: 'close')

        Returns:
            DataFrame with regime probability and current regime columns

        Raises:
            ValueError: If return_col not in DataFrame or insufficient data
        """
        if return_col not in df.columns:
            raise ValueError(f"Column '{return_col}' not found in DataFrame")

        # Calculate returns
        returns = df[return_col].pct_change().dropna()

        # Fit HMM
        self.fit(returns)

        # Get smoothed regime probabilities
        # Index 0 = low volatility, Index 1 = high volatility (typically)
        regime_probs = self.results.smoothed_marginal_probabilities

        # Add regime features to dataframe (align with original df length)
        # Note: First row will be NaN due to pct_change()
        df['regime_prob_0'] = np.nan
        df['regime_prob_1'] = np.nan
        df['current_regime'] = np.nan

        # Fill from index 1 onwards (after first NaN from pct_change)
        df.loc[df.index[1:], 'regime_prob_0'] = regime_probs[0].values
        df.loc[df.index[1:], 'regime_prob_1'] = regime_probs[1].values
        df.loc[df.index[1:], 'current_regime'] = regime_probs.idxmax(axis=1).values

        # Identify which regime is low vs high volatility
        # Try different parameter name formats (statsmodels API varies)
        try:
            regime_0_var = self.results.params['sigma2[0]']
            regime_1_var = self.results.params['sigma2[1]']
        except KeyError:
            try:
                regime_0_var = self.results.params['sigma2.0']
                regime_1_var = self.results.params['sigma2.1']
            except KeyError:
                # Fallback: use variance from filtered probabilities
                regime_0_var = returns.loc[df.index[1:]][df.loc[df.index[1:], 'current_regime'] == 0].var()
                regime_1_var = returns.loc[df.index[1:]][df.loc[df.index[1:], 'current_regime'] == 1].var()

        low_vol_regime = 0 if regime_0_var < regime_1_var else 1
        df['is_low_volatility'] = (df['current_regime'] == low_vol_regime).astype(int)

        logger.info(f"Detected regimes: Regime 0 var={regime_0_var:.6f}, Regime 1 var={regime_1_var:.6f}")
        logger.info(f"Low volatility regime: {low_vol_regime}")

        return df
