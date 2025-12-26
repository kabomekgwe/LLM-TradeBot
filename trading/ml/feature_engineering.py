"""Feature Engineering - Generate ML features from OHLCV data.

Implements 50+ features across multiple categories:
- Returns (simple, log, forward)
- Technical indicators (MA, RSI, MACD, Bollinger Bands)
- Volatility metrics (ATR, standard deviation, Keltner channels)
- Momentum indicators (ROC, momentum, Williams %R)
- Volume metrics (OBV, volume ratios, VWAP)
- Price patterns (highs/lows, price position)
- Statistical features (skewness, kurtosis, autocorrelation)
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FeatureMetadata:
    """Metadata about engineered features."""
    feature_names: List[str]
    feature_categories: Dict[str, List[str]]
    required_history: int  # Minimum candles needed
    target_column: Optional[str] = None


class FeatureEngineer:
    """Generate ML-ready features from OHLCV data.

    Transforms raw price data into 50+ engineered features
    suitable for machine learning models.

    Example:
        >>> engineer = FeatureEngineer()
        >>> features = engineer.transform(ohlcv_df)
        >>> features.shape
        (800, 55)  # 800 samples, 55 features
    """

    def __init__(
        self,
        windows: Optional[List[int]] = None,
        include_target: bool = True,
        target_horizon: int = 1,
    ):
        """Initialize feature engineer.

        Args:
            windows: Lookback windows for rolling features (default: [5, 10, 20, 50])
            include_target: Whether to include target column (future returns)
            target_horizon: Forward periods for target (default: 1)
        """
        self.logger = logging.getLogger(__name__)
        self.windows = windows or [5, 10, 20, 50]
        self.include_target = include_target
        self.target_horizon = target_horizon

        # Feature categories
        self.feature_categories = {
            'returns': [],
            'technical': [],
            'volatility': [],
            'momentum': [],
            'volume': [],
            'price_patterns': [],
            'statistical': [],
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform OHLCV data into ML features.

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with engineered features
        """
        if len(df) < max(self.windows) + 10:
            raise ValueError(f"Insufficient data: need at least {max(self.windows) + 10} candles")

        # Copy to avoid modifying original
        data = df.copy()

        # Ensure lowercase column names
        data.columns = data.columns.str.lower()

        # Add all features
        data = self._add_return_features(data)
        data = self._add_technical_indicators(data)
        data = self._add_volatility_features(data)
        data = self._add_momentum_features(data)
        data = self._add_volume_features(data)
        data = self._add_price_patterns(data)
        data = self._add_statistical_features(data)

        # Add target if requested
        if self.include_target:
            data = self._add_target(data)

        # Drop NaN rows (from rolling calculations)
        data = data.dropna()

        self.logger.info(f"Generated {len(data.columns)} features from {len(data)} samples")

        return data

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        # Simple returns
        for window in self.windows:
            col_name = f'return_{window}'
            df[col_name] = df['close'].pct_change(window)
            self.feature_categories['returns'].append(col_name)

        # Log returns
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        self.feature_categories['returns'].append('log_return_1')

        # Intraday return
        df['intraday_return'] = (df['close'] - df['open']) / df['open']
        self.feature_categories['returns'].append('intraday_return')

        # High-low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        self.feature_categories['returns'].append('hl_range')

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features."""
        # Moving averages
        for window in self.windows:
            col_name = f'sma_{window}'
            df[col_name] = df['close'].rolling(window).mean()
            self.feature_categories['technical'].append(col_name)

            # Price relative to MA
            col_name_ratio = f'price_to_sma_{window}'
            df[col_name_ratio] = df['close'] / df[col_name]
            self.feature_categories['technical'].append(col_name_ratio)

        # Exponential moving averages
        for window in [12, 26]:
            col_name = f'ema_{window}'
            df[col_name] = df['close'].ewm(span=window, adjust=False).mean()
            self.feature_categories['technical'].append(col_name)

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        self.feature_categories['technical'].extend(['macd', 'macd_signal', 'macd_histogram'])

        # RSI
        for window in [14]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            col_name = f'rsi_{window}'
            df[col_name] = 100 - (100 / (1 + rs))
            self.feature_categories['technical'].append(col_name)

        # Bollinger Bands
        for window in [20]:
            sma = df['close'].rolling(window).mean()
            std = df['close'].rolling(window).std()
            df[f'bb_upper_{window}'] = sma + (2 * std)
            df[f'bb_lower_{window}'] = sma - (2 * std)
            df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / sma
            df[f'bb_position_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (
                df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']
            )
            self.feature_categories['technical'].extend([
                f'bb_upper_{window}', f'bb_lower_{window}',
                f'bb_width_{window}', f'bb_position_{window}'
            ])

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        # ATR (Average True Range)
        for window in [14]:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            col_name = f'atr_{window}'
            df[col_name] = true_range.rolling(window).mean()
            df[f'atr_ratio_{window}'] = df[col_name] / df['close']
            self.feature_categories['volatility'].extend([col_name, f'atr_ratio_{window}'])

        # Rolling standard deviation
        for window in self.windows:
            col_name = f'std_{window}'
            df[col_name] = df['close'].rolling(window).std()
            df[f'std_ratio_{window}'] = df[col_name] / df['close']
            self.feature_categories['volatility'].extend([col_name, f'std_ratio_{window}'])

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""
        # Rate of Change (ROC)
        for window in self.windows:
            col_name = f'roc_{window}'
            df[col_name] = df['close'].pct_change(window) * 100
            self.feature_categories['momentum'].append(col_name)

        # Momentum
        for window in [10, 20]:
            col_name = f'momentum_{window}'
            df[col_name] = df['close'] - df['close'].shift(window)
            self.feature_categories['momentum'].append(col_name)

        # Williams %R
        for window in [14]:
            high_max = df['high'].rolling(window).max()
            low_min = df['low'].rolling(window).min()
            col_name = f'williams_r_{window}'
            df[col_name] = -100 * ((high_max - df['close']) / (high_max - low_min))
            self.feature_categories['momentum'].append(col_name)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume moving averages
        for window in self.windows:
            col_name = f'volume_sma_{window}'
            df[col_name] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[col_name]
            self.feature_categories['volume'].extend([col_name, f'volume_ratio_{window}'])

        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        self.feature_categories['volume'].append('obv')

        # Volume-weighted average price (VWAP) approximation
        for window in [20]:
            col_name = f'vwap_{window}'
            df[col_name] = (df['close'] * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
            df[f'price_to_vwap_{window}'] = df['close'] / df[col_name]
            self.feature_categories['volume'].extend([col_name, f'price_to_vwap_{window}'])

        return df

    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features."""
        # Distance to recent highs/lows
        for window in [20, 50]:
            df[f'dist_to_high_{window}'] = (df['high'].rolling(window).max() - df['close']) / df['close']
            df[f'dist_to_low_{window}'] = (df['close'] - df['low'].rolling(window).min()) / df['close']
            self.feature_categories['price_patterns'].extend([
                f'dist_to_high_{window}', f'dist_to_low_{window}'
            ])

        # Price position in range
        for window in [20]:
            high_max = df['high'].rolling(window).max()
            low_min = df['low'].rolling(window).min()
            col_name = f'price_position_{window}'
            df[col_name] = (df['close'] - low_min) / (high_max - low_min)
            self.feature_categories['price_patterns'].append(col_name)

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        # Skewness and kurtosis of returns
        for window in [20]:
            returns = df['close'].pct_change()
            df[f'skew_{window}'] = returns.rolling(window).skew()
            df[f'kurt_{window}'] = returns.rolling(window).kurt()
            self.feature_categories['statistical'].extend([f'skew_{window}', f'kurt_{window}'])

        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variable (future returns)."""
        df['target'] = df['close'].shift(-self.target_horizon).pct_change(self.target_horizon)

        # Binary target: 1 if price increases, 0 otherwise
        df['target_binary'] = (df['target'] > 0).astype(int)

        return df

    def get_feature_names(self, exclude_target: bool = True) -> List[str]:
        """Get list of all feature names.

        Args:
            exclude_target: Whether to exclude target columns

        Returns:
            List of feature column names
        """
        all_features = []
        for category_features in self.feature_categories.values():
            all_features.extend(category_features)

        if exclude_target:
            return all_features
        else:
            return all_features + ['target', 'target_binary']

    def get_metadata(self) -> FeatureMetadata:
        """Get metadata about features.

        Returns:
            FeatureMetadata object
        """
        return FeatureMetadata(
            feature_names=self.get_feature_names(exclude_target=True),
            feature_categories=self.feature_categories,
            required_history=max(self.windows) + 10,
            target_column='target_binary' if self.include_target else None,
        )

    def get_feature_importance_categories(self) -> Dict[str, List[str]]:
        """Get features grouped by category.

        Returns:
            Dictionary mapping category names to feature lists
        """
        return self.feature_categories.copy()
