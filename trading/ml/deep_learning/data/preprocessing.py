"""Hybrid Data Preprocessing Pipeline for LSTM/Transformer Training.

CRITICAL DESIGN DECISIONS:
- Z-score normalization (StandardScaler) NOT min-max (financial data has outliers)
- Chronological splits (NO shuffle) - maintains temporal dependencies
- Fit scaler ONLY on training set, then transform validation/test - prevents leakage
- Vectorized sliding window (Don't hand-roll from research)

This preprocessing pipeline prevents common pitfalls:
- Pitfall 1: Data Leakage via Shuffling - NO shuffle anywhere
- Pitfall from research line 241: StandardScaler better than min-max for financial data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List


class DataPreprocessor:
    """Preprocess time series data for LSTM/Transformer training.

    Example:
        >>> preprocessor = DataPreprocessor(sequence_length=50)
        >>> sequences, labels = preprocessor.create_sequences(
        ...     df,
        ...     feature_columns=['feature_1', 'feature_2'],
        ...     label_column='target_binary',
        ...     fit_scaler=True
        ... )
        >>> sequences.shape
        (950, 50, 2)  # 950 sequences, 50 timesteps, 2 features
    """

    def __init__(self, sequence_length: int = 50):
        """Initialize data preprocessor.

        Args:
            sequence_length: Number of timesteps in each sequence (50-100 typical for LSTM)
        """
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()  # Z-score normalization
        self.is_fitted = False

    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str,
        fit_scaler: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences from dataframe.

        CRITICAL: If fit_scaler=True, this should ONLY be called on training data.
        For validation/test data, call with fit_scaler=False to use the fitted scaler.

        Args:
            df: DataFrame with features and labels
            feature_columns: List of column names to use as features (86 from Phase 5)
            label_column: Name of binary label column
            fit_scaler: If True, fit scaler on this data (ONLY for training set)

        Returns:
            Tuple of (sequences, labels)
            - sequences: shape (N, sequence_length, num_features)
            - labels: shape (N,)

        Raises:
            ValueError: If scaler not fitted and fit_scaler=False
        """
        # Extract features
        features = df[feature_columns].values  # (total_timesteps, num_features)

        # Normalize features
        if fit_scaler:
            features_scaled = self.scaler.fit_transform(features)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError(
                    "Scaler not fitted. Call with fit_scaler=True on training data first."
                )
            features_scaled = self.scaler.transform(features)

        # Create sequences with sliding window (vectorized, not hand-rolled)
        sequences = []
        labels = []

        for i in range(len(features_scaled) - self.sequence_length):
            # Sequence of length sequence_length
            sequences.append(features_scaled[i:i+self.sequence_length])
            # Label is the next timestep after sequence
            labels.append(df[label_column].iloc[i+self.sequence_length])

        return np.array(sequences), np.array(labels)

    def split_time_series(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_size: float = 0.15,
        test_size: float = 0.15
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        """Time-series aware train/validation/test split.

        CRITICAL: NO SHUFFLE - chronological split only (Pitfall 1: Data Leakage)

        Example:
            >>> X = np.random.randn(1000, 50, 86)
            >>> y = np.random.randint(0, 2, size=1000)
            >>> preprocessor = DataPreprocessor()
            >>> (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_time_series(X, y)
            >>> X_train.shape, X_val.shape, X_test.shape
            ((700, 50, 86), (150, 50, 86), (150, 50, 86))

        Args:
            X: sequences shape (N, seq_len, features)
            y: labels shape (N,)
            validation_size: Validation set proportion (default 15%)
            test_size: Test set proportion (default 15%)

        Returns:
            Tuple of ((X_train, y_train), (X_validation, y_validation), (X_test, y_test))
        """
        total_len = len(X)
        test_idx = int(total_len * (1 - test_size))
        validation_idx = int(total_len * (1 - test_size - validation_size))

        # Chronological split (NO SHUFFLE)
        X_train, y_train = X[:validation_idx], y[:validation_idx]
        X_validation, y_validation = X[validation_idx:test_idx], y[validation_idx:test_idx]
        X_test, y_test = X[test_idx:], y[test_idx:]

        return (X_train, y_train), (X_validation, y_validation), (X_test, y_test)

    def get_scaler_params(self) -> dict:
        """Get fitted scaler parameters.

        Returns:
            Dictionary with scaler mean and std
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted yet")

        return {
            'mean': self.scaler.mean_.tolist(),
            'std': self.scaler.scale_.tolist(),
            'n_features': len(self.scaler.mean_),
        }
