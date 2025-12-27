"""
Backtesting.py Strategy wrapper for deep learning models.

CRITICAL: Implements precomputed predictions pattern to prevent look-ahead bias.
All model predictions are computed BEFORE backtesting, ensuring temporal integrity.
"""

import pandas as pd
import numpy as np
from backtesting import Strategy
from typing import Optional, Any
import logging
import torch

logger = logging.getLogger(__name__)


class DeepLearningStrategy(Strategy):
    """
    Backtesting.py strategy wrapper for deep learning models.

    CRITICAL DESIGN:
    - Predictions are precomputed BEFORE backtesting (prevents look-ahead bias)
    - No model inference during backtesting (uses precomputed signals)
    - Threshold-based trading logic (long when prediction >= threshold)

    Parameters:
        predictions (pd.Series): Precomputed predictions aligned to data index
        threshold (float): Prediction confidence threshold for trades (default 0.5)
    """

    # Strategy parameters (can be optimized)
    threshold = 0.5

    def init(self):
        """
        Initialize strategy (called once at backtest start).

        Verifies predictions are provided and aligned with backtest data.
        """
        # Get precomputed predictions from strategy kwargs
        if not hasattr(self, '_predictions'):
            raise ValueError(
                "Predictions not provided. Use bt.run(predictions=...) "
                "or pass via strategy kwargs"
            )

        # Verify predictions align with data
        if len(self._predictions) != len(self.data.df):
            logger.warning(
                f"Predictions length ({len(self._predictions)}) != "
                f"data length ({len(self.data.df)}). Aligning..."
            )
            # Align predictions to data index
            self._predictions = self._predictions.reindex(
                self.data.df.index,
                fill_value=0.5
            )

        logger.info(
            f"DeepLearningStrategy initialized: "
            f"{len(self._predictions)} predictions, threshold={self.threshold}"
        )

    def next(self):
        """
        Execute trading logic on each bar.

        Uses precomputed predictions (NO model inference here).
        Trading logic:
        - Long signal: prediction >= threshold and no position
        - Exit: prediction < threshold and position exists
        """
        # Get current prediction (precomputed, not inferred!)
        current_idx = len(self.data) - 1

        # Handle index alignment
        if current_idx >= len(self._predictions):
            return

        prediction = self._predictions.iloc[current_idx]

        # Trading logic
        if prediction >= self.threshold and not self.position:
            # Buy signal
            self.buy()
        elif prediction < self.threshold and self.position:
            # Exit signal
            self.position.close()


def precompute_predictions(
    model: Any,
    data: pd.DataFrame,
    feature_columns: list,
    sequence_length: int = 60,
    device: str = 'cpu'
) -> pd.Series:
    """
    Precompute model predictions for backtesting.

    CRITICAL: This function runs model inference BEFORE backtesting to prevent
    look-ahead bias. The returned predictions are then passed to backtesting.py
    as a precomputed signal.

    Args:
        model: Trained PyTorch model (BiLSTM or Transformer)
        data: DataFrame with features (OHLCV + engineered features)
        feature_columns: List of feature column names (86 from Phase 5)
        sequence_length: Number of timesteps per sequence (default 60)
        device: Device for inference ('cpu' or 'cuda')

    Returns:
        pd.Series: Predictions aligned to data index (length = len(data) - sequence_length)
    """
    from trading.ml.deep_learning.data.preprocessing import DataPreprocessor

    logger.info(
        f"Precomputing predictions: {len(data)} samples, "
        f"{len(feature_columns)} features, seq_len={sequence_length}"
    )

    # Validate inputs
    if len(data) < sequence_length:
        raise ValueError(
            f"Insufficient data: need {sequence_length}, got {len(data)}"
        )

    missing_cols = set(feature_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    # Prepare data using DataPreprocessor
    preprocessor = DataPreprocessor(sequence_length=sequence_length)

    # Create sequences (fit scaler on this data)
    # Note: In production, you should use scaler fitted on training data only
    sequences, _ = preprocessor.create_sequences(
        df=data,
        feature_columns=feature_columns,
        label_column=feature_columns[0],  # Dummy label, not used
        fit_scaler=True  # Warning: This fits on all data - use trained scaler in production
    )

    # Convert to tensor
    X_tensor = torch.FloatTensor(sequences).to(device)

    # Run inference
    model.eval()
    model = model.to(device)

    predictions = []
    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            outputs = model(batch)

            # Handle different output formats
            if isinstance(outputs, torch.Tensor):
                # Binary classification - get probability of class 1
                if outputs.shape[-1] == 1:
                    # Single output (sigmoid)
                    probs = torch.sigmoid(outputs).squeeze()
                else:
                    # Two outputs (softmax)
                    probs = torch.softmax(outputs, dim=-1)[:, 1]
            else:
                raise ValueError(f"Unexpected model output type: {type(outputs)}")

            predictions.extend(probs.cpu().numpy())

    # Create Series aligned to data index
    # Predictions start at index sequence_length (first complete sequence)
    pred_index = data.index[sequence_length:]
    predictions_series = pd.Series(predictions, index=pred_index)

    logger.info(
        f"Precomputed {len(predictions_series)} predictions "
        f"(mean={predictions_series.mean():.3f}, std={predictions_series.std():.3f})"
    )

    return predictions_series


def precompute_predictions_with_scaler(
    model: Any,
    data: pd.DataFrame,
    feature_columns: list,
    scaler: Any,
    sequence_length: int = 60,
    device: str = 'cpu'
) -> pd.Series:
    """
    Precompute predictions using pre-fitted scaler (prevents data leakage).

    This version accepts a scaler fitted ONLY on training data, ensuring
    no leakage from test/validation data.

    Args:
        model: Trained PyTorch model
        data: DataFrame with features
        feature_columns: List of feature column names
        scaler: Pre-fitted StandardScaler (from training data)
        sequence_length: Number of timesteps per sequence
        device: Device for inference

    Returns:
        pd.Series: Predictions aligned to data index
    """
    logger.info(
        f"Precomputing predictions with pre-fitted scaler: "
        f"{len(data)} samples, {len(feature_columns)} features"
    )

    # Extract and scale features
    features = data[feature_columns].values
    features_scaled = scaler.transform(features)

    # Create sequences manually (without fitting scaler)
    sequences = []
    for i in range(len(features_scaled) - sequence_length):
        sequences.append(features_scaled[i:i+sequence_length])

    sequences = np.array(sequences)

    # Convert to tensor
    X_tensor = torch.FloatTensor(sequences).to(device)

    # Run inference
    model.eval()
    model = model.to(device)

    predictions = []
    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            outputs = model(batch)

            # Handle different output formats
            if isinstance(outputs, torch.Tensor):
                if outputs.shape[-1] == 1:
                    probs = torch.sigmoid(outputs).squeeze()
                else:
                    probs = torch.softmax(outputs, dim=-1)[:, 1]
            else:
                raise ValueError(f"Unexpected model output type: {type(outputs)}")

            predictions.extend(probs.cpu().numpy())

    # Create Series aligned to data index
    pred_index = data.index[sequence_length:]
    predictions_series = pd.Series(predictions, index=pred_index)

    logger.info(
        f"Precomputed {len(predictions_series)} predictions "
        f"(mean={predictions_series.mean():.3f}, std={predictions_series.std():.3f})"
    )

    return predictions_series
