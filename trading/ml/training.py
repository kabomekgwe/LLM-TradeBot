"""Model Training - Unified training pipeline for all ML models.

Provides automated training, validation, and persistence for:
- Data preparation and feature engineering
- Model training with cross-validation
- Hyperparameter optimization (optional)
- Model persistence and versioning
- Performance evaluation and reporting
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import json

from .feature_engineering import FeatureEngineer
from .ensemble import EnsemblePredictor, ModelType


class ModelTrainer:
    """Automated training pipeline for ML models.

    Handles end-to-end training workflow:
    1. Data collection and preparation
    2. Feature engineering
    3. Train/test split
    4. Model training with validation
    5. Performance evaluation
    6. Model persistence

    Example:
        >>> trainer = ModelTrainer(symbol="BTC/USDT")
        >>> trainer.prepare_data(historical_df)
        >>> trainer.train_models()
        >>> trainer.evaluate()
        >>> trainer.save_models()
    """

    def __init__(
        self,
        symbol: str,
        model_dir: Optional[Path] = None,
        feature_windows: Optional[List[int]] = None,
        target_horizon: int = 1,
        test_size: float = 0.2,
    ):
        """Initialize model trainer.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            model_dir: Directory to save models (default: models/)
            feature_windows: Lookback windows for features
            target_horizon: Forward periods for prediction target
            test_size: Fraction of data for testing
        """
        self.logger = logging.getLogger(__name__)

        self.symbol = symbol
        self.model_dir = model_dir or Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Feature engineering
        self.feature_engineer = FeatureEngineer(
            windows=feature_windows or [5, 10, 20, 50],
            include_target=True,
            target_horizon=target_horizon,
        )

        # Test/train split
        self.test_size = test_size

        # Data containers
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None

        # Models
        self.ensemble: Optional[EnsemblePredictor] = None

        # Evaluation results
        self.evaluation_results: Dict[str, Any] = {}

    def prepare_data(self, df: pd.DataFrame):
        """Prepare data for training.

        Args:
            df: DataFrame with OHLCV data
        """
        self.logger.info(f"Preparing data for {self.symbol}...")

        # Engineer features
        features_df = self.feature_engineer.transform(df)

        # Extract features and target
        target_col = 'target_binary'

        if target_col not in features_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in features")

        # Get feature columns (exclude OHLCV and target columns)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'target_binary']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]

        self.feature_names = feature_cols

        X = features_df[feature_cols].values
        y = features_df[target_col].values

        # Split into train/test
        split_idx = int(len(X) * (1 - self.test_size))

        self.X_train = X[:split_idx]
        self.y_train = y[:split_idx]
        self.X_test = X[split_idx:]
        self.y_test = y[split_idx:]

        self.logger.info(
            f"Data prepared: {len(self.X_train)} train samples, "
            f"{len(self.X_test)} test samples, "
            f"{len(self.feature_names)} features"
        )

        # Check class balance
        train_positive = (self.y_train == 1).sum()
        train_negative = (self.y_train == 0).sum()
        self.logger.info(
            f"Class balance (train): {train_positive} positive ({train_positive / len(self.y_train) * 100:.1f}%), "
            f"{train_negative} negative ({train_negative / len(self.y_train) * 100:.1f}%)"
        )

    def train_models(
        self,
        models: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Train ensemble models.

        Args:
            models: List of model types to train (default: all)
            weights: Ensemble weights for each model
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        self.logger.info("Training ensemble models...")

        # Create ensemble
        self.ensemble = EnsemblePredictor(
            models=models,
            weights=weights,
            model_dir=self.model_dir,
        )

        # Train
        self.ensemble.train(
            X=self.X_train,
            y=self.y_train,
            validation_split=0.2,
        )

        self.logger.info("Model training complete")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate trained models on test set.

        Returns:
            Dictionary with evaluation metrics
        """
        if self.ensemble is None:
            raise ValueError("Models not trained. Call train_models() first.")

        if self.X_test is None:
            raise ValueError("Test data not available")

        self.logger.info("Evaluating models on test set...")

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
        )

        # Get ensemble predictions
        y_pred_proba = self.ensemble.predict(self.X_test)
        y_pred = self.ensemble.predict_binary(self.X_test, threshold=0.5)

        # Calculate metrics
        results = {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'test_samples': len(self.X_test),
            'metrics': {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, zero_division=0),
                'recall': recall_score(self.y_test, y_pred, zero_division=0),
                'f1_score': f1_score(self.y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            },
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
            'individual_models': self.ensemble.get_model_metrics(),
        }

        self.evaluation_results = results

        # Log results
        self.logger.info("Ensemble Evaluation Results:")
        self.logger.info(f"  Accuracy:  {results['metrics']['accuracy']:.4f}")
        self.logger.info(f"  Precision: {results['metrics']['precision']:.4f}")
        self.logger.info(f"  Recall:    {results['metrics']['recall']:.4f}")
        self.logger.info(f"  F1 Score:  {results['metrics']['f1_score']:.4f}")
        self.logger.info(f"  ROC AUC:   {results['metrics']['roc_auc']:.4f}")

        return results

    def save_models(self):
        """Save trained models and metadata to disk."""
        if self.ensemble is None:
            raise ValueError("No models to save. Train models first.")

        self.logger.info(f"Saving models for {self.symbol}...")

        # Save ensemble models
        self.ensemble.save(self.symbol)

        # Save feature names
        symbol_dir = self.model_dir / self.symbol.replace("/", "_")
        feature_metadata = {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_windows': self.feature_engineer.windows,
            'target_horizon': self.feature_engineer.target_horizon,
        }

        feature_path = symbol_dir / "feature_metadata.json"
        with open(feature_path, 'w') as f:
            json.dump(feature_metadata, f, indent=2)

        # Save evaluation results
        if self.evaluation_results:
            eval_path = symbol_dir / "evaluation_results.json"
            with open(eval_path, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2)

        self.logger.info(f"Models saved to {symbol_dir}")

    def load_models(self):
        """Load trained models from disk."""
        self.logger.info(f"Loading models for {self.symbol}...")

        # Create ensemble
        self.ensemble = EnsemblePredictor(model_dir=self.model_dir)

        # Load models
        self.ensemble.load(self.symbol)

        # Load feature metadata
        symbol_dir = self.model_dir / self.symbol.replace("/", "_")
        feature_path = symbol_dir / "feature_metadata.json"

        if feature_path.exists():
            with open(feature_path, 'r') as f:
                metadata = json.load(f)
            self.feature_names = metadata['feature_names']

        # Load evaluation results
        eval_path = symbol_dir / "evaluation_results.json"
        if eval_path.exists():
            with open(eval_path, 'r') as f:
                self.evaluation_results = json.load(f)

        self.logger.info(f"Models loaded from {symbol_dir}")

    def predict(self, df: pd.DataFrame) -> Tuple[float, np.ndarray]:
        """Make prediction on new data.

        Args:
            df: DataFrame with recent OHLCV data

        Returns:
            Tuple of (prediction probability, feature values)
        """
        if self.ensemble is None:
            raise ValueError("Models not loaded. Call load_models() first.")

        # Engineer features
        features_df = self.feature_engineer.transform(df)

        # Get latest features
        if self.feature_names is None:
            raise ValueError("Feature names not available")

        latest_features = features_df[self.feature_names].iloc[-1:].values

        # Predict
        prediction = self.ensemble.predict(latest_features)[0]

        return prediction, latest_features

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from ensemble.

        Returns:
            DataFrame with features and importance scores
        """
        if self.ensemble is None:
            raise ValueError("Models not loaded")

        importance = self.ensemble.get_ensemble_feature_importance()

        if self.feature_names is None or len(importance) == 0:
            return pd.DataFrame()

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
        })

        return df.sort_values('importance', ascending=False)

    def analyze_predictions(self) -> Dict[str, Any]:
        """Analyze prediction performance on test set.

        Returns:
            Dictionary with detailed analysis
        """
        if self.ensemble is None or self.X_test is None:
            raise ValueError("Models or test data not available")

        # Get predictions
        y_pred_proba = self.ensemble.predict(self.X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Analyze by confidence level
        confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        analysis = {
            'total_predictions': len(y_pred_proba),
            'by_confidence': {},
        }

        for i in range(len(confidence_bins) - 1):
            low = confidence_bins[i]
            high = confidence_bins[i + 1]

            # Find predictions in this confidence range
            mask = (y_pred_proba >= low) & (y_pred_proba < high)
            n_samples = mask.sum()

            if n_samples > 0:
                accuracy = accuracy_score(self.y_test[mask], y_pred[mask])
            else:
                accuracy = 0.0

            analysis['by_confidence'][f'{low:.1f}-{high:.1f}'] = {
                'count': int(n_samples),
                'accuracy': float(accuracy),
            }

        return analysis

    def retrain(self, df: pd.DataFrame):
        """Retrain models with new data.

        Args:
            df: Updated OHLCV DataFrame
        """
        self.logger.info("Retraining models with new data...")

        # Prepare data
        self.prepare_data(df)

        # Retrain
        self.train_models()

        # Evaluate
        self.evaluate()

        # Save
        self.save_models()

        self.logger.info("Retrain complete")

    def __repr__(self) -> str:
        """String representation."""
        status = "trained" if self.ensemble else "not trained"
        return f"ModelTrainer(symbol={self.symbol}, status={status})"
