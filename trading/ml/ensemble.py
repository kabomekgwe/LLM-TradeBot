"""Ensemble Predictor - Combine multiple ML models with voting.

Implements ensemble learning with:
- LightGBM (gradient boosting)
- XGBoost (extreme gradient boosting)
- Random Forest (tree ensemble)
- LSTM (deep learning)

Uses weighted voting to combine predictions from multiple models
for more robust and accurate forecasts.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path
import pickle
import json


class ModelType(str, Enum):
    """Supported model types."""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    LSTM = "lstm"


class EnsemblePredictor:
    """Ensemble predictor combining multiple ML models.

    Combines predictions from multiple models using weighted voting
    to produce more accurate and robust forecasts.

    Example:
        >>> predictor = EnsemblePredictor(
        ...     models=['lightgbm', 'xgboost', 'random_forest'],
        ...     weights={'lightgbm': 0.4, 'xgboost': 0.4, 'random_forest': 0.2}
        ... )
        >>> predictor.train(X_train, y_train)
        >>> prediction = predictor.predict(X_test)
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        model_dir: Optional[Path] = None,
    ):
        """Initialize ensemble predictor.

        Args:
            models: List of model types to include (default: all)
            weights: Dictionary mapping model types to weights (default: equal weights)
            model_dir: Directory to save/load models (default: models/)
        """
        self.logger = logging.getLogger(__name__)

        # Default to all models
        if models is None:
            models = [m.value for m in ModelType]

        self.model_types = models

        # Default to equal weights
        if weights is None:
            weight = 1.0 / len(self.model_types)
            weights = {model: weight for model in self.model_types}
        else:
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}

        self.weights = weights

        # Model directory
        self.model_dir = model_dir or Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Trained models
        self.models: Dict[str, Any] = {}

        # Feature names
        self.feature_names: Optional[List[str]] = None

        # Model performance metrics
        self.model_metrics: Dict[str, Dict[str, float]] = {}

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        validation_split: float = 0.2,
        **kwargs,
    ):
        """Train all models in the ensemble.

        Args:
            X: Training features
            y: Training labels
            validation_split: Fraction of data for validation
            **kwargs: Additional arguments passed to individual models
        """
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_array = X.values
        else:
            X_array = X

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        # Split into train/validation
        split_idx = int(len(X_array) * (1 - validation_split))
        X_train, X_val = X_array[:split_idx], X_array[split_idx:]
        y_train, y_val = y_array[:split_idx], y_array[split_idx:]

        self.logger.info(f"Training ensemble with {len(self.model_types)} models")
        self.logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Train each model
        for model_type in self.model_types:
            self.logger.info(f"Training {model_type}...")

            try:
                if model_type == ModelType.LIGHTGBM.value:
                    from .models.lightgbm_model import LightGBMModel
                    model = LightGBMModel()

                elif model_type == ModelType.XGBOOST.value:
                    from .models.xgboost_model import XGBoostModel
                    model = XGBoostModel()

                elif model_type == ModelType.RANDOM_FOREST.value:
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        random_state=42,
                        n_jobs=-1,
                    )

                elif model_type == ModelType.LSTM.value:
                    from .models.lstm_model import LSTMModel
                    model = LSTMModel(input_dim=X_train.shape[1])

                else:
                    self.logger.warning(f"Unknown model type: {model_type}, skipping")
                    continue

                # Train model
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                else:
                    model.train(X_train, y_train, X_val, y_val)

                # Evaluate on validation set
                metrics = self._evaluate_model(model, X_val, y_val)
                self.model_metrics[model_type] = metrics

                self.logger.info(
                    f"{model_type} - Accuracy: {metrics['accuracy']:.4f}, "
                    f"Precision: {metrics['precision']:.4f}, "
                    f"Recall: {metrics['recall']:.4f}"
                )

                # Store model
                self.models[model_type] = model

            except Exception as e:
                self.logger.error(f"Failed to train {model_type}: {e}")
                continue

        self.logger.info(f"Ensemble training complete. {len(self.models)}/{len(self.model_types)} models trained")

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make ensemble prediction.

        Args:
            X: Features to predict on

        Returns:
            Array of predictions (probabilities for class 1)
        """
        if not self.models:
            raise ValueError("No models trained. Call train() first.")

        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        # Collect predictions from all models
        predictions = {}

        for model_type, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    # Get probability for class 1
                    pred = model.predict_proba(X_array)[:, 1]
                elif hasattr(model, 'predict'):
                    pred = model.predict(X_array)
                    # Ensure it's probability-like (0-1 range)
                    if pred.min() < 0 or pred.max() > 1:
                        pred = (pred - pred.min()) / (pred.max() - pred.min())
                else:
                    self.logger.warning(f"Model {model_type} has no predict method")
                    continue

                predictions[model_type] = pred

            except Exception as e:
                self.logger.error(f"Prediction failed for {model_type}: {e}")
                continue

        if not predictions:
            raise RuntimeError("All model predictions failed")

        # Weighted ensemble
        ensemble_pred = np.zeros(len(X_array))

        for model_type, pred in predictions.items():
            weight = self.weights.get(model_type, 0)
            ensemble_pred += weight * pred

        return ensemble_pred

    def predict_binary(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Make binary prediction (0 or 1).

        Args:
            X: Features to predict on
            threshold: Probability threshold for class 1

        Returns:
            Array of binary predictions (0 or 1)
        """
        probs = self.predict(X)
        return (probs >= threshold).astype(int)

    def _evaluate_model(self, model: Any, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
        elif hasattr(model, 'predict'):
            y_pred = model.predict(X_val)
            if y_pred.min() >= 0 and y_pred.max() <= 1:
                y_pred_proba = y_pred
                y_pred = (y_pred >= 0.5).astype(int)
            else:
                y_pred_proba = None
        else:
            return {}

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
        }

        if y_pred_proba is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_val, y_pred_proba)
            except:
                pass

        return metrics

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from all models.

        Returns:
            Dictionary mapping model types to feature importance arrays
        """
        importance_dict = {}

        for model_type, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_type] = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):
                importance_dict[model_type] = model.get_feature_importance()
            else:
                self.logger.warning(f"Model {model_type} does not support feature importance")

        return importance_dict

    def get_ensemble_feature_importance(self) -> np.ndarray:
        """Get weighted ensemble feature importance.

        Returns:
            Array of feature importance scores
        """
        importance_dict = self.get_feature_importance()

        if not importance_dict:
            return np.array([])

        # Weighted average
        ensemble_importance = np.zeros(len(next(iter(importance_dict.values()))))

        for model_type, importance in importance_dict.items():
            weight = self.weights.get(model_type, 0)
            ensemble_importance += weight * importance

        return ensemble_importance

    def save(self, symbol: str):
        """Save trained models to disk.

        Args:
            symbol: Trading symbol (for organizing models)
        """
        symbol_dir = self.model_dir / symbol.replace("/", "_")
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Save each model
        for model_type, model in self.models.items():
            model_path = symbol_dir / f"{model_type}.pkl"

            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                self.logger.info(f"Saved {model_type} to {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to save {model_type}: {e}")

        # Save metadata
        metadata = {
            'model_types': self.model_types,
            'weights': self.weights,
            'feature_names': self.feature_names,
            'metrics': self.model_metrics,
        }

        metadata_path = symbol_dir / "ensemble_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved ensemble metadata to {metadata_path}")

    def load(self, symbol: str):
        """Load trained models from disk.

        Args:
            symbol: Trading symbol (for organizing models)
        """
        symbol_dir = self.model_dir / symbol.replace("/", "_")

        if not symbol_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {symbol_dir}")

        # Load metadata
        metadata_path = symbol_dir / "ensemble_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.model_types = metadata['model_types']
        self.weights = metadata['weights']
        self.feature_names = metadata['feature_names']
        self.model_metrics = metadata['metrics']

        # Load each model
        for model_type in self.model_types:
            model_path = symbol_dir / f"{model_type}.pkl"

            if not model_path.exists():
                self.logger.warning(f"Model file not found: {model_path}")
                continue

            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                self.models[model_type] = model
                self.logger.info(f"Loaded {model_type} from {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load {model_type}: {e}")

        self.logger.info(f"Loaded {len(self.models)} models from {symbol_dir}")

    def get_model_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all models.

        Returns:
            Dictionary mapping model types to their metrics
        """
        return self.model_metrics.copy()

    def update_weights(self, weights: Dict[str, float]):
        """Update ensemble weights.

        Args:
            weights: New weights (will be normalized)
        """
        # Normalize weights
        total_weight = sum(weights.values())
        self.weights = {k: v / total_weight for k, v in weights.items()}

        self.logger.info(f"Updated ensemble weights: {self.weights}")
