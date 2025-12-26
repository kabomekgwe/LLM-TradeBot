"""LightGBM Model - Fast gradient boosting for price prediction.

LightGBM is optimized for speed and accuracy, particularly effective
for tabular data with many features.

Advantages:
- Very fast training and prediction
- Handles large datasets efficiently
- Built-in categorical feature support
- Robust to overfitting with proper regularization
"""

import logging
import numpy as np
from typing import Optional, Dict, Any
import lightgbm as lgb


class LightGBMModel:
    """LightGBM classifier for price direction prediction.

    Wrapper around LightGBM with optimized hyperparameters
    for cryptocurrency price forecasting.

    Example:
        >>> model = LightGBMModel()
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict_proba(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 7,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize LightGBM model.

        Args:
            n_estimators: Number of boosting iterations
            learning_rate: Learning rate (eta)
            max_depth: Maximum tree depth
            num_leaves: Maximum leaves per tree
            min_child_samples: Minimum samples per leaf
            subsample: Fraction of samples for each tree
            colsample_bytree: Fraction of features for each tree
            random_state: Random seed
            **kwargs: Additional LightGBM parameters
        """
        self.logger = logging.getLogger(__name__)

        # Model parameters
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'subsample_freq': 1,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': -1,
        }

        # Update with additional parameters
        self.params.update(kwargs)

        # Trained model
        self.model: Optional[lgb.LGBMClassifier] = None

        # Feature importance
        self._feature_importance: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 10,
    ):
        """Train LightGBM model.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_stopping_rounds: Stop if no improvement for N rounds
        """
        self.logger.info("Training LightGBM model...")

        # Create model
        self.model = lgb.LGBMClassifier(**self.params)

        # Prepare validation set
        eval_set = None
        callbacks = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

        # Train
        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            callbacks=callbacks,
        )

        # Store feature importance
        self._feature_importance = self.model.feature_importances_

        self.logger.info(
            f"LightGBM training complete. "
            f"Best iteration: {self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') else self.params['n_estimators']}"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Features to predict on

        Returns:
            Array of predicted labels (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features to predict on

        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.predict_proba(X)

    def get_feature_importance(self, importance_type: str = 'gain') -> np.ndarray:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance ('gain', 'split', or 'weight')

        Returns:
            Array of feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if importance_type == 'gain':
            return self.model.feature_importances_
        elif importance_type == 'split':
            # Number of times feature is used for splitting
            if hasattr(self.model, 'booster_'):
                return self.model.booster_.feature_importance(importance_type='split')
            else:
                return self._feature_importance
        else:
            return self._feature_importance

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        return self.params.copy()

    def set_params(self, **params):
        """Update model parameters.

        Args:
            **params: Parameters to update
        """
        self.params.update(params)

        # Recreate model if it exists
        if self.model is not None:
            self.logger.warning("Parameters updated. Model needs retraining.")
            self.model = None

    def get_booster(self) -> Optional[lgb.Booster]:
        """Get underlying LightGBM booster.

        Returns:
            LightGBM Booster object or None if not trained
        """
        if self.model is None:
            return None

        if hasattr(self.model, 'booster_'):
            return self.model.booster_
        else:
            return None

    def save_model(self, filepath: str):
        """Save model to file.

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")

        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        self.logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'LightGBMModel':
        """Load model from file.

        Args:
            filepath: Path to load model from

        Returns:
            Loaded LightGBMModel instance
        """
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        return model

    def get_training_log(self) -> Dict[str, Any]:
        """Get training metrics log.

        Returns:
            Dictionary with training history
        """
        if self.model is None:
            return {}

        log = {
            'n_estimators': self.params['n_estimators'],
            'learning_rate': self.params['learning_rate'],
            'max_depth': self.params['max_depth'],
        }

        if hasattr(self.model, 'best_iteration_'):
            log['best_iteration'] = self.model.best_iteration_

        if hasattr(self.model, 'best_score_'):
            log['best_score'] = self.model.best_score_

        if hasattr(self.model, 'evals_result_'):
            log['eval_results'] = self.model.evals_result_

        return log

    def __repr__(self) -> str:
        """String representation."""
        status = "trained" if self.model is not None else "not trained"
        return f"LightGBMModel(n_estimators={self.params['n_estimators']}, status={status})"
