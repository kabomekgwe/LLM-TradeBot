"""XGBoost Model - Extreme Gradient Boosting for price prediction.

XGBoost is known for its strong performance in competitions and
provides regularization to prevent overfitting.

Advantages:
- Excellent accuracy on structured data
- Built-in regularization (L1, L2)
- Handles missing values automatically
- Parallel tree construction
"""

import logging
import numpy as np
from typing import Optional, Dict, Any
import xgboost as xgb


class XGBoostModel:
    """XGBoost classifier for price direction prediction.

    Wrapper around XGBoost with optimized hyperparameters
    for cryptocurrency price forecasting.

    Example:
        >>> model = XGBoostModel()
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict_proba(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate (eta)
            max_depth: Maximum tree depth
            min_child_weight: Minimum sum of instance weight in a child
            gamma: Minimum loss reduction for split
            subsample: Fraction of samples for each tree
            colsample_bytree: Fraction of features for each tree
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            random_state: Random seed
            **kwargs: Additional XGBoost parameters
        """
        self.logger = logging.getLogger(__name__)

        # Model parameters
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'n_jobs': -1,
            'tree_method': 'hist',  # Faster histogram-based algorithm
        }

        # Update with additional parameters
        self.params.update(kwargs)

        # Trained model
        self.model: Optional[xgb.XGBClassifier] = None

        # Feature importance
        self._feature_importance: Optional[np.ndarray] = None

        # Training history
        self.evals_result: Dict[str, Dict[str, list]] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 10,
    ):
        """Train XGBoost model.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_stopping_rounds: Stop if no improvement for N rounds
        """
        self.logger.info("Training XGBoost model...")

        # Create model
        self.model = xgb.XGBClassifier(**self.params)

        # Prepare validation set
        eval_set = None

        if X_val is not None and y_val is not None:
            eval_set = [(X, y), (X_val, y_val)]
        else:
            eval_set = [(X, y)]

        # Train
        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
            verbose=False,
        )

        # Store feature importance
        self._feature_importance = self.model.feature_importances_

        # Store evaluation results
        if hasattr(self.model, 'evals_result_'):
            self.evals_result = self.model.evals_result_

        self.logger.info(
            f"XGBoost training complete. "
            f"Best iteration: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.params['n_estimators']}"
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

    def get_feature_importance(self, importance_type: str = 'weight') -> np.ndarray:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover', or 'total_gain')

        Returns:
            Array of feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if importance_type == 'weight':
            # Number of times feature is used for splitting
            return self.model.feature_importances_
        elif importance_type in ['gain', 'cover', 'total_gain']:
            # Get booster and extract importance
            booster = self.model.get_booster()
            importance_dict = booster.get_score(importance_type=importance_type)

            # Convert to array (features may be missing if not used)
            n_features = len(self._feature_importance)
            importance_array = np.zeros(n_features)

            for i in range(n_features):
                feature_name = f'f{i}'
                if feature_name in importance_dict:
                    importance_array[i] = importance_dict[feature_name]

            return importance_array
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

    def get_booster(self) -> Optional[xgb.Booster]:
        """Get underlying XGBoost booster.

        Returns:
            XGBoost Booster object or None if not trained
        """
        if self.model is None:
            return None

        return self.model.get_booster()

    def save_model(self, filepath: str):
        """Save model to file.

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")

        # XGBoost has native save method
        self.model.save_model(filepath)
        self.logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'XGBoostModel':
        """Load model from file.

        Args:
            filepath: Path to load model from

        Returns:
            Loaded XGBoostModel instance
        """
        instance = cls()
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(filepath)

        # Restore feature importance if available
        if hasattr(instance.model, 'feature_importances_'):
            instance._feature_importance = instance.model.feature_importances_

        return instance

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
            'eval_results': self.evals_result,
        }

        if hasattr(self.model, 'best_iteration'):
            log['best_iteration'] = self.model.best_iteration

        if hasattr(self.model, 'best_score'):
            log['best_score'] = self.model.best_score

        return log

    def plot_importance(self, max_num_features: int = 20, importance_type: str = 'weight'):
        """Plot feature importance.

        Args:
            max_num_features: Maximum number of features to plot
            importance_type: Type of importance to plot
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        try:
            import matplotlib.pyplot as plt

            xgb.plot_importance(
                self.model,
                importance_type=importance_type,
                max_num_features=max_num_features,
            )
            plt.tight_layout()
            plt.show()

        except ImportError:
            self.logger.warning("matplotlib not installed. Cannot plot importance.")

    def __repr__(self) -> str:
        """String representation."""
        status = "trained" if self.model is not None else "not trained"
        return f"XGBoostModel(n_estimators={self.params['n_estimators']}, status={status})"
