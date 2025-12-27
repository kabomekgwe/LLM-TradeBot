"""
Stacking ensemble wrapper for sklearn StackingClassifier.
"""

from typing import Dict, Any
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from trading.ml.ensemble.base_ensemble import BaseEnsemble
from trading.logging_config import get_logger

logger = get_logger(__name__)


class StackingEnsemble(BaseEnsemble):
    """
    Wrapper for sklearn StackingClassifier.

    Uses meta-model (LogisticRegression) to learn optimal combination
    of base models, especially effective in low volatility regimes.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        cv: int = 5,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Initialize stacking ensemble.

        Args:
            models: Dictionary of {name: model_instance}
            cv: Number of cross-validation folds
            n_estimators: Number of estimators (inherited from base)
            random_state: Random seed (inherited from base)
        """
        super().__init__(n_estimators=n_estimators, random_state=random_state)

        # Convert models dict to list of tuples for sklearn
        estimators = list(models.items())

        self.stacking_classifier = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=random_state),
            cv=cv,
            stack_method='predict_proba'
        )

        self.cv = cv
        logger.info(
            "Stacking ensemble initialized",
            extra={
                'cv': cv,
                'n_models': len(models),
                'model_names': list(models.keys()),
                'meta_model': 'LogisticRegression'
            }
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """
        Train stacking ensemble with cross-validation.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            self: Fitted ensemble
        """
        self._validate_features(X, context="training")

        logger.info(
            "Training stacking ensemble",
            extra={
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'cv': self.cv
            }
        )

        self.stacking_classifier.fit(X, y)
        self.n_features_in_ = X.shape[1]
        self.is_fitted = True

        logger.info(
            "Stacking ensemble trained",
            extra={
                'n_features': self.n_features_in_,
                'meta_model_trained': True
            }
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using stacking ensemble.

        Args:
            X: Features for prediction (n_samples, n_features)

        Returns:
            predictions: Class predictions (n_samples,)
        """
        self._validate_fitted()
        self._validate_features(X, context="prediction")

        predictions = self.stacking_classifier.predict(X)

        self._log_prediction(
            n_samples=X.shape[0],
            predictions=predictions,
            metadata={'cv': self.cv, 'meta_model': 'LogisticRegression'}
        )

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using stacking ensemble.

        Args:
            X: Features for prediction (n_samples, n_features)

        Returns:
            probabilities: Class probabilities (n_samples, n_classes)
        """
        self._validate_fitted()
        self._validate_features(X, context="prediction")

        probabilities = self.stacking_classifier.predict_proba(X)

        self._log_prediction(
            n_samples=X.shape[0],
            predictions=self.stacking_classifier.predict(X),
            probabilities=probabilities,
            metadata={'cv': self.cv, 'meta_model': 'LogisticRegression'}
        )

        return probabilities
