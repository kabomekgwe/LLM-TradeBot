"""
Voting ensemble wrapper for sklearn VotingClassifier.
"""

from typing import Dict, Any
import numpy as np
from sklearn.ensemble import VotingClassifier

from trading.ml.ensemble.base_ensemble import BaseEnsemble
from trading.logging_config import get_logger

logger = get_logger(__name__)


class VotingEnsemble(BaseEnsemble):
    """
    Wrapper for sklearn VotingClassifier.

    Uses soft voting (probability averaging) for robust predictions,
    especially in high volatility regimes.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        voting: str = 'soft',
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Initialize voting ensemble.

        Args:
            models: Dictionary of {name: model_instance}
            voting: 'soft' (probability averaging) or 'hard' (majority vote)
            n_estimators: Number of estimators (inherited from base)
            random_state: Random seed (inherited from base)
        """
        super().__init__(n_estimators=n_estimators, random_state=random_state)

        # Convert models dict to list of tuples for sklearn
        estimators = list(models.items())

        self.voting_classifier = VotingClassifier(
            estimators=estimators,
            voting=voting
        )

        self.voting = voting
        logger.info(
            "Voting ensemble initialized",
            extra={
                'voting': voting,
                'n_models': len(models),
                'model_names': list(models.keys())
            }
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'VotingEnsemble':
        """
        Train voting ensemble.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            self: Fitted ensemble
        """
        self._validate_features(X, context="training")

        logger.info(
            "Training voting ensemble",
            extra={
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'voting': self.voting
            }
        )

        self.voting_classifier.fit(X, y)
        self.n_features_in_ = X.shape[1]
        self.is_fitted = True

        logger.info(
            "Voting ensemble trained",
            extra={'n_features': self.n_features_in_}
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using voting ensemble.

        Args:
            X: Features for prediction (n_samples, n_features)

        Returns:
            predictions: Class predictions (n_samples,)
        """
        self._validate_fitted()
        self._validate_features(X, context="prediction")

        predictions = self.voting_classifier.predict(X)

        self._log_prediction(
            n_samples=X.shape[0],
            predictions=predictions,
            metadata={'voting': self.voting}
        )

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using voting ensemble.

        Args:
            X: Features for prediction (n_samples, n_features)

        Returns:
            probabilities: Class probabilities (n_samples, n_classes)
        """
        self._validate_fitted()
        self._validate_features(X, context="prediction")

        probabilities = self.voting_classifier.predict_proba(X)

        self._log_prediction(
            n_samples=X.shape[0],
            predictions=self.voting_classifier.predict(X),
            probabilities=probabilities,
            metadata={'voting': self.voting}
        )

        return probabilities
