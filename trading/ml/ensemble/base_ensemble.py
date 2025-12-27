"""
Base ensemble class providing common functionality for all ensemble strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from trading.logging_config import get_logger
from trading.exceptions import ModelError

logger = get_logger(__name__)


class BaseEnsemble(ABC):
    """
    Abstract base class for ensemble strategies.

    Provides common functionality including logging, error handling,
    and feature validation. All ensemble implementations must inherit
    from this class and implement the abstract methods.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize base ensemble.

        Args:
            n_estimators: Number of trees for tree-based models
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.is_fitted = False
        self.n_features_in_ = None
        logger.info(
            "Initializing ensemble",
            extra={
                'ensemble_type': self.__class__.__name__,
                'n_estimators': n_estimators,
                'random_state': random_state
            }
        )

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseEnsemble':
        """
        Train the ensemble on training data.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            self: Fitted ensemble
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble.

        Args:
            X: Features for prediction (n_samples, n_features)

        Returns:
            predictions: Class predictions (n_samples,)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using the ensemble.

        Args:
            X: Features for prediction (n_samples, n_features)

        Returns:
            probabilities: Class probabilities (n_samples, n_classes)
        """
        pass

    def _validate_features(self, X: np.ndarray, context: str = "prediction") -> None:
        """
        Validate feature dimensions match training.

        Args:
            X: Features to validate
            context: Context for error messages (e.g., "prediction", "training")

        Raises:
            ModelError: If feature dimensions don't match
        """
        if not isinstance(X, np.ndarray):
            raise ModelError(
                f"Features must be numpy array, got {type(X).__name__}",
                model_name=self.__class__.__name__,
                context={'context': context}
            )

        if X.ndim != 2:
            raise ModelError(
                f"Features must be 2D array, got {X.ndim}D",
                model_name=self.__class__.__name__,
                context={'context': context, 'shape': X.shape}
            )

        if self.is_fitted and self.n_features_in_ is not None:
            if X.shape[1] != self.n_features_in_:
                raise ModelError(
                    f"Feature count mismatch: expected {self.n_features_in_}, got {X.shape[1]}",
                    model_name=self.__class__.__name__,
                    context={
                        'context': context,
                        'expected_features': self.n_features_in_,
                        'received_features': X.shape[1]
                    }
                )

    def _validate_fitted(self) -> None:
        """
        Check if ensemble has been fitted.

        Raises:
            ModelError: If ensemble has not been fitted
        """
        if not self.is_fitted:
            raise ModelError(
                "Ensemble must be fitted before making predictions",
                model_name=self.__class__.__name__,
                context={'is_fitted': False}
            )

    def _log_prediction(
        self,
        n_samples: int,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log prediction details for observability.

        Args:
            n_samples: Number of samples predicted
            predictions: Class predictions
            probabilities: Class probabilities (optional)
            metadata: Additional metadata to log
        """
        log_data = {
            'ensemble_type': self.__class__.__name__,
            'n_samples': n_samples,
            'prediction_distribution': {
                'class_0': int(np.sum(predictions == 0)),
                'class_1': int(np.sum(predictions == 1))
            }
        }

        if probabilities is not None:
            log_data['mean_confidence'] = float(np.mean(np.max(probabilities, axis=1)))

        if metadata:
            log_data.update(metadata)

        logger.info("Ensemble prediction complete", extra=log_data)
