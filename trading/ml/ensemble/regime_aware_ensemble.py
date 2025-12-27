"""
Regime-aware ensemble that automatically switches strategies based on volatility regimes.

Integrates with Phase 5's HMM volatility regime detector to select optimal ensemble
strategy (voting, stacking, or dynamic) based on current market conditions.
"""

from typing import Dict, Any, Tuple
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from trading.ml.ensemble.base_ensemble import BaseEnsemble
from trading.ml.ensemble.model_registry import ModelRegistry
from trading.logging_config import get_logger
from trading.exceptions import ModelError

logger = get_logger(__name__)


class RegimeAwareEnsemble(BaseEnsemble):
    """
    Regime-aware ensemble with automatic strategy switching.

    Combines LightGBM, XGBoost, and Random Forest with three ensemble strategies:
    - Voting: Soft voting (probability averaging) for high volatility regimes
    - Stacking: Meta-model learns combination for low volatility regimes
    - Dynamic: Best recent performer for transitional regimes

    Strategy selection based on Phase 5's HMM volatility regime detector.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize regime-aware ensemble.

        Args:
            n_estimators: Number of trees for tree-based models
            random_state: Random seed for reproducibility
        """
        super().__init__(n_estimators=n_estimators, random_state=random_state)

        # Initialize model registry
        self.model_registry = ModelRegistry()

        # Initialize base models
        self._initialize_models()

        # Ensemble strategies (initialized during fit)
        self.voting_ensemble = None
        self.stacking_ensemble = None

        # Dynamic selection state
        self.recent_performance = {
            'lgbm': [],
            'xgb': [],
            'rf': []
        }

        logger.info(
            "Regime-aware ensemble initialized",
            extra={
                'n_estimators': n_estimators,
                'n_models': len(self.model_registry)
            }
        )

    def _initialize_models(self) -> None:
        """Initialize the three base models."""
        # LightGBM
        lgbm_model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=10,
            learning_rate=0.1,
            random_state=self.random_state,
            verbose=-1
        )
        self.model_registry.register_model('lgbm', lgbm_model)

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=self.random_state,
            verbosity=0
        )
        self.model_registry.register_model('xgb', xgb_model)

        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=20,
            random_state=self.random_state
        )
        self.model_registry.register_model('rf', rf_model)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegimeAwareEnsemble':
        """
        Train all ensemble strategies.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            self: Fitted ensemble
        """
        self._validate_features(X, context="training")

        logger.info(
            "Training regime-aware ensemble",
            extra={
                'n_samples': X.shape[0],
                'n_features': X.shape[1]
            }
        )

        # Train base models individually (for dynamic selection)
        for name, model in self.model_registry.get_models().items():
            logger.info(f"Training {name}...")
            model.fit(X, y)

        # Create and train voting ensemble
        estimators = list(self.model_registry.get_models().items())
        self.voting_ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        self.voting_ensemble.fit(X, y)

        # Create and train stacking ensemble
        self.stacking_ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=self.random_state),
            cv=5,
            stack_method='predict_proba'
        )
        self.stacking_ensemble.fit(X, y)

        self.n_features_in_ = X.shape[1]
        self.is_fitted = True

        logger.info(
            "Regime-aware ensemble trained",
            extra={
                'n_features': self.n_features_in_,
                'strategies': ['voting', 'stacking', 'dynamic']
            }
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using voting ensemble (default).

        For regime-aware predictions, use predict_with_regime() instead.

        Args:
            X: Features for prediction (n_samples, n_features)

        Returns:
            predictions: Class predictions (n_samples,)
        """
        self._validate_fitted()
        self._validate_features(X, context="prediction")

        predictions = self.voting_ensemble.predict(X)

        self._log_prediction(
            n_samples=X.shape[0],
            predictions=predictions,
            metadata={'strategy': 'voting', 'regime_aware': False}
        )

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using voting ensemble (default).

        For regime-aware predictions, use predict_with_regime() instead.

        Args:
            X: Features for prediction (n_samples, n_features)

        Returns:
            probabilities: Class probabilities (n_samples, n_classes)
        """
        self._validate_fitted()
        self._validate_features(X, context="prediction")

        probabilities = self.voting_ensemble.predict_proba(X)

        return probabilities

    def predict_with_regime(
        self,
        X: np.ndarray,
        regime_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Make regime-aware predictions with automatic strategy selection.

        Args:
            X: Features for prediction (n_samples, n_features)
            regime_info: Dict with keys:
                - current_regime: 0 or 1
                - is_low_volatility: 0 or 1
                - regime_prob_0: float [0, 1]
                - regime_prob_1: float [0, 1]

        Returns:
            predictions: Class predictions (n_samples,)
            metadata: Dict with strategy, reason, individual predictions
        """
        self._validate_fitted()
        self._validate_features(X, context="prediction")

        # Validate regime info
        self._validate_regime_info(regime_info)

        # Extract regime details
        is_low_vol = regime_info['is_low_volatility']
        regime = regime_info['current_regime']
        regime_prob = regime_info[f'regime_prob_{regime}']

        # Get individual model predictions
        individual_predictions = {}
        individual_probabilities = {}

        for name, model in self.model_registry.get_models().items():
            individual_predictions[name] = int(model.predict(X)[0])
            individual_probabilities[name] = float(model.predict_proba(X)[0, 1])

        # Strategy selection based on regime
        if is_low_vol and regime_prob > 0.7:
            # Low volatility, confident → Stacking
            strategy = 'stacking'
            prediction = self.stacking_ensemble.predict(X)
            prediction_proba = self.stacking_ensemble.predict_proba(X)[0, 1]
            reason = f"Low volatility regime (prob={regime_prob:.2f}) → meta-model learns combination"

        elif regime_prob < 0.6:
            # Transitional regime → Dynamic selection (best recent performer)
            strategy = 'dynamic'
            # Use model with highest individual probability
            best_model = max(individual_probabilities, key=individual_probabilities.get)
            prediction = np.array([individual_predictions[best_model]])
            prediction_proba = individual_probabilities[best_model]
            reason = f"Transitional regime (prob={regime_prob:.2f}) → selected {best_model}"

        else:
            # High volatility or default → Voting
            strategy = 'voting'
            prediction = self.voting_ensemble.predict(X)
            prediction_proba = self.voting_ensemble.predict_proba(X)[0, 1]
            reason = f"High volatility regime (prob={regime_prob:.2f}) → majority vote"

        # Metadata for observability
        metadata = {
            'strategy': strategy,
            'reason': reason,
            'regime': int(regime),
            'is_low_volatility': bool(is_low_vol),
            'regime_confidence': float(regime_prob),
            'individual_predictions': individual_predictions,
            'individual_probabilities': individual_probabilities,
            'ensemble_probability': float(prediction_proba)
        }

        # Log prediction
        logger.info(
            "Regime-aware prediction",
            extra={
                'strategy': strategy,
                'regime': int(regime),
                'is_low_vol': bool(is_low_vol),
                'regime_prob': float(regime_prob),
                'prediction': int(prediction[0]),
                'confidence': float(prediction_proba)
            }
        )

        return prediction, metadata

    def _validate_regime_info(self, regime_info: Dict[str, Any]) -> None:
        """
        Validate regime information dict.

        Args:
            regime_info: Regime information to validate

        Raises:
            ModelError: If regime info is invalid
        """
        required_keys = ['current_regime', 'is_low_volatility', 'regime_prob_0', 'regime_prob_1']
        missing_keys = [key for key in required_keys if key not in regime_info]

        if missing_keys:
            raise ModelError(
                f"Missing regime info keys: {missing_keys}",
                model_name="RegimeAwareEnsemble",
                context={'required_keys': required_keys, 'missing_keys': missing_keys}
            )

        # Validate regime probabilities sum to 1
        prob_sum = regime_info['regime_prob_0'] + regime_info['regime_prob_1']
        if not np.isclose(prob_sum, 1.0, atol=0.01):
            raise ModelError(
                f"Regime probabilities don't sum to 1: {prob_sum}",
                model_name="RegimeAwareEnsemble",
                context={'regime_info': regime_info}
            )

    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        """
        Aggregate feature importance across all models.

        Args:
            feature_names: List of feature names (must match n_features)

        Returns:
            importance: Dictionary of {feature_name: aggregated_importance}
                        sorted by importance (descending)
        """
        if len(feature_names) != self.n_features_in_:
            raise ModelError(
                f"Feature names count mismatch: expected {self.n_features_in_}, got {len(feature_names)}",
                model_name="RegimeAwareEnsemble",
                context={'expected': self.n_features_in_, 'received': len(feature_names)}
            )

        importance_dict = {}

        # Extract importance from each model
        for name, model in self.model_registry.get_models().items():
            importance_dict[name] = dict(zip(feature_names, model.feature_importances_))

        # Average importance across models
        aggregated = {}
        for feature in feature_names:
            feature_importances = [
                importance_dict[model][feature]
                for model in importance_dict.keys()
            ]
            aggregated[feature] = np.mean(feature_importances)

        # Sort by importance (descending)
        return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))
