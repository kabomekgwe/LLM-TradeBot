"""PredictAgent - ML-based price forecasting.

This is Agent #3 in the 8-agent system.
Uses ensemble ML models (LightGBM + XGBoost + Random Forest) with regime-aware
strategy switching to forecast price direction.
"""

from typing import Any, Dict
import numpy as np
import os

from .base_agent import BaseAgent
from ..exceptions import (
    ModelPredictionError,
    InvalidIndicatorDataError,
    ModelError,
)
from ..ml.ensemble.regime_aware_ensemble import RegimeAwareEnsemble
from ..ml.ensemble.persistence import EnsemblePersistence
from ..ml.feature_engineering import FeatureEngineer


class PredictAgent(BaseAgent):
    """Prediction agent - ML price forecasting.

    Uses ensemble ML models (LightGBM + XGBoost + Random Forest) with
    regime-aware strategy switching to predict price direction.
    """

    def __init__(self, provider, config):
        """Initialize PredictAgent with pre-trained ensemble model.

        Args:
            provider: Exchange provider instance
            config: TradingConfig instance
        """
        super().__init__(provider, config)

        # Initialize ensemble and persistence
        self.ensemble = None
        self.metadata = None
        self.feature_engineer = FeatureEngineer(include_target=False)

        # Try to load ensemble models
        persistence = EnsemblePersistence(model_dir='trading/ml/models/ensemble')

        if persistence.model_exists():
            try:
                models, metadata = persistence.load_models()

                # Create ensemble and register loaded models
                self.ensemble = RegimeAwareEnsemble()

                # Replace default models with loaded models
                self.ensemble.model_registry._models = models

                # Mark ensemble as fitted
                self.ensemble.is_fitted = True
                self.ensemble.n_features_in_ = metadata.get('n_features', 86)

                self.metadata = metadata
                self.logger.info(
                    "Loaded ensemble models",
                    extra={
                        'n_models': len(models),
                        'models': list(models.keys()),
                        'n_features': self.ensemble.n_features_in_,
                        'trained_date': metadata.get('trained_date', 'unknown')
                    }
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to load ensemble models: {e}",
                    extra={'error': str(e)},
                    exc_info=True
                )
                self.ensemble = None
        else:
            self.logger.warning("No trained ensemble models found. Run train_ensemble.py first.")

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate ML price prediction using ensemble models.

        Args:
            context: Must contain "quant_analyst" with indicators and regime_info

        Returns:
            Context updated with "ml_prediction" containing:
                - "direction": "up", "down", or "neutral"
                - "confidence": Float 0-1
                - "probability_up": Raw probability for transparency
                - "strategy": Ensemble strategy used ("voting", "stacking", "dynamic")
                - "individual_predictions": Predictions from each model
                - "regime_info": Regime detection details (if available)

        Example:
            >>> result = await agent.execute(context)
            >>> result["ml_prediction"]["direction"]
            'up'
            >>> result["ml_prediction"]["confidence"]
            0.73
            >>> result["ml_prediction"]["strategy"]
            'stacking'
        """
        # Check if ML predictions are enabled
        if not self.config.enable_ml_predictions:
            self.log_decision("ML predictions disabled, returning neutral")
            return {
                "ml_prediction": {
                    "direction": "neutral",
                    "confidence": 0.0,
                    "enabled": False,
                }
            }

        # Fallback if no ensemble trained
        if not self.ensemble:
            self.log_decision("No trained ensemble available, returning neutral")
            return {
                "ml_prediction": {
                    "direction": "neutral",
                    "confidence": 0.0,
                    "reason": "No trained ensemble available - run train_ensemble.py",
                }
            }

        # Get indicators from QuantAnalystAgent
        quant_analyst = context.get("quant_analyst", {})
        indicators = quant_analyst.get("indicators", {})

        if not indicators:
            self.log_decision("No indicator data from QuantAnalyst, returning neutral")
            return {
                "ml_prediction": {
                    "direction": "neutral",
                    "confidence": 0.0,
                    "reason": "No indicator data from QuantAnalyst",
                }
            }

        # Extract feature names from metadata
        feature_names = self.metadata.get('feature_names', [])

        if not feature_names:
            self.log_decision("No feature names in metadata, cannot extract features")
            return {
                "ml_prediction": {
                    "direction": "neutral",
                    "confidence": 0.0,
                    "reason": "No feature names in metadata",
                }
            }

        # Extract features from indicators
        # Note: This is a simplified version - in production, the features would come
        # from QuantAnalystAgent's full feature engineering pipeline
        try:
            # For now, create a feature array matching expected size
            # In production, QuantAnalyst would provide all 86 features
            features = np.zeros((1, len(feature_names)))

            # Map available indicators to features (simplified)
            # This assumes QuantAnalyst has already engineered all features
            if 'engineered_features' in indicators:
                # Use pre-engineered features from QuantAnalyst
                features = np.array([indicators['engineered_features']])
            else:
                # Fallback: return neutral if features not available
                self.log_decision("Engineered features not available in indicators")
                return {
                    "ml_prediction": {
                        "direction": "neutral",
                        "confidence": 0.0,
                        "reason": "Engineered features not available - QuantAnalyst needs update",
                    }
                }

        except (KeyError, TypeError, ValueError) as e:
            self.log_decision(
                "feature_extraction_failed",
                level="error",
                error=str(e),
            )
            raise InvalidIndicatorDataError(f"Failed to extract features: {e}")

        # Get regime info (if available)
        regime_info = quant_analyst.get('regime_info', None)

        # Predict using ensemble
        try:
            if regime_info:
                # Regime-aware prediction
                prediction, metadata = self.ensemble.predict_with_regime(features, regime_info)

                prob_up = metadata['ensemble_probability']
                strategy = metadata['strategy']
                individual_predictions = metadata['individual_predictions']
                individual_probabilities = metadata['individual_probabilities']
                regime_details = {
                    'current_regime': metadata['regime'],
                    'is_low_volatility': metadata['is_low_volatility'],
                    'regime_confidence': metadata['regime_confidence'],
                    'reason': metadata['reason']
                }
            else:
                # Default voting prediction (no regime info)
                prediction = self.ensemble.predict(features)
                prob_up = self.ensemble.predict_proba(features)[0, 1]
                strategy = 'voting'
                individual_predictions = {}
                individual_probabilities = {}
                regime_details = None

                self.log_decision("No regime info available, using default voting strategy")

        except ModelError as e:
            self.log_decision(
                "ensemble_prediction_failed",
                level="critical",
                error=str(e),
                exc_info=True,
            )
            raise ModelPredictionError(f"Ensemble prediction failed: {e}")

        # Convert to direction and confidence
        direction = "up" if prob_up > 0.5 else "down"
        confidence = abs(prob_up - 0.5) * 2  # Scale to [0, 1]

        ml_prediction = {
            "direction": direction,
            "confidence": confidence,
            "probability_up": prob_up,
            "strategy": strategy,
            "individual_predictions": individual_predictions,
            "individual_probabilities": individual_probabilities,
            "regime_info": regime_details,
            "reason": f"Ensemble {strategy}: {prob_up:.2%} up, conf={confidence:.2f}",
        }

        self.log_decision(
            f"Ensemble prediction: {direction} (strategy={strategy}, prob_up={prob_up:.2%}, confidence={confidence:.2f})",
            extra={
                'strategy': strategy,
                'individual_predictions': individual_predictions,
                'regime_info': regime_details
            }
        )

        return {"ml_prediction": ml_prediction}
