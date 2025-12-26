"""PredictAgent - ML-based price forecasting.

This is Agent #3 in the 8-agent system.
Uses LightGBM ML models to forecast price direction.
"""

from typing import Any
import lightgbm as lgb
import numpy as np
import os

from .base_agent import BaseAgent


class PredictAgent(BaseAgent):
    """Prediction agent - ML price forecasting.

    Uses machine learning (LightGBM) to predict 30-minute price direction
    based on historical features from QuantAnalystAgent.
    """

    def __init__(self, provider, config):
        """Initialize PredictAgent with pre-trained LightGBM model.

        Args:
            provider: Exchange provider instance
            config: TradingConfig instance
        """
        super().__init__(provider, config)

        # Load pre-trained LightGBM model
        model_path = 'trading/ml/models/lgbm_predictor.txt'
        if os.path.exists(model_path):
            self.model = lgb.Booster(model_file=model_path)
            self.logger.info(f"Loaded LightGBM model from {model_path}")
        else:
            self.model = None
            self.logger.warning(f"No trained model found at {model_path}. Run train_lightgbm.py first.")

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate ML price prediction using LightGBM model.

        Args:
            context: Must contain "quant_analyst" with indicators from QuantAnalystAgent

        Returns:
            Context updated with "ml_prediction" containing:
                - "direction": "up", "down", or "neutral"
                - "confidence": Float 0-1
                - "probability_up": Raw probability for transparency

        Example:
            >>> result = await agent.execute(context)
            >>> result["ml_prediction"]["direction"]
            'up'
            >>> result["ml_prediction"]["confidence"]
            0.73
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

        # Fallback if no model trained
        if not self.model:
            self.log_decision("No trained model available, returning neutral")
            return {
                "ml_prediction": {
                    "direction": "neutral",
                    "confidence": 0.0,
                    "reason": "No trained model available - run train_lightgbm.py",
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

        # Extract features (must match training feature order)
        try:
            features = np.array([[
                indicators['rsi']['value'],
                indicators['macd']['macd'],
                indicators['macd']['signal'],
                indicators['macd']['histogram'],
                indicators['bollinger']['upper'],
                indicators['bollinger']['middle'],
                indicators['bollinger']['lower'],
                0.0  # Price returns placeholder (calculate if needed)
            ]])
        except (KeyError, TypeError) as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return {
                "ml_prediction": {
                    "direction": "neutral",
                    "confidence": 0.0,
                    "reason": f"Feature extraction error: {e}",
                }
            }

        # Predict probability of price going up
        prob_up = self.model.predict(features, num_iteration=self.model.best_iteration)[0]

        # Convert to direction and confidence
        # prob_up in [0, 1], where >0.5 means price will go up
        direction = "up" if prob_up > 0.5 else "down"
        confidence = abs(prob_up - 0.5) * 2  # Scale to [0, 1]: confidence in prediction

        ml_prediction = {
            "direction": direction,
            "confidence": confidence,
            "probability_up": prob_up,
            "reason": f"ML prediction: {prob_up:.2%} up, conf={confidence:.2f}",
        }

        self.log_decision(
            f"ML prediction: {direction} (prob_up={prob_up:.2%}, confidence={confidence:.2f})"
        )

        return {"ml_prediction": ml_prediction}
