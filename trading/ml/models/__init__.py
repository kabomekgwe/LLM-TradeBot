"""ML Models - Individual model implementations.

Provides wrappers for various ML algorithms:
- LightGBM: Fast gradient boosting
- XGBoost: Extreme gradient boosting
- Random Forest: Tree ensemble (sklearn)
- LSTM: Deep learning time series

Each model provides a consistent interface:
- fit(X, y): Train the model
- predict(X): Make predictions
- predict_proba(X): Get prediction probabilities
- get_feature_importance(): Get feature importances
"""

from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel
from .lstm_model import LSTMModel

__all__ = [
    "LightGBMModel",
    "XGBoostModel",
    "LSTMModel",
]
