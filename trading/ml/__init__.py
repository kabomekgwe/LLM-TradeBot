"""Machine Learning Module - Advanced price prediction ensemble.

Multi-model ensemble combining:
- LightGBM (gradient boosting)
- XGBoost (extreme gradient boosting)
- Random Forest (tree ensemble)
- LSTM (deep learning time series)

Features:
- 50+ engineered features (technical, statistical, volatility)
- Automated model training and retraining
- Ensemble voting with configurable weights
- Feature importance analysis
- Out-of-sample validation

Components:
- FeatureEngineer: Generate ML features from OHLCV data
- EnsemblePredictor: Combine multiple models with voting
- ModelTrainer: Train and persist models
- Individual model implementations (LightGBM, XGBoost, RF, LSTM)

Example Usage:
    ```python
    from trading.ml import EnsemblePredictor, FeatureEngineer

    # Create feature engineer
    feature_engineer = FeatureEngineer()

    # Load historical data
    df = load_ohlcv_data("BTC/USDT", timeframe="1h", limit=1000)

    # Engineer features
    features = feature_engineer.transform(df)

    # Create ensemble predictor
    predictor = EnsemblePredictor(
        models=['lightgbm', 'xgboost', 'random_forest', 'lstm'],
        weights={'lightgbm': 0.3, 'xgboost': 0.3, 'random_forest': 0.2, 'lstm': 0.2}
    )

    # Train models
    predictor.train(features, target_column='future_return')

    # Make predictions
    prediction = predictor.predict(latest_features)
    # Returns: probability of price increase (0-1)

    # Get feature importance
    importance = predictor.get_feature_importance()
    ```

Configuration:
    Set in TradingConfig or environment:
    ```bash
    TRADING_ENABLE_ML_PREDICTIONS=true
    ML_MODEL_DIR=models/
    ML_RETRAIN_INTERVAL_HOURS=24
    ML_ENSEMBLE_WEIGHTS='{"lightgbm": 0.3, "xgboost": 0.3, "rf": 0.2, "lstm": 0.2}'
    ```

Model Persistence:
    Trained models saved to: models/{symbol}/{model_name}.pkl
    Feature metadata saved to: models/{symbol}/feature_metadata.json
"""

from .feature_engineering import FeatureEngineer
from .ensemble import EnsemblePredictor, ModelType
from .training import ModelTrainer

__all__ = [
    "FeatureEngineer",
    "EnsemblePredictor",
    "ModelTrainer",
    "ModelType",
]

__version__ = "1.0.0"
