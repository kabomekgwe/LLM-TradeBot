"""
Train ensemble model using Phase 5's 86 features and regime detection.

This script:
1. Fetches historical OHLCV data
2. Engineers 86 features using Phase 5's feature pipeline
3. Detects volatility regimes
4. Trains ensemble (LightGBM, XGBoost, Random Forest)
5. Benchmarks ensemble vs single LightGBM
6. Saves models using security-hardened persistence
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb

from trading.config import TradingConfig
from trading.providers.factory import create_provider
from trading.ml.feature_engineering import FeatureEngineer
from trading.ml.ensemble.regime_aware_ensemble import RegimeAwareEnsemble
from trading.ml.ensemble.persistence import EnsemblePersistence
from trading.logging_config import get_logger

logger = get_logger(__name__)


async def fetch_historical_data(exchange, symbol: str, timeframe: str, limit: int = 10000):
    """
    Fetch historical OHLCV data from exchange.

    Args:
        exchange: Exchange instance
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Timeframe (e.g., '5m')
        limit: Number of candles to fetch

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Fetching {limit} candles for {symbol} {timeframe}")

    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    df = pd.DataFrame(
        ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

    logger.info(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    return df


def create_labels(df: pd.DataFrame, lookahead: int = 5) -> np.ndarray:
    """
    Create binary labels for price direction.

    Args:
        df: DataFrame with 'close' column
        lookahead: Number of candles to look ahead

    Returns:
        Binary labels (1 = price up, 0 = price down)
    """
    future_close = df['close'].shift(-lookahead)
    current_close = df['close']

    # 1 if price goes up, 0 if price goes down
    labels = (future_close > current_close).astype(int)

    return labels.values


async def main():
    """Main training pipeline."""
    logger.info("=" * 80)
    logger.info("ENSEMBLE MODEL TRAINING")
    logger.info("=" * 80)

    # Load config
    config = TradingConfig.from_env()

    # Create exchange provider
    exchange = await create_provider(config, demo=True)
    logger.info(f"Connected to {exchange.id}")

    # Fetch historical data
    symbol = 'BTC/USDT'
    timeframe = '5m'
    df = await fetch_historical_data(exchange, symbol, timeframe, limit=10000)

    logger.info(f"Data shape: {df.shape}")

    # Engineer features
    logger.info("Engineering features...")
    feature_engineer = FeatureEngineer(include_target=False)

    # Transform data - this will populate feature_categories
    df_features = feature_engineer.transform(df)

    logger.info(f"Features engineered: {df_features.shape}")

    # Get feature names
    feature_names = feature_engineer.get_feature_names(exclude_target=True)
    logger.info(f"Total features: {len(feature_names)}")

    # Create labels (5 candles ahead = ~25 minutes on 5m timeframe)
    logger.info("Creating labels...")
    labels = create_labels(df, lookahead=5)

    # Align features and labels (remove NaN rows)
    valid_rows = ~(df_features[feature_names].isna().any(axis=1) | pd.isna(labels))
    X = df_features[feature_names][valid_rows].values
    y = labels[valid_rows]

    logger.info(f"Valid samples: {len(X)} (removed {len(df) - len(X)} NaN rows)")

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    logger.info(f"Train label distribution: {np.bincount(y_train)}")
    logger.info(f"Test label distribution: {np.bincount(y_test)}")

    # ========================================================================
    # BASELINE: Single LightGBM Model
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("BASELINE: Single LightGBM Model")
    logger.info("=" * 80)

    lgbm_baseline = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )

    logger.info("Training baseline LightGBM...")
    lgbm_baseline.fit(X_train, y_train)

    baseline_pred = lgbm_baseline.predict(X_test)
    baseline_pred_proba = lgbm_baseline.predict_proba(X_test)[:, 1]

    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    baseline_precision = precision_score(y_test, baseline_pred, zero_division=0)
    baseline_recall = recall_score(y_test, baseline_pred, zero_division=0)
    baseline_f1 = f1_score(y_test, baseline_pred, zero_division=0)
    baseline_auc = roc_auc_score(y_test, baseline_pred_proba)

    logger.info(f"Baseline LightGBM Performance:")
    logger.info(f"  Accuracy:  {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    logger.info(f"  Precision: {baseline_precision:.4f}")
    logger.info(f"  Recall:    {baseline_recall:.4f}")
    logger.info(f"  F1 Score:  {baseline_f1:.4f}")
    logger.info(f"  AUC-ROC:   {baseline_auc:.4f}")

    # ========================================================================
    # ENSEMBLE: RegimeAwareEnsemble
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("ENSEMBLE: RegimeAwareEnsemble (LightGBM + XGBoost + RF)")
    logger.info("=" * 80)

    ensemble = RegimeAwareEnsemble(n_estimators=100, random_state=42)

    logger.info("Training ensemble (this may take a few minutes)...")
    ensemble.fit(X_train, y_train)

    logger.info("Making ensemble predictions...")

    # Extract regime info from test set
    df_test_features = df_features[valid_rows].iloc[-len(X_test):]

    # Check if regime features exist
    regime_columns = ['regime_prob_0', 'regime_prob_1', 'current_regime', 'is_low_volatility']
    has_regime_features = all(col in df_test_features.columns for col in regime_columns)

    if has_regime_features:
        logger.info("Using regime-aware predictions")
        ensemble_predictions = []
        ensemble_probabilities = []

        for i in range(len(X_test)):
            regime_info = {
                'current_regime': int(df_test_features.iloc[i]['current_regime']),
                'is_low_volatility': int(df_test_features.iloc[i]['is_low_volatility']),
                'regime_prob_0': float(df_test_features.iloc[i]['regime_prob_0']),
                'regime_prob_1': float(df_test_features.iloc[i]['regime_prob_1'])
            }

            pred, metadata = ensemble.predict_with_regime(X_test[i:i+1], regime_info)
            ensemble_predictions.append(pred[0])
            ensemble_probabilities.append(metadata['ensemble_probability'])

        ensemble_predictions = np.array(ensemble_predictions)
        ensemble_probabilities = np.array(ensemble_probabilities)
    else:
        logger.warning("Regime features not found, using default voting strategy")
        ensemble_predictions = ensemble.predict(X_test)
        ensemble_probabilities = ensemble.predict_proba(X_test)[:, 1]

    # Calculate ensemble metrics
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
    ensemble_precision = precision_score(y_test, ensemble_predictions, zero_division=0)
    ensemble_recall = recall_score(y_test, ensemble_predictions, zero_division=0)
    ensemble_f1 = f1_score(y_test, ensemble_predictions, zero_division=0)
    ensemble_auc = roc_auc_score(y_test, ensemble_probabilities)

    logger.info(f"Ensemble Performance:")
    logger.info(f"  Accuracy:  {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
    logger.info(f"  Precision: {ensemble_precision:.4f}")
    logger.info(f"  Recall:    {ensemble_recall:.4f}")
    logger.info(f"  F1 Score:  {ensemble_f1:.4f}")
    logger.info(f"  AUC-ROC:   {ensemble_auc:.4f}")

    # ========================================================================
    # BENCHMARK: Compare Ensemble vs Baseline
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("BENCHMARK: Ensemble vs Baseline")
    logger.info("=" * 80)

    improvement_accuracy = ((ensemble_accuracy - baseline_accuracy) / baseline_accuracy) * 100
    improvement_auc = ((ensemble_auc - baseline_auc) / baseline_auc) * 100

    logger.info(f"Accuracy improvement: {improvement_accuracy:+.1f}%")
    logger.info(f"AUC-ROC improvement:  {improvement_auc:+.1f}%")

    if improvement_accuracy >= 10:
        logger.info("✅ Ensemble meets 10-20% accuracy improvement target!")
    else:
        logger.warning(f"⚠️  Ensemble improvement ({improvement_accuracy:.1f}%) below 10% target")

    # ========================================================================
    # FEATURE IMPORTANCE
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("TOP 10 FEATURES (Aggregated Across All Models)")
    logger.info("=" * 80)

    feature_importance = ensemble.get_feature_importance(feature_names)

    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
        logger.info(f"{i:2d}. {feature:40s} {importance:.6f}")

    # ========================================================================
    # SAVE MODELS
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("SAVING ENSEMBLE MODELS")
    logger.info("=" * 80)

    persistence = EnsemblePersistence(model_dir='trading/ml/models/ensemble')

    metadata = {
        'symbol': symbol,
        'timeframe': timeframe,
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'trained_date': datetime.now().isoformat(),
        'baseline_accuracy': float(baseline_accuracy),
        'ensemble_accuracy': float(ensemble_accuracy),
        'improvement_pct': float(improvement_accuracy),
        'ensemble_auc': float(ensemble_auc),
        'baseline_auc': float(baseline_auc),
    }

    models = ensemble.model_registry.get_models()

    logger.info("Saving models with security-hardened persistence...")
    persistence.save_models(models, metadata)

    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Models saved to: {persistence.model_dir}")
    logger.info(f"✅ XGBoost model: xgboost_model.json (JSON format - SECURE)")
    logger.info(f"✅ LightGBM model: lightgbm_model.txt (text format - SECURE)")
    logger.info(f"✅ Random Forest model: random_forest_model.joblib (sklearn standard)")
    logger.info(f"✅ Metadata: metadata.json (JSON format - SECURE)")

    # Close exchange
    await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
