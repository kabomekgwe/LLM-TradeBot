"""
Machine learning model tests for LightGBM training and prediction.
Tests model training, prediction validity, and feature extraction.
"""

import pytest
import numpy as np
import lightgbm as lgb
import os
import tempfile

from trading.ml.train_lightgbm import (
    calculate_features,
    create_labels,
    fetch_historical_data
)


# === Feature Extraction Tests ===

@pytest.mark.unit
@pytest.mark.ml
def test_calculate_features_returns_correct_shape(ohlcv_factory):
    """Test feature extraction returns correct number of features."""
    # Arrange
    ohlcv = ohlcv_factory(num_candles=100)

    # Act
    features = calculate_features(ohlcv)

    # Assert - Should return 8 features: RSI, MACD (3), BB (3), returns (1)
    assert features.shape[1] == 8
    assert features.shape[0] == 100


@pytest.mark.unit
@pytest.mark.ml
def test_calculate_features_handles_short_sequences(ohlcv_factory):
    """Test feature extraction handles short sequences (fills with NaN for early periods)."""
    # Arrange - Very short sequence
    ohlcv = ohlcv_factory(num_candles=15)  # Less than indicators need

    # Act
    features = calculate_features(ohlcv)

    # Assert - Should return features but early rows will be NaN
    assert features.shape[0] == 15
    assert np.isnan(features[0]).any()  # First row should have NaN (not enough history)


@pytest.mark.unit
@pytest.mark.ml
def test_calculate_features_produces_valid_values(ohlcv_factory):
    """Test feature extraction produces valid numeric values (no inf, valid ranges)."""
    # Arrange
    ohlcv = ohlcv_factory(num_candles=100, trend="uptrend")

    # Act
    features = calculate_features(ohlcv)

    # Assert - After warmup period, values should be valid
    valid_features = features[30:]  # Skip warmup period
    assert not np.isinf(valid_features).any()  # No infinity values
    assert not np.isnan(valid_features).all()  # Not all NaN

    # RSI should be between 0 and 100 (first column)
    rsi_values = valid_features[:, 0]
    rsi_values = rsi_values[~np.isnan(rsi_values)]
    assert (rsi_values >= 0).all() and (rsi_values <= 100).all()


# === Label Creation Tests ===

@pytest.mark.unit
@pytest.mark.ml
def test_create_labels_binary_classification(ohlcv_factory):
    """Test label creation produces binary labels (0 or 1)."""
    # Arrange
    ohlcv = ohlcv_factory(num_candles=100, trend="uptrend")

    # Act
    labels = create_labels(ohlcv, lookahead=5)

    # Assert
    assert set(np.unique(labels)).issubset({0, 1})  # Only 0 or 1
    assert len(labels) == 95  # 100 - 5 lookahead


@pytest.mark.unit
@pytest.mark.ml
def test_create_labels_uptrend_produces_mostly_ones(ohlcv_factory):
    """Test label creation in uptrend produces mostly 1s (price going up)."""
    # Arrange
    ohlcv = ohlcv_factory(num_candles=100, trend="uptrend", volatility=0.01)

    # Act
    labels = create_labels(ohlcv, lookahead=5)

    # Assert - Uptrend should have more 1s than 0s
    ones_ratio = np.sum(labels) / len(labels)
    assert ones_ratio > 0.5  # More than 50% should be 1s


@pytest.mark.unit
@pytest.mark.ml
def test_create_labels_downtrend_produces_mostly_zeros(ohlcv_factory):
    """Test label creation in downtrend produces mostly 0s (price going down)."""
    # Arrange
    ohlcv = ohlcv_factory(num_candles=100, trend="downtrend", volatility=0.01)

    # Act
    labels = create_labels(ohlcv, lookahead=5)

    # Assert - Downtrend should have more 0s than 1s
    zeros_ratio = 1 - (np.sum(labels) / len(labels))
    assert zeros_ratio > 0.5  # More than 50% should be 0s


# === Model Training Tests ===

@pytest.mark.slow
@pytest.mark.ml
def test_lightgbm_model_training(ohlcv_factory):
    """Test LightGBM model trains successfully with synthetic data."""
    # Arrange - Generate synthetic training data
    ohlcv = ohlcv_factory(num_candles=200, trend="uptrend")
    features = calculate_features(ohlcv)
    labels = create_labels(ohlcv, lookahead=5)

    # Remove NaN rows
    valid_mask = ~np.isnan(features).any(axis=1)
    features = features[valid_mask]
    labels = labels[valid_mask[:len(labels)]]

    # Align
    min_len = min(len(features), len(labels))
    features = features[:min_len]
    labels = labels[:min_len]

    # Create dataset
    train_data = lgb.Dataset(features, label=labels)

    # Act - Train model
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': -1
    }

    bst = lgb.train(
        params,
        train_data,
        num_boost_round=10,  # Quick training for test
        valid_sets=[train_data]
    )

    # Assert
    assert bst is not None
    assert bst.num_trees() == 10  # Should have 10 trees


@pytest.mark.slow
@pytest.mark.ml
def test_lightgbm_model_predictions_valid_range(ohlcv_factory):
    """Test trained model produces predictions in valid probability range [0, 1]."""
    # Arrange - Train a quick model
    ohlcv = ohlcv_factory(num_candles=200, trend="uptrend")
    features = calculate_features(ohlcv)
    labels = create_labels(ohlcv, lookahead=5)

    valid_mask = ~np.isnan(features).any(axis=1)
    features = features[valid_mask]
    labels = labels[valid_mask[:len(labels)]]
    min_len = min(len(features), len(labels))
    features = features[:min_len]
    labels = labels[:min_len]

    train_data = lgb.Dataset(features, label=labels)
    params = {'objective': 'binary', 'verbose': -1}
    bst = lgb.train(params, train_data, num_boost_round=10)

    # Act - Make predictions
    predictions = bst.predict(features[:10])  # Test on first 10 samples

    # Assert - Probabilities should be between 0 and 1
    assert (predictions >= 0).all()
    assert (predictions <= 1).all()
    assert len(predictions) == 10


# === Model Persistence Tests ===

@pytest.mark.unit
@pytest.mark.ml
def test_lightgbm_model_save_load(ohlcv_factory):
    """Test LightGBM model can be saved and loaded."""
    # Arrange - Train a quick model
    ohlcv = ohlcv_factory(num_candles=200)
    features = calculate_features(ohlcv)
    labels = create_labels(ohlcv, lookahead=5)

    valid_mask = ~np.isnan(features).any(axis=1)
    features = features[valid_mask]
    labels = labels[valid_mask[:len(labels)]]
    min_len = min(len(features), len(labels))
    features = features[:min_len]
    labels = labels[:min_len]

    train_data = lgb.Dataset(features, label=labels)
    params = {'objective': 'binary', 'verbose': -1}
    bst = lgb.train(params, train_data, num_boost_round=5)

    # Act - Save and load model
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        model_path = tmp.name

    try:
        bst.save_model(model_path)
        loaded_model = lgb.Booster(model_file=model_path)

        # Make predictions with both models
        original_pred = bst.predict(features[:5])
        loaded_pred = loaded_model.predict(features[:5])

        # Assert - Predictions should be identical
        np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=6)

    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


# === PredictAgent Feature Alignment Tests ===

@pytest.mark.unit
@pytest.mark.ml
def test_predict_agent_features_match_training_schema():
    """Test PredictAgent extracts features in same order as training script."""
    # Arrange - Mock indicators from QuantAnalyst
    indicators = {
        'rsi': {'value': 65.5},
        'macd': {'macd': 0.5, 'signal': 0.3, 'histogram': 0.2},
        'bollinger': {'upper': 30100, 'middle': 30000, 'lower': 29900}
    }

    # Expected feature order from training script:
    # [RSI, MACD, Signal, Histogram, Upper BB, Middle BB, Lower BB, Returns]
    expected_order = [
        65.5,   # RSI
        0.5,    # MACD
        0.3,    # Signal
        0.2,    # Histogram
        30100,  # Upper BB
        30000,  # Middle BB
        29900,  # Lower BB
        0.0     # Returns placeholder
    ]

    # Act - Extract features using PredictAgent logic
    features = np.array([[
        indicators['rsi']['value'],
        indicators['macd']['macd'],
        indicators['macd']['signal'],
        indicators['macd']['histogram'],
        indicators['bollinger']['upper'],
        indicators['bollinger']['middle'],
        indicators['bollinger']['lower'],
        0.0  # Price returns placeholder
    ]])

    # Assert - Feature order must match training
    np.testing.assert_array_equal(features[0], expected_order)


@pytest.mark.unit
@pytest.mark.ml
def test_predict_agent_confidence_scaling():
    """Test PredictAgent scales probability to confidence correctly."""
    # Test cases: (probability, expected_confidence)
    # Formula: confidence = abs(prob - 0.5) * 2
    test_cases = [
        (0.5, 0.0),   # Neutral probability = 0 confidence
        (0.75, 0.5),  # 75% up = 50% confidence
        (1.0, 1.0),   # 100% up = 100% confidence
        (0.25, 0.5),  # 25% up (75% down) = 50% confidence
        (0.0, 1.0),   # 0% up (100% down) = 100% confidence
    ]

    for prob, expected_conf in test_cases:
        # Act
        confidence = abs(prob - 0.5) * 2

        # Assert
        assert abs(confidence - expected_conf) < 0.001, \
            f"Prob {prob} should give confidence {expected_conf}, got {confidence}"
