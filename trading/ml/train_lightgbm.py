"""
LightGBM Model Training Script
Trains price direction prediction model using historical OHLCV data and technical indicators.
"""

import asyncio
import numpy as np
import lightgbm as lgb
import talib
from trading.providers.factory import ExchangeProviderFactory
from trading.config import TradingConfig
import os


async def fetch_historical_data(symbol: str, limit: int = 1000):
    """Fetch historical OHLCV data for training."""
    config = TradingConfig.from_env()  # Load from environment
    provider = ExchangeProviderFactory.create(config)

    # Fetch historical 5m candles
    ohlcv = await provider.fetch_ohlcv(symbol, timeframe='5m', limit=limit)
    return ohlcv


def calculate_features(ohlcv):
    """Calculate technical indicator features from OHLCV data."""
    close = np.array([float(candle.close) for candle in ohlcv])

    # TA-Lib indicators (same as QuantAnalystAgent)
    rsi = talib.RSI(close, timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # Price momentum features
    returns = np.diff(close) / close[:-1]
    returns = np.append(0, returns)  # Prepend 0 for alignment

    # Combine features
    features = np.column_stack([
        rsi,
        macd,
        macdsignal,
        macdhist,
        upperband,
        middleband,
        lowerband,
        returns
    ])

    return features


def create_labels(ohlcv, lookahead: int = 5):
    """Create binary labels: 1 if price goes up in next N candles, 0 otherwise."""
    close = np.array([float(candle.close) for candle in ohlcv])
    labels = np.zeros(len(close))

    for i in range(len(close) - lookahead):
        future_price = close[i + lookahead]
        current_price = close[i]
        labels[i] = 1 if future_price > current_price else 0

    # Last lookahead candles can't have labels (no future data)
    return labels[:-lookahead]


async def main():
    symbol = "BTC/USDT"
    print(f"Fetching historical data for {symbol}...")

    # Fetch historical data
    ohlcv = await fetch_historical_data(symbol, limit=1000)
    print(f"Fetched {len(ohlcv)} candles")

    # Calculate features
    features = calculate_features(ohlcv)
    labels = create_labels(ohlcv, lookahead=5)

    # Remove NaN rows (early periods lack indicator data)
    valid_mask = ~np.isnan(features).any(axis=1)
    features = features[valid_mask]
    labels = labels[valid_mask[:len(labels)]]  # Labels are shorter due to lookahead

    # Align features and labels
    min_len = min(len(features), len(labels))
    features = features[:min_len]
    labels = labels[:min_len]

    print(f"Training samples: {len(features)} (after removing NaN)")

    # Split train/validation (80/20)
    split_idx = int(len(features) * 0.8)
    X_train, X_val = features[:split_idx], features[split_idx:]
    y_train, y_val = labels[:split_idx], labels[split_idx:]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Training parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }

    print("Training LightGBM model...")
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    # Save model
    os.makedirs('trading/ml/models', exist_ok=True)
    model_path = 'trading/ml/models/lgbm_predictor.txt'
    bst.save_model(model_path)
    print(f"Model saved to {model_path}")
    print(f"Best iteration: {bst.best_iteration}")


if __name__ == "__main__":
    asyncio.run(main())
