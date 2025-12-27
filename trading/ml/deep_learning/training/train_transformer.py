"""Transformer Training Script with Validation.

Comprehensive training pipeline for Transformer architecture:
1. Fetches 10k historical candles from exchange
2. Engineers 86 features using Phase 5 pipeline
3. Creates sequences with DataPreprocessor (reused from 07-01)
4. Trains Transformer with causal masking
5. Implements early stopping
6. Saves best model checkpoint

Training strategies from research (07-RESEARCH.md):
- AdamW optimizer (better than Adam for weight decay)
- ReduceLROnPlateau scheduler (automatic LR adjustment)
- BCEWithLogitsLoss for binary classification
- Early stopping (patience=10)
- NO shuffle in DataLoader (prevents data leakage - Pitfall 1)
- Sequence length 50 (prevents O(n²) memory explosion - Pitfall 2)
- CPU device (MacBook compatible)

CRITICAL DIFFERENCES FROM LSTM:
- Uses TransformerClassifier instead of BiLSTMClassifier
- Causal masking handled inside model.forward()
- Same hyperparameters (d_model=128, nhead=8, num_layers=2)
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np

from trading.ml.deep_learning.models.transformer_model import TransformerClassifier
from trading.ml.deep_learning.data.dataset import TimeSeriesDataset
from trading.ml.deep_learning.data.preprocessing import DataPreprocessor
from trading.ml.feature_engineering import FeatureEngineer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_and_prepare_data(
    symbol: str = 'BTC/USDT',
    timeframe: str = '5m',
    limit: int = 10000,
    sequence_length: int = 50
) -> Tuple[DataPreprocessor, Tuple, Tuple, Tuple, list]:
    """Fetch historical data and prepare for Transformer training.

    CRITICAL: Same data pipeline as LSTM (reuses DataPreprocessor from 07-01).

    Args:
        symbol: Trading pair (default: BTC/USDT)
        timeframe: Candle timeframe (default: 5m)
        limit: Number of candles to fetch (default: 10000)
        sequence_length: Sequence length (50 prevents O(n²) explosion - Pitfall 2)

    Returns:
        Tuple of (preprocessor, train_data, val_data, test_data, feature_columns)
    """
    logger.info(f"Fetching {limit} candles for {symbol} {timeframe}")
    logger.warning("Using synthetic data for demo - replace with real CCXT fetch in production")

    # Generate synthetic OHLCV data
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq='5min')
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.cumsum(np.random.randn(limit)) + 50000,
        'high': np.cumsum(np.random.randn(limit)) + 50100,
        'low': np.cumsum(np.random.randn(limit)) + 49900,
        'close': np.cumsum(np.random.randn(limit)) + 50000,
        'volume': np.abs(np.random.randn(limit) * 1000)
    })

    logger.info(f"Generated synthetic OHLCV data: {len(df)} candles")

    # Engineer features using Phase 5 pipeline
    logger.info("Engineering features using Phase 5 pipeline...")
    engineer = FeatureEngineer(
        windows=[5, 10, 20, 50],
        include_target=True,
        target_horizon=5
    )

    df_features = engineer.transform(df)
    logger.info(f"Engineered {len(df_features.columns)} features from {len(df_features)} samples")

    feature_columns = engineer.get_feature_names(exclude_target=True)
    logger.info(f"Using {len(feature_columns)} features for training")

    preprocessor = DataPreprocessor(sequence_length=sequence_length)

    logger.info("Creating sequences with sliding window...")
    sequences, labels = preprocessor.create_sequences(
        df_features,
        feature_columns=feature_columns,
        label_column='target_binary',
        fit_scaler=True
    )
    logger.info(f"Created {len(sequences)} sequences")

    logger.info("Splitting into train/validation/test (chronological)...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_time_series(
        sequences, labels,
        validation_size=0.15,
        test_size=0.15
    )

    logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Class balance - Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")

    return preprocessor, (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_columns


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train Transformer for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, (sequences, labels) in enumerate(train_loader):
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate Transformer on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def main():
    """Main training loop for Transformer."""
    torch.manual_seed(42)
    np.random.seed(42)

    torch.set_default_device('cpu')
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # Hyperparameters from 07-RESEARCH.md
    SEQUENCE_LENGTH = 50
    BATCH_SIZE = 32
    D_MODEL = 128
    NHEAD = 8
    NUM_LAYERS = 2
    DROPOUT = 0.2
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 100
    PATIENCE = 10

    logger.info("=" * 80)
    logger.info("FETCHING AND PREPARING DATA")
    logger.info("=" * 80)

    result_tuple = fetch_and_prepare_data(
        symbol='BTC/USDT',
        timeframe='5m',
        limit=10000,
        sequence_length=SEQUENCE_LENGTH
    )

    preprocessor = result_tuple[0]
    X_train, y_train = result_tuple[1]
    X_val, y_val = result_tuple[2]
    X_test, y_test = result_tuple[3]
    feature_columns = result_tuple[4]

    num_features = len(feature_columns)
    logger.info(f"Input features: {num_features}")

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    logger.info("=" * 80)
    logger.info("CREATING TRANSFORMER MODEL")
    logger.info("=" * 80)

    model = TransformerClassifier(
        input_size=num_features,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: TransformerClassifier")
    logger.info(f"Parameters: {total_params:,}")
    logger.info(f"Architecture: d_model={D_MODEL}, nhead={NHEAD}, num_layers={NUM_LAYERS}")
    logger.info(f"Causal masking: ENABLED (prevents future data leakage - Pitfall 4)")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    logger.info("=" * 80)
    logger.info("TRAINING")
    logger.info("=" * 80)

    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    patience_counter = 0
    best_model_path = Path('trading/ml/models/deep_learning/transformer_model.pth')
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'hyperparameters': {
                    'input_size': num_features,
                    'd_model': D_MODEL,
                    'nhead': NHEAD,
                    'num_layers': NUM_LAYERS,
                    'dropout': DROPOUT,
                    'sequence_length': SEQUENCE_LENGTH,
                },
                'feature_columns': feature_columns,
                'scaler_params': preprocessor.get_scaler_params(),
            }, best_model_path)

            logger.info(f"✓ Best model saved (Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement ({patience_counter}/{PATIENCE})")

            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    logger.info("=" * 80)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 80)

    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_accuracy = validate(model, test_loader, criterion, device)

    logger.info(f"Best Validation - Loss: {best_val_loss:.4f}, Accuracy: {best_val_accuracy:.4f}")
    logger.info(f"Test Set - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

    if test_accuracy > 0.55:
        logger.info("✓ SUCCESS: Test accuracy > 55% (better than random 50%)")
    else:
        logger.warning(f"Test accuracy {test_accuracy:.4f} <= 0.55 - may need more training/tuning")

    logger.info(f"Model saved to: {best_model_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
