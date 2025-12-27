"""Model Comparison Script for LLM-TradeBot.

Compares BiLSTM vs Transformer performance on same test set.
Reports accuracy, precision, recall, F1, and inference time.

This comparison is CRITICAL because:
- Research shows BiLSTM often outperforms Transformers for financial time series
- Transformer may capture different patterns (long-range dependencies)
- Inference time comparison critical for live trading (target < 500ms from 07-CONTEXT.md)
"""

import asyncio
import logging
import torch
import time
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from pathlib import Path
from typing import Dict, Tuple

from trading.ml.deep_learning.models.lstm_model import BiLSTMClassifier
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


def load_model(
    model_path: Path,
    model_class,
    hyperparameters: dict,
    device: torch.device
):
    """Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        model_class: BiLSTMClassifier or TransformerClassifier
        hyperparameters: Model hyperparameters from checkpoint
        device: torch device
        
    Returns:
        Loaded model in eval mode
    """
    model = model_class(**hyperparameters).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def fetch_test_data(
    sequence_length: int = 50,
    limit: int = 10000
) -> Tuple[np.ndarray, np.ndarray, DataPreprocessor, list]:
    """Fetch and prepare test data.
    
    Uses same data generation as training scripts for consistency.
    
    Args:
        sequence_length: Sequence length (default: 50)
        limit: Number of candles (default: 10000)
        
    Returns:
        Tuple of (X_test, y_test, preprocessor, feature_columns)
    """
    logger.info(f"Generating test data ({limit} candles)...")
    
    # Generate synthetic OHLCV data (same as training)
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq='5min')
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.cumsum(np.random.randn(limit)) + 50000,
        'high': np.cumsum(np.random.randn(limit)) + 50100,
        'low': np.cumsum(np.random.randn(limit)) + 49900,
        'close': np.cumsum(np.random.randn(limit)) + 50000,
        'volume': np.abs(np.random.randn(limit) * 1000)
    })
    
    # Engineer features
    engineer = FeatureEngineer(
        windows=[5, 10, 20, 50],
        include_target=True,
        target_horizon=5
    )
    
    df_features = engineer.transform(df)
    feature_columns = engineer.get_feature_names(exclude_target=True)
    
    # Create preprocessor and sequences
    preprocessor = DataPreprocessor(sequence_length=sequence_length)
    sequences, labels = preprocessor.create_sequences(
        df_features,
        feature_columns=feature_columns,
        label_column='target_binary',
        fit_scaler=True
    )
    
    # Use only test split (last 15%)
    _, _, (X_test, y_test) = preprocessor.split_time_series(
        sequences, labels,
        validation_size=0.15,
        test_size=0.15
    )
    
    logger.info(f"Test set: {len(X_test)} samples")
    
    return X_test, y_test, preprocessor, feature_columns


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device
) -> Dict:
    """Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test sequences
        y_test: Test labels
        device: torch device
        
    Returns:
        Dictionary with metrics (accuracy, precision, recall, f1, inference_time, confusion_matrix)
    """
    model.eval()
    
    # Convert to tensors
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (torch.sigmoid(outputs) > 0.5).float()
    
    end_time = time.time()
    total_time = (end_time - start_time) * 1000  # Convert to ms
    avg_time_per_prediction = total_time / len(X_test)
    
    # Convert to numpy
    predictions_np = predictions.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_np, predictions_np)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_np,
        predictions_np,
        average='binary',
        zero_division=0
    )
    cm = confusion_matrix(y_test_np, predictions_np)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'inference_time_ms': avg_time_per_prediction,
        'confusion_matrix': cm,
    }


async def main():
    """Main comparison function."""
    logger.info("=" * 80)
    logger.info("MODEL COMPARISON: BiLSTM vs Transformer")
    logger.info("=" * 80)
    
    # Force CPU device
    torch.set_default_device('cpu')
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Model paths
    lstm_path = Path('trading/ml/models/deep_learning/lstm_model.pth')
    transformer_path = Path('trading/ml/models/deep_learning/transformer_model.pth')
    
    # Check if models exist
    if not lstm_path.exists():
        logger.error(f"LSTM model not found at {lstm_path}")
        logger.error("Please run: python trading/ml/deep_learning/training/train_lstm.py")
        return
    
    if not transformer_path.exists():
        logger.error(f"Transformer model not found at {transformer_path}")
        logger.error("Please run: python trading/ml/deep_learning/training/train_transformer.py")
        return
    
    logger.info(f"✓ Found LSTM model: {lstm_path}")
    logger.info(f"✓ Found Transformer model: {transformer_path}")
    
    # Load models
    logger.info("\nLoading models...")
    
    lstm_checkpoint = torch.load(lstm_path, map_location=device)
    lstm_hyperparams = {
        'input_size': lstm_checkpoint['hyperparameters']['input_size'],
        'hidden_size': lstm_checkpoint['hyperparameters']['hidden_size'],
        'num_layers': lstm_checkpoint['hyperparameters']['num_layers'],
        'dropout': lstm_checkpoint['hyperparameters']['dropout'],
    }
    lstm_model, _ = load_model(lstm_path, BiLSTMClassifier, lstm_hyperparams, device)
    logger.info(f"✓ Loaded BiLSTM model")
    
    transformer_checkpoint = torch.load(transformer_path, map_location=device)
    transformer_hyperparams = {
        'input_size': transformer_checkpoint['hyperparameters']['input_size'],
        'd_model': transformer_checkpoint['hyperparameters']['d_model'],
        'nhead': transformer_checkpoint['hyperparameters']['nhead'],
        'num_layers': transformer_checkpoint['hyperparameters']['num_layers'],
        'dropout': transformer_checkpoint['hyperparameters']['dropout'],
    }
    transformer_model, _ = load_model(transformer_path, TransformerClassifier, transformer_hyperparams, device)
    logger.info(f"✓ Loaded Transformer model")
    
    # Fetch test data
    logger.info("\nPreparing test data...")
    X_test, y_test, preprocessor, feature_columns = fetch_test_data(
        sequence_length=50,
        limit=10000
    )
    
    # Evaluate BiLSTM
    logger.info("\nEvaluating BiLSTM...")
    lstm_metrics = evaluate_model(lstm_model, X_test, y_test, device)
    
    # Evaluate Transformer
    logger.info("Evaluating Transformer...")
    transformer_metrics = evaluate_model(transformer_model, X_test, y_test, device)
    
    # Print comparison results
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 80)
    
    logger.info("\nBiLSTM:")
    logger.info(f"  - Accuracy:       {lstm_metrics['accuracy']:.4f}")
    logger.info(f"  - Precision:      {lstm_metrics['precision']:.4f}")
    logger.info(f"  - Recall:         {lstm_metrics['recall']:.4f}")
    logger.info(f"  - F1 Score:       {lstm_metrics['f1']:.4f}")
    logger.info(f"  - Inference Time: {lstm_metrics['inference_time_ms']:.2f} ms/prediction")
    logger.info(f"  - Confusion Matrix:")
    logger.info(f"    {lstm_metrics['confusion_matrix']}")
    
    logger.info("\nTransformer:")
    logger.info(f"  - Accuracy:       {transformer_metrics['accuracy']:.4f}")
    logger.info(f"  - Precision:      {transformer_metrics['precision']:.4f}")
    logger.info(f"  - Recall:         {transformer_metrics['recall']:.4f}")
    logger.info(f"  - F1 Score:       {transformer_metrics['f1']:.4f}")
    logger.info(f"  - Inference Time: {transformer_metrics['inference_time_ms']:.2f} ms/prediction")
    logger.info(f"  - Confusion Matrix:")
    logger.info(f"    {transformer_metrics['confusion_matrix']}")
    
    # Determine winner
    logger.info("\n" + "=" * 80)
    if lstm_metrics['accuracy'] > transformer_metrics['accuracy']:
        winner = "BiLSTM"
        diff = lstm_metrics['accuracy'] - transformer_metrics['accuracy']
    elif transformer_metrics['accuracy'] > lstm_metrics['accuracy']:
        winner = "Transformer"
        diff = transformer_metrics['accuracy'] - lstm_metrics['accuracy']
    else:
        winner = "TIE"
        diff = 0.0
    
    if winner == "TIE":
        logger.info("Result: TIE (same accuracy)")
    else:
        logger.info(f"Winner: {winner} (by {diff:.4f} accuracy)")
    
    # Check inference time target
    logger.info("\nInference Time Analysis:")
    target_ms = 500  # Target from 07-CONTEXT.md
    
    for name, metrics in [("BiLSTM", lstm_metrics), ("Transformer", transformer_metrics)]:
        if metrics['inference_time_ms'] < target_ms:
            logger.info(f"  ✓ {name}: {metrics['inference_time_ms']:.2f} ms < {target_ms} ms (PASS)")
        else:
            logger.info(f"  ✗ {name}: {metrics['inference_time_ms']:.2f} ms > {target_ms} ms (FAIL)")
    
    logger.info("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
