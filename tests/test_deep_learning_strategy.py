"""Integration Tests for Deep Learning Strategy.

Test coverage:
- ModelPersistence: Save/load models securely
- DataPreprocessor: Sequence creation, chronological splits
- TimeSeriesDataset: Dataset indexing and length
- DeepLearningStrategy: Prediction format, independent execution
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import tempfile
import shutil

from trading.ml.deep_learning.models.lstm_model import BiLSTMClassifier
from trading.ml.deep_learning.models.transformer_model import TransformerClassifier
from trading.ml.deep_learning.data.dataset import TimeSeriesDataset
from trading.ml.deep_learning.data.preprocessing import DataPreprocessor
from trading.ml.deep_learning.persistence import ModelPersistence
from trading.ml.deep_learning.deep_learning_strategy import DeepLearningStrategy
from trading.config import TradingConfig
from trading.state import TradingState


class TestModelPersistence:
    """Test model persistence with security-hardened state_dict loading."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for models."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir)

    def test_save_load_lstm(self, temp_dir):
        """Test saving and loading BiLSTM model - verify parameters match."""
        # Create model
        model = BiLSTMClassifier(input_size=86, hidden_size=128, num_layers=2, dropout=0.2)

        # Save model
        persistence = ModelPersistence(models_dir=temp_dir)
        success = persistence.save_lstm(model)
        assert success is True

        # Load model
        loaded_model = persistence.load_lstm()
        assert loaded_model is not None

        # Verify parameters match
        original_params = list(model.parameters())
        loaded_params = list(loaded_model.parameters())
        assert len(original_params) == len(loaded_params)

        for orig, loaded in zip(original_params, loaded_params):
            assert torch.allclose(orig, loaded)

    def test_save_load_transformer(self, temp_dir):
        """Test saving and loading Transformer model."""
        # Create model
        model = TransformerClassifier(input_size=86, d_model=128, nhead=8, num_layers=2, dropout=0.2)

        # Save model
        persistence = ModelPersistence(models_dir=temp_dir)
        success = persistence.save_transformer(model)
        assert success is True

        # Load model
        loaded_model = persistence.load_transformer()
        assert loaded_model is not None

        # Verify forward pass works
        x = torch.randn(32, 50, 86)
        output = loaded_model(x)
        assert output.shape == (32,)

    def test_load_nonexistent_model(self, temp_dir):
        """Test graceful failure when model doesn't exist (returns None)."""
        persistence = ModelPersistence(models_dir=temp_dir)

        # Should return None, not raise exception
        lstm_model = persistence.load_lstm()
        assert lstm_model is None

        transformer_model = persistence.load_transformer()
        assert transformer_model is None


class TestDataPreprocessor:
    """Test data preprocessing for sequence creation."""

    def test_create_sequences(self):
        """Test sequence creation with sliding window."""
        # Create dummy data
        df = pd.DataFrame({
            'feature_1': np.arange(100),
            'feature_2': np.arange(100) * 2,
            'label': np.random.randint(0, 2, size=100)
        })

        preprocessor = DataPreprocessor(sequence_length=10)

        # Create sequences
        sequences, labels = preprocessor.create_sequences(
            df,
            feature_columns=['feature_1', 'feature_2'],
            label_column='label',
            fit_scaler=True
        )

        # Verify shapes
        assert sequences.shape == (90, 10, 2)  # 100 - 10 = 90 sequences
        assert labels.shape == (90,)

        # Verify scaler was fitted
        assert preprocessor.is_fitted is True

    def test_chronological_split_no_shuffle(self):
        """Test chronological split - verify temporal order maintained."""
        # Create data with increasing values
        X = np.arange(1000).reshape(1000, 1, 1)
        y = np.arange(1000)

        preprocessor = DataPreprocessor(sequence_length=10)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_time_series(
            X, y,
            validation_size=0.15,
            test_size=0.15
        )

        # Verify chronological order (no shuffle)
        # Train should be earliest, test should be latest
        assert y_train[0] < y_train[-1]  # Train is ascending
        assert y_train[-1] < y_val[0]  # Train ends before validation starts
        assert y_val[-1] < y_test[0]  # Validation ends before test starts

        # Verify split sizes
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == 1000

    def test_scaler_fit_only_on_train(self):
        """Test that scaler raises error if used before fitting."""
        df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'label': np.random.randint(0, 2, size=100)
        })

        preprocessor = DataPreprocessor(sequence_length=10)

        # Should raise error if fit_scaler=False without fitting first
        with pytest.raises(ValueError, match="Scaler not fitted"):
            preprocessor.create_sequences(
                df,
                feature_columns=['feature_1'],
                label_column='label',
                fit_scaler=False
            )


class TestTimeSeriesDataset:
    """Test PyTorch dataset for time series."""

    def test_dataset_indexing(self):
        """Test dataset returns correct shapes."""
        sequences = np.random.randn(100, 50, 86)
        labels = np.random.randint(0, 2, size=100)

        dataset = TimeSeriesDataset(sequences, labels)

        # Test indexing
        seq, label = dataset[0]
        assert seq.shape == (50, 86)
        assert isinstance(label, torch.Tensor)

    def test_dataset_length(self):
        """Test __len__ works correctly."""
        sequences = np.random.randn(100, 50, 86)
        labels = np.random.randint(0, 2, size=100)

        dataset = TimeSeriesDataset(sequences, labels)
        assert len(dataset) == 100


class TestDeepLearningStrategy:
    """Test deep learning strategy with mocks."""

    @pytest.fixture
    def mock_config(self):
        """Create mock TradingConfig."""
        config = Mock(spec=TradingConfig)
        config.max_position_size_usd = 1000.0
        config.max_daily_drawdown_pct = 5.0
        config.max_open_positions = 3
        config.decision_threshold = 0.6
        return config

    @pytest.fixture
    def mock_provider(self):
        """Create mock exchange provider."""
        provider = Mock()
        provider.fetch_ohlcv = AsyncMock()
        return provider

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_independent_execution(self, mock_config, temp_dir):
        """Test that strategy has separate PositionManager/RiskAudit instances."""
        # Create mock provider
        mock_provider = Mock()

        # Create and save dummy model
        model = BiLSTMClassifier(input_size=86, hidden_size=128, num_layers=2, dropout=0.2)
        persistence = ModelPersistence(models_dir=temp_dir / "models")
        persistence.save_lstm(model)

        with patch('trading.ml.deep_learning.deep_learning_strategy.ModelPersistence') as MockPersistence:
            MockPersistence.return_value.load_lstm.return_value = model

            # Create two strategy instances
            strategy1 = DeepLearningStrategy(
                config=mock_config,
                provider=mock_provider,
                model_type='lstm',
                spec_dir=temp_dir / "specs1"
            )

            strategy2 = DeepLearningStrategy(
                config=mock_config,
                provider=mock_provider,
                model_type='lstm',
                spec_dir=temp_dir / "specs2"
            )

            # Verify they have separate state instances (independent)
            assert strategy1.state is not strategy2.state
            assert strategy1.risk_audit is not strategy2.risk_audit
            assert strategy1.spec_dir != strategy2.spec_dir


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
