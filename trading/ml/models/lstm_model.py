"""LSTM Model - Deep learning time series prediction.

Long Short-Term Memory (LSTM) networks are specialized RNNs
designed to capture temporal dependencies in sequential data.

Advantages:
- Captures temporal patterns in price movements
- Learns complex non-linear relationships
- Handles variable-length sequences
- Robust to vanishing gradient problem

Uses PyTorch for implementation (lighter than TensorFlow).
"""

import logging
import numpy as np
from typing import Optional, Tuple
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. LSTM model will not be available.")


class LSTMNetwork(nn.Module):
    """LSTM neural network architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """Initialize LSTM network.

        Args:
            input_dim: Number of input features
            hidden_dim: Size of hidden layer
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out


class LSTMModel:
    """LSTM model wrapper for price direction prediction.

    Provides scikit-learn-like interface for PyTorch LSTM model.

    Example:
        >>> model = LSTMModel(input_dim=50)
        >>> model.train(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        device: Optional[str] = None,
    ):
        """Initialize LSTM model.

        Args:
            input_dim: Number of input features
            hidden_dim: Size of hidden layer
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM model. Install with: pip install torch")

        self.logger = logging.getLogger(__name__)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Model
        self.model = LSTMNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        # Optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

        # Training history
        self.train_losses = []
        self.val_losses = []

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_patience: int = 10,
    ):
        """Train LSTM model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_stopping_patience: Stop if no improvement for N epochs
        """
        self.logger.info(f"Training LSTM model on {self.device}...")

        # Reshape input: (samples, features) -> (samples, 1, features)
        # Each sample is treated as a sequence of length 1
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)

        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Validation data
        if X_val is not None and y_val is not None:
            X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        else:
            X_val_tensor = None
            y_val_tensor = None

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # Average training loss
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            if X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
                    self.val_losses.append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                if (epoch + 1) % 10 == 0:
                    self.logger.info(
                        f"Epoch {epoch + 1}/{self.epochs} - "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}")

        self.logger.info("LSTM training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features to predict on

        Returns:
            Array of predicted probabilities for class 1
        """
        # Reshape input
        X = X.reshape(X.shape[0], 1, X.shape[1])

        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)

        # Convert to numpy and flatten
        return predictions.cpu().numpy().flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (sklearn-compatible).

        Args:
            X: Features to predict on

        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        proba_1 = self.predict(X)
        proba_0 = 1 - proba_1

        return np.column_stack([proba_0, proba_1])

    def predict_binary(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels.

        Args:
            X: Features to predict on
            threshold: Classification threshold

        Returns:
            Array of binary predictions (0 or 1)
        """
        proba = self.predict(X)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (approximation).

        Note: LSTM doesn't have built-in feature importance like tree models.
        This returns uniform importance as a placeholder.

        Returns:
            Array of feature importance scores (uniform)
        """
        self.logger.warning("LSTM does not have native feature importance. Returning uniform weights.")
        return np.ones(self.input_dim) / self.input_dim

    def save_model(self, filepath: str):
        """Save model to file.

        Args:
            filepath: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, filepath)

        self.logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'LSTMModel':
        """Load model from file.

        Args:
            filepath: Path to load model from

        Returns:
            Loaded LSTMModel instance
        """
        checkpoint = torch.load(filepath)

        # Create instance with saved parameters
        instance = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout'],
        )

        # Load state
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        instance.train_losses = checkpoint['train_losses']
        instance.val_losses = checkpoint['val_losses']

        return instance

    def get_training_log(self) -> dict:
        """Get training history.

        Returns:
            Dictionary with training and validation losses
        """
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs_trained': len(self.train_losses),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LSTMModel(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers})"
        )
