"""BiLSTM Classifier for Financial Time Series.

Implements Bidirectional Long Short-Term Memory (BiLSTM) neural network
for binary classification of price movements (up/down).

Architecture follows research-backed best practices:
- Bidirectional LSTM captures both past and future context within sequence
- Dropout between layers prevents overfitting
- Uses last timestep output for classification
- 2 * hidden_size for FC layer (bidirectional doubles output dimension)

References:
- PyTorch LSTM documentation
- Financial time series ML research (ScienceDirect 2022)
"""

import torch
import torch.nn as nn
from typing import Tuple


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier for price direction prediction.

    CRITICAL DESIGN DECISIONS:
    - bidirectional=True: Captures both past and future context (5-10% better than unidirectional)
    - dropout=0.2: Prevents overfitting (research recommends 20-40% for financial data)
    - batch_first=True: Input shape (batch, seq_len, features) for easier data handling
    - Uses last timestep output: Standard approach for sequence classification

    Example:
        >>> model = BiLSTMClassifier(input_size=86, hidden_size=128, num_layers=2, dropout=0.2)
        >>> x = torch.randn(32, 50, 86)  # batch=32, seq_len=50, features=86
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([32])
    """

    def __init__(
        self,
        input_size: int = 86,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """Initialize BiLSTM classifier.

        Args:
            input_size: Number of input features (86 from Phase 5's feature engineering)
            hidden_size: Number of hidden units in LSTM (128 proven for financial data)
            num_layers: Number of stacked LSTM layers (2-3 typical, more risks overfitting)
            dropout: Dropout rate between LSTM layers (0.2 = 20% dropout)
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # Only apply dropout if multiple layers
            bidirectional=True,  # BiLSTM (output dimension = 2 * hidden_size)
            batch_first=True  # Input shape: (batch, seq_len, features)
        )

        # Dropout after LSTM
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer (2 * hidden_size because bidirectional)
        self.fc = nn.Linear(2 * hidden_size, 1)  # Binary classification (single logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through BiLSTM.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Logits of shape (batch,) for binary classification
        """
        # LSTM forward pass
        # lstm_out shape: (batch, seq_len, 2 * hidden_size)
        # hidden/cell shape: (2 * num_layers, batch, hidden_size) [2x for bidirectional]
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last timestep output (standard for classification)
        last_output = lstm_out[:, -1, :]  # Shape: (batch, 2 * hidden_size)

        # Apply dropout
        out = self.dropout(last_output)

        # Fully connected layer
        logits = self.fc(out)  # Shape: (batch, 1)

        # Squeeze to (batch,) for BCEWithLogitsLoss compatibility
        return logits.squeeze(-1)

    def get_model_info(self) -> dict:
        """Get model architecture information.

        Returns:
            Dictionary with model configuration and parameter count
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'architecture': 'BiLSTM',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout_rate,
            'bidirectional': True,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        }
