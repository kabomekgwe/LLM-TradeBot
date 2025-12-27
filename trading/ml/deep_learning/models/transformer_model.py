"""Transformer Encoder Architecture with Causal Masking for Financial Time Series.

CRITICAL DESIGN DECISIONS:
- Causal masking MANDATORY: Prevents future data leakage (Pitfall 4 from 07-RESEARCH.md)
- Positional encoding: Standard sinusoidal encoding (verified math, GPU-optimized)
- Sequence length 50: Prevents O(n²) memory explosion (Pitfall 2)
- d_model=128: Must be divisible by nhead (128 / 8 = 16)

This architecture prevents common pitfalls:
- Pitfall 2: O(n²) attention complexity - sequence length kept to 50
- Pitfall 4: Future data leakage - causal mask creates upper triangular attention
- Don't hand-roll positional encoding - use standard implementation

Source: PyTorch docs + time series Transformer tutorials (07-RESEARCH.md lines 147-246)
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for time series Transformer.

    Uses sinusoidal functions to inject position information into embeddings.
    This is the standard approach from "Attention is All You Need" (Vaswani et al., 2017).

    Example:
        >>> pos_enc = PositionalEncoding(d_model=128)
        >>> x = torch.randn(32, 50, 128)  # (batch, seq_len, d_model)
        >>> x_pos = pos_enc(x)
        >>> x_pos.shape
        torch.Size([32, 50, 128])
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.

        Args:
            d_model: Dimension of the model (must match input embeddings)
            max_len: Maximum sequence length (default 5000)
        """
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        # Register as buffer (not trainable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added, same shape as input
        """
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    """Transformer encoder with causal masking for binary classification.

    Architecture:
    - Input projection: 86 features → d_model
    - Positional encoding: sinusoidal position injection
    - Transformer encoder: multi-head self-attention with causal mask
    - Classification head: d_model → 1 (binary logit)

    CRITICAL: Uses causal mask to prevent future data leakage (Pitfall 4).
    Without causal mask, model "cheats" by seeing future timesteps.

    Example:
        >>> model = TransformerClassifier(
        ...     input_size=86,
        ...     d_model=128,
        ...     nhead=8,
        ...     num_layers=2,
        ...     dropout=0.2
        ... )
        >>> x = torch.randn(32, 50, 86)  # (batch, seq_len, features)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([32])  # Binary classification logits
    """

    def __init__(
        self,
        input_size: int = 86,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """Initialize Transformer classifier.

        Args:
            input_size: Number of input features (86 from Phase 5)
            d_model: Model dimension (must be divisible by nhead)
            nhead: Number of attention heads (4, 8, 16 typical)
            num_layers: Number of transformer encoder layers (2-4 typical)
            dropout: Dropout rate (0.2 = 20% dropout)
        """
        super().__init__()

        # Input projection (map 86 features to d_model)
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,           # Model dimension (128)
            nhead=nhead,               # Number of attention heads (8)
            dim_feedforward=4*d_model, # FFN dimension (512 = 4 * 128)
            dropout=dropout,           # Dropout rate (0.2)
            batch_first=True           # Input shape: (batch, seq_len, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.fc = nn.Linear(d_model, 1)  # Binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal masking.

        CRITICAL: Creates causal mask to prevent attending to future timesteps.
        This prevents future data leakage (Pitfall 4 from 07-RESEARCH.md).

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Binary classification logits of shape (batch,)
        """
        # x shape: (batch, seq_len, input_size)

        # Project input to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create causal mask (prevent attending to future)
        # Upper triangular mask with -inf (attention weights become 0 after softmax)
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len) * float('-inf'),
            diagonal=1
        ).to(x.device)

        # Transformer encoder with causal mask
        transformer_out = self.transformer_encoder(
            x,
            mask=causal_mask
        )  # (batch, seq_len, d_model)

        # Use last timestep output
        last_output = transformer_out[:, -1, :]  # (batch, d_model)

        # Apply dropout
        out = self.dropout(last_output)

        # Classification head
        logits = self.fc(out)  # (batch, 1)

        return logits.squeeze(-1)  # (batch,)
