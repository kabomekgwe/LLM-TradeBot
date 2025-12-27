"""PyTorch Dataset for Time Series Sequences.

CRITICAL: This dataset does NOT shuffle data - maintains temporal order.
Data must be pre-normalized and pre-windowed before being passed to this dataset.

Design decisions to prevent common pitfalls:
- NO shuffle: Prevents data leakage (Pitfall 1 from research)
- Expects pre-normalized data: Preprocessing handles StandardScaler
- FloatTensor conversion: PyTorch compatibility
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences.

    CRITICAL: Data must be pre-normalized and pre-windowed.
    This dataset does NOT shuffle - maintains temporal order.

    Example:
        >>> sequences = np.random.randn(1000, 50, 86)  # 1000 samples, 50 timesteps, 86 features
        >>> labels = np.random.randint(0, 2, size=1000)  # Binary labels
        >>> dataset = TimeSeriesDataset(sequences, labels)
        >>> len(dataset)
        1000
        >>> seq, label = dataset[0]
        >>> seq.shape
        torch.Size([50, 86])
    """

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """Initialize time series dataset.

        Args:
            sequences: np.array of shape (N, seq_len, features) - already normalized
            labels: np.array of shape (N,) - binary labels (0 or 1)

        Raises:
            ValueError: If sequences and labels have different lengths
        """
        if len(sequences) != len(labels):
            raise ValueError(
                f"Sequences and labels must have same length: "
                f"got {len(sequences)} vs {len(labels)}"
            )

        # Convert to PyTorch tensors
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (sequence, label)
            - sequence: shape (seq_len, features)
            - label: scalar (0 or 1)
        """
        return self.sequences[idx], self.labels[idx]

    def get_info(self) -> dict:
        """Get dataset information.

        Returns:
            Dictionary with dataset statistics
        """
        return {
            'num_samples': len(self.sequences),
            'sequence_length': self.sequences.shape[1],
            'num_features': self.sequences.shape[2],
            'positive_samples': int(self.labels.sum().item()),
            'negative_samples': int((1 - self.labels).sum().item()),
            'class_balance': float(self.labels.mean().item()),
        }
