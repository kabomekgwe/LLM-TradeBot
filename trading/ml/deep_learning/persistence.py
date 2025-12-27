"""Model Persistence for Deep Learning Models.

Provides security-hardened model loading using PyTorch's state_dict format.

SECURITY DESIGN:
- Uses state_dict only (NOT pickle or joblib which allow arbitrary code execution)
- Model architecture defined in code, only weights loaded from file
- Validates model files exist before loading
- Graceful failure if models not found

WHY this is secure:
- PyTorch state_dict is just a dictionary of tensors (no executable code)
- Architecture instantiated from trusted source code
- No deserialization of arbitrary Python objects
- Safe from pickle-based attacks

References:
- PyTorch security best practices
- 07-03-PLAN.md security requirements
"""

import torch
from pathlib import Path
from typing import Optional

from trading.ml.deep_learning.models.lstm_model import BiLSTMClassifier
from trading.ml.deep_learning.models.transformer_model import TransformerClassifier
from trading.logging_config import get_logger


class ModelPersistence:
    """Handles loading and saving of trained deep learning models.

    SECURITY: Uses PyTorch state_dict format only (safe serialization).
    Architecture is defined in code, only weights are loaded from files.

    Example:
        >>> persistence = ModelPersistence()
        >>> lstm_model = persistence.load_lstm()
        >>> if lstm_model is not None:
        ...     print("LSTM model loaded successfully")
    """

    def __init__(
        self,
        models_dir: Path = Path("trading/ml/models/deep_learning")
    ):
        """Initialize model persistence handler.

        Args:
            models_dir: Directory containing trained model files
        """
        self.models_dir = Path(models_dir)
        self.logger = get_logger(__name__)

        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Model file paths
        self.lstm_path = self.models_dir / "lstm_model.pth"
        self.transformer_path = self.models_dir / "transformer_model.pth"

    def load_lstm(
        self,
        input_size: int = 86,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ) -> Optional[BiLSTMClassifier]:
        """Load trained BiLSTM model using state_dict (secure).

        SECURITY: Architecture instantiated from code, only weights loaded.

        Args:
            input_size: Number of input features (must match training)
            hidden_size: Hidden dimension (must match training)
            num_layers: Number of LSTM layers (must match training)
            dropout: Dropout rate (must match training)

        Returns:
            BiLSTMClassifier instance with loaded weights, or None if file not found
        """
        if not self.lstm_path.exists():
            self.logger.warning(
                f"LSTM model file not found: {self.lstm_path}. "
                "Train model first using: python trading/ml/deep_learning/training/train_lstm.py"
            )
            return None

        try:
            # Step 1: Instantiate architecture from trusted code
            model = BiLSTMClassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )

            # Step 2: Load ONLY the state_dict (weights only, no code)
            state_dict = torch.load(
                self.lstm_path,
                map_location=torch.device('cpu'),  # Always load to CPU first
                weights_only=True  # PyTorch 2.0+ security flag
            )

            # Step 3: Load weights into architecture
            model.load_state_dict(state_dict)

            # Set to evaluation mode (not Python eval!)
            model.eval()

            self.logger.info(f"Successfully loaded LSTM model from {self.lstm_path}")
            return model

        except Exception as e:
            self.logger.error(f"Failed to load LSTM model: {e}")
            return None

    def load_transformer(
        self,
        input_size: int = 86,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2
    ) -> Optional[TransformerClassifier]:
        """Load trained Transformer model using state_dict (secure).

        SECURITY: Architecture instantiated from code, only weights loaded.

        Args:
            input_size: Number of input features (must match training)
            d_model: Model dimension (must match training)
            nhead: Number of attention heads (must match training)
            num_layers: Number of encoder layers (must match training)
            dropout: Dropout rate (must match training)

        Returns:
            TransformerClassifier instance with loaded weights, or None if file not found
        """
        if not self.transformer_path.exists():
            self.logger.warning(
                f"Transformer model file not found: {self.transformer_path}. "
                "Train model first using: python trading/ml/deep_learning/training/train_transformer.py"
            )
            return None

        try:
            # Step 1: Instantiate architecture from trusted code
            model = TransformerClassifier(
                input_size=input_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout
            )

            # Step 2: Load ONLY the state_dict (weights only, no code)
            state_dict = torch.load(
                self.transformer_path,
                map_location=torch.device('cpu'),  # Always load to CPU first
                weights_only=True  # PyTorch 2.0+ security flag
            )

            # Step 3: Load weights into architecture
            model.load_state_dict(state_dict)

            # Set to evaluation mode (not Python eval!)
            model.eval()

            self.logger.info(f"Successfully loaded Transformer model from {self.transformer_path}")
            return model

        except Exception as e:
            self.logger.error(f"Failed to load Transformer model: {e}")
            return None

    def save_lstm(self, model: BiLSTMClassifier) -> bool:
        """Save BiLSTM model using state_dict (secure).

        Args:
            model: Trained BiLSTMClassifier instance

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            torch.save(model.state_dict(), self.lstm_path)
            self.logger.info(f"Saved LSTM model to {self.lstm_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save LSTM model: {e}")
            return False

    def save_transformer(self, model: TransformerClassifier) -> bool:
        """Save Transformer model using state_dict (secure).

        Args:
            model: Trained TransformerClassifier instance

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            torch.save(model.state_dict(), self.transformer_path)
            self.logger.info(f"Saved Transformer model to {self.transformer_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save Transformer model: {e}")
            return False

    def model_exists(self, model_type: str) -> bool:
        """Check if a trained model exists.

        Args:
            model_type: "lstm" or "transformer"

        Returns:
            True if model file exists, False otherwise
        """
        if model_type == "lstm":
            return self.lstm_path.exists()
        elif model_type == "transformer":
            return self.transformer_path.exists()
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'lstm' or 'transformer'")
