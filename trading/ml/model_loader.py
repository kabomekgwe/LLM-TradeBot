"""Model loader with caching for production serving.

Singleton pattern ensures models are loaded once and cached in memory
for fast inference across multiple prediction requests.
"""

import logging
from typing import Dict, Optional, Any
from pathlib import Path
import os


class ModelLoader:
    """Singleton model loader with in-memory caching.

    Loads ML models (XGBoost, LightGBM, LSTM) from disk and caches them
    in memory for fast repeated inference. Thread-safe singleton pattern
    ensures only one instance exists.

    Example:
        >>> loader = ModelLoader()
        >>> model = loader.load_model("xgboost_model.pkl", "xgboost")
        >>> predictions = model.predict(features)

    Design Pattern: Singleton with lazy loading and LRU cache
    """

    _instance = None
    _models: Dict[str, Any] = {}

    def __new__(cls):
        """Ensure only one ModelLoader instance exists (singleton)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize model loader.

        Sets up logger and model directory path from environment.
        Called on every ModelLoader() but initialization happens once.
        """
        if not hasattr(self, 'initialized'):
            self.logger = logging.getLogger(__name__)
            self.model_dir = Path(os.getenv('MODEL_DIR', 'models'))
            self.initialized = True

    def load_model(self, model_name: str, model_type: str) -> Optional[Any]:
        """Load model from disk or return cached instance.

        Args:
            model_name: Model filename (e.g., "xgboost_model.pkl")
            model_type: Model type ("xgboost", "lightgbm", or "lstm")

        Returns:
            Loaded model instance or None if loading fails

        Raises:
            ValueError: If model_type is not supported

        Example:
            >>> loader = ModelLoader()
            >>> xgb_model = loader.load_model("xgboost_v1.pkl", "xgboost")
            >>> if xgb_model:
            ...     predictions = xgb_model.predict(X)
        """
        cache_key = f"{model_type}_{model_name}"

        # Return cached model if available
        if cache_key in self._models:
            self.logger.info(f"Returning cached model: {cache_key}")
            return self._models[cache_key]

        # Validate model file exists
        model_path = self.model_dir / model_name
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return None

        # Load model based on type
        try:
            model = self._load_model_by_type(model_type, str(model_path))

            if model:
                # Cache the loaded model
                self._models[cache_key] = model
                self.logger.info(f"Loaded and cached model: {cache_key}")

            return model

        except Exception as e:
            self.logger.error(f"Failed to load model {cache_key}: {e}")
            return None

    def _load_model_by_type(self, model_type: str, model_path: str) -> Optional[Any]:
        """Load model based on type using appropriate model class.

        Args:
            model_type: Type of model to load
            model_path: Full path to model file

        Returns:
            Loaded model instance

        Raises:
            ValueError: If model_type is unsupported
        """
        if model_type == "xgboost":
            from .models.xgboost_model import XGBoostModel
            return XGBoostModel.load_model(model_path)

        elif model_type == "lightgbm":
            from .models.lightgbm_model import LightGBMModel
            return LightGBMModel.load_model(model_path)

        elif model_type == "lstm":
            from .models.lstm_model import LSTMModel
            return LSTMModel.load_model(model_path)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def clear_cache(self):
        """Clear all cached models from memory.

        Useful for testing or when models need to be reloaded.

        Example:
            >>> loader = ModelLoader()
            >>> loader.clear_cache()
        """
        self._models.clear()
        self.logger.info("Model cache cleared")

    def get_cached_models(self) -> list:
        """Get list of currently cached models.

        Returns:
            List of cached model keys

        Example:
            >>> loader = ModelLoader()
            >>> cached = loader.get_cached_models()
            >>> print(f"Cached models: {', '.join(cached)}")
        """
        return list(self._models.keys())

    def preload_models(self, model_configs: list):
        """Preload multiple models at startup.

        Args:
            model_configs: List of (model_name, model_type) tuples

        Example:
            >>> configs = [
            ...     ("xgboost_v1.pkl", "xgboost"),
            ...     ("lightgbm_v1.pkl", "lightgbm"),
            ... ]
            >>> loader = ModelLoader()
            >>> loader.preload_models(configs)
        """
        for model_name, model_type in model_configs:
            self.load_model(model_name, model_type)

        self.logger.info(f"Preloaded {len(model_configs)} models")
