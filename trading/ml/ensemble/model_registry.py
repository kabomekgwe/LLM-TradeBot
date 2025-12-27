"""
Model registry for managing ensemble base models.

Provides clean abstraction for adding, removing, and retrieving models
without modifying ensemble code.
"""

from typing import Any, Dict, List
from trading.logging_config import get_logger
from trading.exceptions import ModelError

logger = get_logger(__name__)


class ModelRegistry:
    """
    Clean abstraction for model management.

    Makes it trivial to add/remove models from ensemble without
    rewriting code. Validates models have required methods.
    """

    def __init__(self):
        """Initialize empty model registry."""
        self._models: Dict[str, Any] = {}
        logger.info("Model registry initialized")

    def register_model(self, name: str, model: Any) -> None:
        """
        Register a new model in the registry.

        Args:
            name: Unique identifier for the model
            model: Model instance (must have fit, predict, predict_proba methods)

        Raises:
            ModelError: If model doesn't have required methods or name already exists
        """
        if name in self._models:
            raise ModelError(
                f"Model '{name}' already registered",
                model_name=name,
                context={'existing_models': list(self._models.keys())}
            )

        # Validate model has required methods
        required_methods = ['fit', 'predict', 'predict_proba']
        missing_methods = [
            method for method in required_methods
            if not hasattr(model, method)
        ]

        if missing_methods:
            raise ModelError(
                f"Model missing required methods: {missing_methods}",
                model_name=name,
                context={
                    'required_methods': required_methods,
                    'missing_methods': missing_methods
                }
            )

        self._models[name] = model
        logger.info(
            "Model registered",
            extra={
                'model_name': name,
                'model_type': type(model).__name__,
                'total_models': len(self._models)
            }
        )

    def remove_model(self, name: str) -> None:
        """
        Remove a model from the registry.

        Args:
            name: Model identifier to remove

        Raises:
            ModelError: If model not found
        """
        if name not in self._models:
            raise ModelError(
                f"Model '{name}' not found in registry",
                model_name=name,
                context={'available_models': list(self._models.keys())}
            )

        del self._models[name]
        logger.info(
            "Model removed",
            extra={
                'model_name': name,
                'remaining_models': len(self._models)
            }
        )

    def get_model(self, name: str) -> Any:
        """
        Get a specific model by name.

        Args:
            name: Model identifier

        Returns:
            model: Model instance

        Raises:
            ModelError: If model not found
        """
        if name not in self._models:
            raise ModelError(
                f"Model '{name}' not found in registry",
                model_name=name,
                context={'available_models': list(self._models.keys())}
            )

        return self._models[name]

    def get_models(self) -> Dict[str, Any]:
        """
        Get all registered models.

        Returns:
            models: Dictionary of {name: model}
        """
        return self._models.copy()

    def get_model_names(self) -> List[str]:
        """
        Get names of all registered models.

        Returns:
            names: List of model names
        """
        return list(self._models.keys())

    def __len__(self) -> int:
        """Return number of registered models."""
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        """Check if model is registered."""
        return name in self._models
