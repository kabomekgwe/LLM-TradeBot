"""ML Model Serving API - FastAPI endpoints for predictions.

Provides REST API for serving ML model predictions with model caching
and error handling. Supports XGBoost, LightGBM, and LSTM models.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
from pathlib import Path

from ..ml.model_loader import ModelLoader

# Create router
router = APIRouter(prefix="/api/v1/ml", tags=["ml-serving"])


# Request/Response models
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint.

    Example:
        {
            "model_name": "xgboost_model.pkl",
            "model_type": "xgboost",
            "features": [[1.5, 2.3, 0.8, 1.2, 0.5]]
        }
    """

    model_name: str = Field(..., description="Model filename (e.g., 'xgboost_model.pkl')")
    model_type: str = Field(
        ...,
        description="Model type: 'xgboost', 'lightgbm', or 'lstm'",
        pattern="^(xgboost|lightgbm|lstm)$"
    )
    features: List[List[float]] = Field(
        ...,
        description="2D array of features (samples x features)",
        min_items=1
    )


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint.

    Example:
        {
            "predictions": [0.75, 0.82],
            "model_name": "xgboost_model.pkl",
            "model_type": "xgboost",
            "num_samples": 2
        }
    """

    predictions: List[float] = Field(..., description="Model predictions")
    model_name: str = Field(..., description="Model used for predictions")
    model_type: str = Field(..., description="Type of model")
    num_samples: int = Field(..., description="Number of samples predicted")


class ModelInfo(BaseModel):
    """Model metadata."""

    name: str = Field(..., description="Model filename")
    size_mb: float = Field(..., description="File size in megabytes")
    type: str = Field(default="unknown", description="Inferred model type")


class ModelsListResponse(BaseModel):
    """Response model for models listing endpoint."""

    models: List[ModelInfo] = Field(..., description="List of available models")
    total: int = Field(..., description="Total number of models")


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Serve predictions from cached ML models.

    Loads model from cache (or disk if first request) and returns predictions.
    Supports batch predictions for multiple samples.

    Args:
        request: Prediction request with model info and features

    Returns:
        Predictions with metadata

    Raises:
        HTTPException: 404 if model not found, 500 if prediction fails

    Example:
        ```bash
        curl -X POST http://localhost:5173/api/v1/ml/predict \\
          -H "Content-Type: application/json" \\
          -d '{
            "model_name": "xgboost_model.pkl",
            "model_type": "xgboost",
            "features": [[1.5, 2.3, 0.8, 1.2, 0.5]]
          }'
        ```

    Response:
        ```json
        {
            "predictions": [0.75],
            "model_name": "xgboost_model.pkl",
            "model_type": "xgboost",
            "num_samples": 1
        }
        ```
    """
    # Load model (from cache or disk)
    loader = ModelLoader()
    model = loader.load_model(request.model_name, request.model_type)

    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_name}' not found or failed to load"
        )

    try:
        # Convert features to numpy array
        X = np.array(request.features)

        # Validate feature dimensions
        if X.ndim != 2:
            raise ValueError(
                f"Features must be 2D array, got {X.ndim}D. "
                "Use [[features]] for single sample."
            )

        # Get predictions
        # For binary classification, use predict_proba and take positive class probability
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X)
            # Take probability of positive class (index 1)
            if predictions.ndim == 2 and predictions.shape[1] == 2:
                predictions = predictions[:, 1]
        else:
            predictions = model.predict(X)

        return PredictionResponse(
            predictions=predictions.tolist(),
            model_name=request.model_name,
            model_type=request.model_type,
            num_samples=len(predictions)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """List all available models in the models directory.

    Scans the models directory and returns metadata for each model file.
    Supported file extensions: .pkl, .h5, .pth

    Returns:
        List of model metadata

    Example:
        ```bash
        curl http://localhost:5173/api/v1/ml/models
        ```

    Response:
        ```json
        {
            "models": [
                {
                    "name": "xgboost_model.pkl",
                    "size_mb": 2.5,
                    "type": "xgboost"
                }
            ],
            "total": 1
        }
        ```
    """
    model_dir = Path("models")

    if not model_dir.exists():
        return ModelsListResponse(models=[], total=0)

    models = []

    for file_path in model_dir.iterdir():
        if file_path.is_file() and file_path.suffix in ['.pkl', '.h5', '.pth', '.bin']:
            # Infer model type from filename
            model_type = "unknown"
            name_lower = file_path.name.lower()
            if "xgboost" in name_lower or "xgb" in name_lower:
                model_type = "xgboost"
            elif "lightgbm" in name_lower or "lgb" in name_lower:
                model_type = "lightgbm"
            elif "lstm" in name_lower or "rnn" in name_lower:
                model_type = "lstm"

            models.append(
                ModelInfo(
                    name=file_path.name,
                    size_mb=round(file_path.stat().st_size / (1024 * 1024), 2),
                    type=model_type
                )
            )

    # Sort by name
    models.sort(key=lambda m: m.name)

    return ModelsListResponse(
        models=models,
        total=len(models)
    )


@router.get("/cache", response_model=dict)
async def get_cache_status():
    """Get current model cache status.

    Returns information about models currently loaded in memory.

    Returns:
        Cache status with loaded models

    Example:
        ```bash
        curl http://localhost:5173/api/v1/ml/cache
        ```

    Response:
        ```json
        {
            "cached_models": ["xgboost_xgboost_model.pkl"],
            "count": 1
        }
        ```
    """
    loader = ModelLoader()
    cached = loader.get_cached_models()

    return {
        "cached_models": cached,
        "count": len(cached)
    }


@router.post("/cache/clear", response_model=dict)
async def clear_cache():
    """Clear model cache.

    Removes all models from memory. Next prediction request will
    reload models from disk.

    Returns:
        Success message

    Example:
        ```bash
        curl -X POST http://localhost:5173/api/v1/ml/cache/clear
        ```
    """
    loader = ModelLoader()
    loader.clear_cache()

    return {
        "status": "success",
        "message": "Model cache cleared"
    }


@router.get("/health", response_model=dict)
async def health_check():
    """ML serving health check.

    Returns:
        Health status

    Example:
        ```bash
        curl http://localhost:5173/api/v1/ml/health
        ```
    """
    return {
        "status": "healthy",
        "service": "ml-serving"
    }
