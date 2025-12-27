"""
Security-hardened model persistence for ensemble models.

CRITICAL SECURITY: Uses native formats only to prevent code execution risks.
- XGBoost: JSON format (safe, human-readable, no code execution)
- LightGBM: Text format (safe, human-readable, no code execution)
- Random Forest: joblib (sklearn standard, acceptable)
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import joblib
import xgboost as xgb
import lightgbm as lgb

from trading.logging_config import get_logger
from trading.exceptions import ModelError

logger = get_logger(__name__)


class EnsemblePersistence:
    """
    Security-hardened model serialization.

    Uses native JSON/text formats for XGBoost and LightGBM to prevent
    arbitrary code execution risks from unsafe serialization.
    """

    def __init__(self, model_dir: str = 'trading/ml/models/ensemble'):
        """
        Initialize persistence manager.

        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Ensemble persistence initialized",
            extra={'model_dir': str(self.model_dir)}
        )

    def save_models(
        self,
        models: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        """
        Save ensemble models using native formats.

        SECURITY CRITICAL:
        - XGBoost: Native JSON format (REQUIRED, no unsafe alternatives)
        - LightGBM: Native text format (REQUIRED, no unsafe alternatives)
        - Random Forest: joblib (sklearn standard, acceptable)

        Args:
            models: Dictionary of {name: model_instance}
            metadata: Metadata to save (versions, features, training date)

        Raises:
            ModelError: If model saving fails
        """
        try:
            # Save XGBoost model (SECURITY: Native JSON format only)
            if 'xgb' in models:
                xgb_path = self.model_dir / 'xgboost_model.json'
                models['xgb'].save_model(str(xgb_path))
                logger.info(
                    "XGBoost model saved",
                    extra={
                        'path': str(xgb_path),
                        'format': 'JSON',
                        'security': 'native_format'
                    }
                )

            # Save LightGBM model (SECURITY: Native text format only)
            if 'lgbm' in models:
                lgbm_path = self.model_dir / 'lightgbm_model.txt'
                # LightGBM sklearn wrapper has booster_ attribute
                if hasattr(models['lgbm'], 'booster_'):
                    models['lgbm'].booster_.save_model(str(lgbm_path))
                else:
                    # If already a Booster object
                    models['lgbm'].save_model(str(lgbm_path))
                logger.info(
                    "LightGBM model saved",
                    extra={
                        'path': str(lgbm_path),
                        'format': 'text',
                        'security': 'native_format'
                    }
                )

            # Save Random Forest (joblib standard for sklearn)
            if 'rf' in models:
                rf_path = self.model_dir / 'random_forest_model.joblib'
                joblib.dump(models['rf'], rf_path)
                logger.info(
                    "Random Forest model saved",
                    extra={
                        'path': str(rf_path),
                        'format': 'joblib',
                        'security': 'sklearn_standard'
                    }
                )

            # Save metadata as JSON (SECURITY: Safe format)
            metadata_path = self.model_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(
                "Metadata saved",
                extra={
                    'path': str(metadata_path),
                    'format': 'JSON'
                }
            )

        except Exception as e:
            raise ModelError(
                f"Failed to save ensemble models: {e}",
                model_name="EnsemblePersistence",
                context={'error': str(e)}
            )

    def load_models(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load ensemble models from disk.

        Returns:
            models: Dictionary of {name: model_instance}
            metadata: Metadata dict

        Raises:
            ModelError: If model loading fails
        """
        try:
            models = {}

            # Load XGBoost model (SECURITY: Native JSON format)
            xgb_path = self.model_dir / 'xgboost_model.json'
            if xgb_path.exists():
                xgb_model = xgb.XGBClassifier()
                xgb_model.load_model(str(xgb_path))
                models['xgb'] = xgb_model
                logger.info(
                    "XGBoost model loaded",
                    extra={'path': str(xgb_path), 'format': 'JSON'}
                )

            # Load LightGBM model (SECURITY: Native text format)
            lgbm_path = self.model_dir / 'lightgbm_model.txt'
            if lgbm_path.exists():
                # Load as Booster, then wrap in sklearn interface
                booster = lgb.Booster(model_file=str(lgbm_path))
                # Create LGBMClassifier and assign booster
                lgbm_model = lgb.LGBMClassifier()
                lgbm_model._Booster = booster
                models['lgbm'] = lgbm_model
                logger.info(
                    "LightGBM model loaded",
                    extra={'path': str(lgbm_path), 'format': 'text'}
                )

            # Load Random Forest (joblib)
            rf_path = self.model_dir / 'random_forest_model.joblib'
            if rf_path.exists():
                rf_model = joblib.load(rf_path)
                models['rf'] = rf_model
                logger.info(
                    "Random Forest model loaded",
                    extra={'path': str(rf_path), 'format': 'joblib'}
                )

            # Load metadata (JSON)
            metadata_path = self.model_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(
                    "Metadata loaded",
                    extra={'path': str(metadata_path)}
                )
            else:
                metadata = {}

            if not models:
                raise ModelError(
                    "No ensemble models found",
                    model_name="EnsemblePersistence",
                    context={'model_dir': str(self.model_dir)}
                )

            return models, metadata

        except Exception as e:
            raise ModelError(
                f"Failed to load ensemble models: {e}",
                model_name="EnsemblePersistence",
                context={'error': str(e)}
            )

    def model_exists(self) -> bool:
        """
        Check if saved models exist.

        Returns:
            exists: True if at least one model file exists
        """
        xgb_exists = (self.model_dir / 'xgboost_model.json').exists()
        lgbm_exists = (self.model_dir / 'lightgbm_model.txt').exists()
        rf_exists = (self.model_dir / 'random_forest_model.joblib').exists()

        return xgb_exists or lgbm_exists or rf_exists
