"""
Ensemble model framework for combining multiple ML models.

This module provides a production-ready ensemble framework that combines
LightGBM, XGBoost, and Random Forest with regime-aware strategy switching.
"""

from trading.ml.ensemble.base_ensemble import BaseEnsemble
from trading.ml.ensemble.model_registry import ModelRegistry
from trading.ml.ensemble.persistence import EnsemblePersistence
from trading.ml.ensemble.voting_ensemble import VotingEnsemble
from trading.ml.ensemble.stacking_ensemble import StackingEnsemble
from trading.ml.ensemble.regime_aware_ensemble import RegimeAwareEnsemble

__all__ = [
    'BaseEnsemble',
    'ModelRegistry',
    'EnsemblePersistence',
    'VotingEnsemble',
    'StackingEnsemble',
    'RegimeAwareEnsemble',
]
