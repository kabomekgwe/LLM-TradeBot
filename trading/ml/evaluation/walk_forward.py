"""
Walk-forward validation for time-series models.

CRITICAL: Implements chronological cross-validation that prevents look-ahead bias
and data leakage. This is MANDATORY for time-series - standard k-fold CV causes
massive data leakage by training on future data and testing on past data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from typing import Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time-series models.
    
    Uses expanding window approach: train on all historical data up to a point,
    test on a window of future data, then advance forward in time.
    
    CRITICAL DESIGN:
    - Chronological splits ONLY (NO shuffle anywhere)
    - Scaler fitted ONLY on training data in each fold
    - train_end < test_start for all folds (temporal order maintained)
    
    Parameters:
        initial_train_size (int): Initial training window size (default: 252 = 1 year)
        test_size (int): Test window size (default: 60 = 3 months)
        step_size (int): Number of samples to advance each fold (default: 30 = 1 month)
    """
    
    def __init__(
        self,
        initial_train_size: int = 252,
        test_size: int = 60,
        step_size: int = 30
    ):
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        
        logger.info(
            f"WalkForwardValidator initialized: "
            f"train={initial_train_size}, test={test_size}, step={step_size}"
        )
    
    def validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Perform walk-forward validation.
        
        Args:
            model: Model with fit() and predict() methods (NOT fitted yet)
            X: Features (DataFrame with any index)
            y: Labels (Series with same index as X)
        
        Returns:
            DataFrame with columns:
                - fold: Fold number
                - train_start, train_end: Training window indices
                - test_start, test_end: Test window indices
                - accuracy, precision, recall: Performance metrics
                - predictions: Array of predictions for this fold
                - actuals: Array of actual labels for this fold
        
        CRITICAL: Scaler fitted ONLY on training data in each fold.
        This prevents data leakage (Pitfall 1 from research).
        """
        results = []
        train_end = self.initial_train_size
        fold = 1
        
        # Verify we have enough data
        if len(X) < self.initial_train_size + self.test_size:
            raise ValueError(
                f"Insufficient data: need {self.initial_train_size + self.test_size}, "
                f"got {len(X)}"
            )
        
        logger.info(f"Starting walk-forward validation with {len(X)} samples")
        
        while train_end + self.test_size <= len(X):
            # Chronological split (NO shuffle!)
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[train_end:train_end + self.test_size]
            y_test = y.iloc[train_end:train_end + self.test_size]
            
            # CRITICAL: Verify temporal order
            train_start_idx = 0
            train_end_idx = train_end - 1
            test_start_idx = train_end
            test_end_idx = train_end + self.test_size - 1
            
            assert train_end_idx < test_start_idx, \
                f"Temporal order violated: train_end={train_end_idx} >= test_start={test_start_idx}"
            
            # Fit scaler ONLY on training data (prevents data leakage)
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)  # Use fitted scaler
            
            # Train on past, predict on future
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                
                # Store results
                results.append({
                    'fold': fold,
                    'train_start': train_start_idx,
                    'train_end': train_end_idx,
                    'test_start': test_start_idx,
                    'test_end': test_end_idx,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'predictions': y_pred,
                    'actuals': y_test.values
                })
                
                logger.info(
                    f"Fold {fold}: "
                    f"train=[{train_start_idx}:{train_end_idx}], "
                    f"test=[{test_start_idx}:{test_end_idx}], "
                    f"acc={accuracy:.3f}, prec={precision:.3f}, rec={recall:.3f}"
                )
                
            except Exception as e:
                logger.error(f"Fold {fold} failed: {e}")
                raise
            
            # Move forward in time
            train_end += self.step_size
            fold += 1
        
        results_df = pd.DataFrame(results)
        logger.info(
            f"Walk-forward validation complete: {len(results_df)} folds, "
            f"avg accuracy={results_df['accuracy'].mean():.3f}"
        )
        
        return results_df
    
    def get_split_count(self, data_length: int) -> int:
        """
        Calculate number of folds for given data length.
        
        Args:
            data_length: Total number of samples
        
        Returns:
            Number of folds that will be created
        """
        if data_length < self.initial_train_size + self.test_size:
            return 0
        
        train_end = self.initial_train_size
        fold_count = 0
        
        while train_end + self.test_size <= data_length:
            fold_count += 1
            train_end += self.step_size
        
        return fold_count
