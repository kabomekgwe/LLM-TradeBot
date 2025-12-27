"""
Unit tests for walk-forward validation.

Tests ensure:
- Chronological splits (train_end < test_start)
- No shuffle (temporal order maintained)
- Scaler fitted only on training data
- Correct fold count
- Results structure
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from unittest.mock import Mock, patch
from trading.ml.evaluation.walk_forward import WalkForwardValidator


class TestWalkForwardValidator:
    """Test suite for WalkForwardValidator."""
    
    def test_chronological_splits(self):
        """Verify train_end < test_start for all folds."""
        # Create dummy data
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(1000, 10))
        y = pd.Series(np.random.randint(0, 2, 1000))
        
        # Run validation
        validator = WalkForwardValidator(
            initial_train_size=252,
            test_size=60,
            step_size=30
        )
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = validator.validate(model, X, y)
        
        # Verify temporal order for all folds
        for _, row in results.iterrows():
            assert row['train_end'] < row['test_start'], \
                f"Temporal order violated in fold {row['fold']}: " \
                f"train_end={row['train_end']} >= test_start={row['test_start']}"
    
    def test_no_shuffle(self):
        """Verify temporal order maintained (no shuffle)."""
        # Create monotonically increasing data
        np.random.seed(42)
        X = pd.DataFrame({'feature': range(500)})
        y = pd.Series(range(500)) > 250  # Split at midpoint
        
        validator = WalkForwardValidator(
            initial_train_size=200,
            test_size=50,
            step_size=50
        )
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = validator.validate(model, X, y)
        
        # Check first fold
        first_fold = results.iloc[0]
        assert first_fold['train_start'] == 0
        assert first_fold['train_end'] == 199
        assert first_fold['test_start'] == 200
        assert first_fold['test_end'] == 249
    
    def test_scaler_fitted_on_train_only(self):
        """CRITICAL: Verify scaler fit() called only with training data."""
        # This test catches data leakage (Pitfall 1 from research)
        
        # Create data where test set has different distribution
        np.random.seed(42)
        X_train_data = np.random.randn(300, 5)  # Mean 0, std 1
        X_test_data = np.random.randn(100, 5) * 10 + 50  # Mean 50, std 10
        X_combined = np.vstack([X_train_data, X_test_data])
        
        X = pd.DataFrame(X_combined)
        y = pd.Series([0] * 300 + [1] * 100)
        
        # Mock StandardScaler to track fit() calls
        with patch('trading.ml.evaluation.walk_forward.StandardScaler') as MockScaler:
            mock_scaler_instance = Mock()
            mock_scaler_instance.fit.return_value = mock_scaler_instance
            mock_scaler_instance.transform.side_effect = lambda x: x  # Pass through
            MockScaler.return_value = mock_scaler_instance
            
            validator = WalkForwardValidator(
                initial_train_size=300,
                test_size=100,
                step_size=100
            )
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            
            try:
                results = validator.validate(model, X, y)
            except:
                pass  # We're just checking fit() calls
            
            # Verify fit() was called
            assert mock_scaler_instance.fit.called, "Scaler fit() was not called"
            
            # Get the data passed to fit()
            fit_call_args = mock_scaler_instance.fit.call_args[0][0]
            
            # Verify fit() was called with training data only (300 samples)
            # NOT with combined data (400 samples)
            assert len(fit_call_args) == 300, \
                f"Scaler fitted on {len(fit_call_args)} samples, expected 300 (training only)"
    
    def test_fold_count(self):
        """Verify correct number of folds based on window sizes."""
        validator = WalkForwardValidator(
            initial_train_size=252,
            test_size=60,
            step_size=30
        )
        
        # Test with 1000 samples
        expected_folds = validator.get_split_count(1000)
        
        # Calculate manually:
        # First fold: train=0-251, test=252-311
        # Second fold: train=0-281, test=282-341
        # Third fold: train=0-311, test=312-371
        # ...
        # Continue until train_end + test_size > 1000
        
        # train_end starts at 252, increases by 30 each time
        # Stops when train_end + 60 > 1000, i.e., train_end > 940
        # Number of steps: (940 - 252) / 30 + 1 = 23.93... → 24 folds
        
        assert expected_folds == 24, f"Expected 24 folds, got {expected_folds}"
    
    def test_results_structure(self):
        """Verify returned DataFrame has all required columns."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(500, 10))
        y = pd.Series(np.random.randint(0, 2, 500))
        
        validator = WalkForwardValidator(
            initial_train_size=200,
            test_size=50,
            step_size=50
        )
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = validator.validate(model, X, y)
        
        # Check required columns
        required_columns = [
            'fold', 'train_start', 'train_end', 'test_start', 'test_end',
            'accuracy', 'precision', 'recall', 'predictions', 'actuals'
        ]
        for col in required_columns:
            assert col in results.columns, f"Missing column: {col}"
        
        # Check fold numbering
        assert results['fold'].iloc[0] == 1
        assert results['fold'].is_monotonic_increasing
    
    def test_insufficient_data(self):
        """Verify error raised when insufficient data."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(100, 10))  # Only 100 samples
        y = pd.Series(np.random.randint(0, 2, 100))
        
        validator = WalkForwardValidator(
            initial_train_size=252,  # Requires 252 + 60 = 312 samples
            test_size=60,
            step_size=30
        )
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        with pytest.raises(ValueError, match="Insufficient data"):
            validator.validate(model, X, y)
