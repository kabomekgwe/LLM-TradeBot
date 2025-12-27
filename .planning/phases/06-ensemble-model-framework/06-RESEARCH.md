# Phase 6: Ensemble Model Framework - Research

**Phase:** 06-ensemble-model-framework
**Researched:** 2025-12-27
**Context:** [06-CONTEXT.md](06-CONTEXT.md)

## Overview

This research covers the ecosystem for implementing a production-ready ensemble model framework that combines LightGBM, XGBoost, and Random Forest with intelligent regime-based strategy switching. The framework must support multiple combination strategies (voting, stacking, dynamic selection) and automatically adapt based on volatility regimes detected by Phase 5's HMM.

---

## 1. Standard Stack (Recommended Technologies)

### Core Ensemble Framework
- **scikit-learn 1.8.0** - Production-ready ensemble infrastructure
  - `VotingClassifier` - Hard/soft voting for ensemble combination
  - `StackingClassifier` - Meta-model learns optimal combination with cross-validation
  - `RandomForestClassifier` - One of three tree-based models
  - Native cross-validation support prevents information leakage

### Gradient Boosting Libraries
- **XGBoost 3.1.1+** - Robust regularization, excellent accuracy
  - Python API: `xgboost.XGBClassifier` (scikit-learn compatible)
  - Key parameters: `n_estimators`, `max_depth`, `learning_rate`, `objective`
  - Native model I/O: `save_model()` / `load_model()` (JSON format)

- **LightGBM 4.x** (already integrated) - Fast, memory-efficient
  - 2-5x faster training than XGBoost
  - 40-60% lower memory usage
  - Histogram-based algorithm for large datasets

- **Random Forest (scikit-learn)** - Decorrelated trees, non-linear patterns
  - `sklearn.ensemble.RandomForestClassifier`
  - Naturally handles high-dimensional features
  - Resilient to noise in financial data

### Model Persistence (SECURITY CRITICAL)

**⚠️ SECURITY WARNING:** Never use pickle for model serialization in production. Pickle can execute arbitrary code when loading untrusted content.

**Safe serialization approaches:**

- **XGBoost models:** Use native JSON format ONLY
  ```python
  # ✅ SAFE: Native XGBoost JSON format
  model.save_model('model.json')
  model.load_model('model.json')

  # ❌ DANGEROUS: Never use pickle for XGBoost
  # pickle.dump(xgb_model, file)  # DO NOT DO THIS
  ```

- **LightGBM models:** Use native text format
  ```python
  # ✅ SAFE: Native LightGBM text format
  model.booster_.save_model('model.txt')
  lgb.Booster(model_file='model.txt')
  ```

- **scikit-learn models (Random Forest):** Use joblib for numpy efficiency
  ```python
  # ✅ ACCEPTABLE: joblib for sklearn models only
  # Note: joblib uses pickle internally but is standard for sklearn
  import joblib
  joblib.dump(rf_model, 'rf_model.joblib')
  rf_model = joblib.load('rf_model.joblib')
  ```

**Why native formats are required:**
- XGBoost JSON format guarantees cross-version compatibility
- No arbitrary code execution risk
- Faster loading and smaller file sizes
- Industry best practice for production systems

### Supporting Libraries
- **pandas-ta** (Phase 5 dependency) - Feature engineering already migrated
- **statsmodels** (Phase 5 dependency) - MarkovRegression for regime detection
- **numpy, pandas** - Core data manipulation

---

## 2. Architecture Patterns

### Ensemble Combination Strategies

#### **Voting (sklearn.ensemble.VotingClassifier)**
```python
from sklearn.ensemble import VotingClassifier

# Hard voting: Majority rule (class labels)
voting_clf = VotingClassifier(
    estimators=[
        ('lgbm', lgbm_model),
        ('xgb', xgb_model),
        ('rf', rf_model)
    ],
    voting='hard'  # or 'soft' for probability averaging
)

# Soft voting: Average predicted probabilities (recommended for calibrated models)
voting_clf = VotingClassifier(
    estimators=[...],
    voting='soft'  # Uses predict_proba() from base models
)
```

**When to use:**
- High volatility regimes (robust to noise, majority rule)
- Base models are well-calibrated
- Simple, interpretable combination

**Pros:**
- Reduces variance through averaging
- Simple to implement and explain
- No additional training required

**Cons:**
- Assumes equal model importance
- Cannot learn non-linear combinations

#### **Stacking (sklearn.ensemble.StackingClassifier)**
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Stacking with cross-validated predictions
stacking_clf = StackingClassifier(
    estimators=[
        ('lgbm', lgbm_model),
        ('xgb', xgb_model),
        ('rf', rf_model)
    ],
    final_estimator=LogisticRegression(),  # Meta-model
    cv=5,  # Cross-validation prevents information leakage
    stack_method='predict_proba'  # Use probabilities instead of class labels
)
```

**When to use:**
- Low volatility regimes (learns optimal combination)
- Models have complementary strengths
- Want to learn non-linear combinations

**Pros:**
- Meta-model learns optimal weighting
- Cross-validation prevents overfitting
- Can capture complex interactions

**Cons:**
- More complex, harder to interpret
- Requires additional training
- Risk of overfitting if not cross-validated

#### **Blending (Manual Implementation)**
```python
# Blending: Holdout set approach
X_train, X_blend, y_train, y_blend = train_test_split(X, y, test_size=0.2)

# Train base models on X_train
lgbm_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Get predictions on holdout set (X_blend)
blend_features = np.column_stack([
    lgbm_model.predict_proba(X_blend)[:, 1],
    xgb_model.predict_proba(X_blend)[:, 1],
    rf_model.predict_proba(X_blend)[:, 1]
])

# Train meta-model on blend set
meta_model = LogisticRegression()
meta_model.fit(blend_features, y_blend)
```

**When to use:**
- Faster than stacking (single holdout set vs cross-validation)
- Less prone to overfitting than stacking
- Resource-constrained environments

**Pros:**
- Simpler than stacking (no cross-validation)
- 8 seconds vs 42 seconds training time (blending vs stacking benchmarks)
- Prevents information leakage

**Cons:**
- Uses less data for training (holdout set)
- Less robust than stacking (single fold vs k-fold)

#### **Dynamic Selection (Custom Implementation)**
```python
class DynamicEnsemble:
    """Select best recent performer based on validation window."""

    def __init__(self, models, validation_window=50):
        self.models = models  # {'lgbm': model, 'xgb': model, 'rf': model}
        self.validation_window = validation_window
        self.performance_tracker = {name: [] for name in models.keys()}

    def update_performance(self, y_true, predictions):
        """Track model accuracy over sliding window."""
        for name, pred in predictions.items():
            accuracy = accuracy_score(y_true[-self.validation_window:],
                                     pred[-self.validation_window:])
            self.performance_tracker[name].append(accuracy)

    def predict(self, X):
        """Select model with best recent performance."""
        recent_performance = {
            name: np.mean(scores[-10:])  # Last 10 accuracy scores
            for name, scores in self.performance_tracker.items()
        }
        best_model_name = max(recent_performance, key=recent_performance.get)
        return self.models[best_model_name].predict(X)
```

**When to use:**
- Transitional volatility regimes (regime is changing)
- Models have varying performance across conditions
- Want adaptive selection without meta-learning

**Pros:**
- Adapts to changing market conditions
- No meta-model training required
- Can detect model degradation

**Cons:**
- Requires performance tracking infrastructure
- May be unstable during rapid regime transitions
- Single model selection (doesn't combine predictions)

### Regime-Aware Strategy Switching

```python
class RegimeAwareEnsemble:
    """Switch ensemble strategies based on volatility regime."""

    def __init__(self, base_models, regime_detector):
        self.base_models = base_models
        self.regime_detector = regime_detector

        # Create ensemble strategies
        self.voting_ensemble = VotingClassifier(
            estimators=list(base_models.items()),
            voting='soft'
        )
        self.stacking_ensemble = StackingClassifier(
            estimators=list(base_models.items()),
            final_estimator=LogisticRegression(),
            cv=5
        )
        self.dynamic_ensemble = DynamicEnsemble(base_models)

    def predict(self, X, df_with_regime):
        """Select strategy based on current regime."""
        # Get current regime from Phase 5's HMM detector
        current_regime = df_with_regime['current_regime'].iloc[-1]
        is_low_vol = df_with_regime['is_low_volatility'].iloc[-1]
        regime_prob = df_with_regime[f'regime_prob_{current_regime}'].iloc[-1]

        # Strategy selection logic
        if is_low_vol and regime_prob > 0.7:
            # Low volatility, confident regime detection → Stacking
            strategy = 'stacking'
            prediction = self.stacking_ensemble.predict(X)
        elif regime_prob < 0.6:
            # Transitional regime (low confidence) → Dynamic selection
            strategy = 'dynamic'
            prediction = self.dynamic_ensemble.predict(X)
        else:
            # High volatility or default → Voting
            strategy = 'voting'
            prediction = self.voting_ensemble.predict(X)

        return prediction, strategy, {
            'regime': current_regime,
            'is_low_vol': is_low_vol,
            'regime_prob': regime_prob
        }
```

### Feature Importance Aggregation

```python
def aggregate_feature_importance(models, feature_names):
    """
    Aggregate feature importance across all models.

    Uses WISFC (Weighted Importance Score and Frequency Count):
    - Combines importance magnitude and consistency
    - Weights features by rank and frequency across models
    """
    importance_dict = {}

    for model_name, model in models.items():
        # Extract feature importance (model-specific)
        if hasattr(model, 'feature_importances_'):  # XGBoost, LightGBM, Random Forest
            importances = model.feature_importances_
        else:
            raise ValueError(f"Model {model_name} does not support feature importances")

        # Store with model name
        importance_dict[model_name] = dict(zip(feature_names, importances))

    # Aggregate using WISFC approach
    aggregated = {}
    for feature in feature_names:
        # Get importance from each model
        feature_importances = [
            importance_dict[model][feature]
            for model in importance_dict.keys()
        ]

        # Average importance across models (simple aggregation)
        aggregated[feature] = np.mean(feature_importances)

    # Sort by aggregated importance
    return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))
```

**Advanced WISFC Framework:**
- Weighted Importance Score and Frequency Count (2025 research)
- Combines ranked outputs from diverse models
- Assigns weighted score based on rank and frequency
- More robust than simple averaging
- Addresses explanation disagreement across models

### Model Serialization Pattern (SECURITY HARDENED)

```python
import joblib
from pathlib import Path

class EnsembleModelPersistence:
    """
    Save/load ensemble models with version tracking.

    SECURITY: Uses native formats for XGBoost/LightGBM to avoid
    arbitrary code execution risks.
    """

    def __init__(self, model_dir='models/ensemble'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_models(self, models, metadata):
        """
        Save models with native formats + metadata.

        SECURITY CRITICAL:
        - XGBoost: Native JSON format (NOT serialization libraries)
        - LightGBM: Native text format (NOT serialization libraries)
        - Random Forest: joblib (standard for sklearn, acceptable risk)
        """
        # XGBoost: Use native JSON API (REQUIRED for security)
        models['xgb'].save_model(str(self.model_dir / 'xgboost_model.json'))

        # LightGBM: Use native text API (REQUIRED for security)
        models['lgbm'].booster_.save_model(str(self.model_dir / 'lightgbm_model.txt'))

        # Random Forest: Use joblib (standard for sklearn)
        joblib.dump(models['rf'], self.model_dir / 'random_forest_model.joblib')

        # Save metadata (versions, feature names, training date)
        joblib.dump({
            'xgboost_version': xgboost.__version__,
            'lightgbm_version': lightgbm.__version__,
            'sklearn_version': sklearn.__version__,
            'feature_names': metadata['feature_names'],
            'trained_date': metadata['trained_date'],
            'training_samples': metadata['training_samples']
        }, self.model_dir / 'metadata.joblib')

    def load_models(self):
        """Load models with version verification."""
        import xgboost as xgb
        import lightgbm as lgb

        # Load XGBoost from JSON (safe, no code execution)
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(str(self.model_dir / 'xgboost_model.json'))

        # Load LightGBM from text (safe, no code execution)
        lgbm_model = lgb.Booster(model_file=str(self.model_dir / 'lightgbm_model.txt'))

        # Load Random Forest (joblib standard for sklearn)
        rf_model = joblib.load(self.model_dir / 'random_forest_model.joblib')

        # Load metadata
        metadata = joblib.load(self.model_dir / 'metadata.joblib')

        return {
            'xgb': xgb_model,
            'lgbm': lgbm_model,
            'rf': rf_model
        }, metadata
```

---

## 3. What NOT to Hand-Roll

### ❌ DO NOT Hand-Roll Ensemble Combination Logic
**Use:** `sklearn.ensemble.VotingClassifier`, `sklearn.ensemble.StackingClassifier`

**Why:**
- Battle-tested implementations with proper cross-validation
- Handles edge cases (missing predictions, probability calibration)
- VotingClassifier/StackingClassifier prevent information leakage
- Stacking uses nested k-fold cross-validation automatically

**Red flags:**
- Manually averaging predictions without cross-validation
- Implementing custom voting logic
- Combining predictions without holdout/CV strategy

### ❌ DO NOT Hand-Roll Model Evaluation Metrics
**Use:** `sklearn.metrics` (accuracy_score, roc_auc_score, classification_report)

**Why:**
- Standard implementations used across industry
- Consistent metric definitions
- Prevents calculation bugs

**Red flags:**
- Reimplementing accuracy, precision, recall from scratch
- Custom metric formulas without validation

### ❌ DO NOT Hand-Roll Feature Importance Calculations
**Use:** `model.feature_importances_` (XGBoost, LightGBM, Random Forest)

**Why:**
- Native implementations are optimized and correct
- Consistent across tree-based models
- WISFC aggregation framework for combining (research-backed)

**Red flags:**
- Manually calculating Gini importance
- Custom permutation importance without proper validation

### ❌ DO NOT Use Unsafe Serialization for XGBoost/LightGBM
**Use:** Native XGBoost JSON API (`save_model()`, `load_model()`)
**Use:** Native LightGBM text API (`booster_.save_model()`)

**Why (SECURITY CRITICAL):**
- Serialization libraries can execute arbitrary code when loading
- JSON/text formats are safe (no code execution)
- Cross-version compatibility guaranteed
- Industry security best practice

**Red flags:**
- Using any serialization library for XGBoost/LightGBM models
- Loading models from untrusted sources
- Not validating model file integrity

### ✅ CAN Build Custom (But Validate Thoroughly)

**Regime-aware strategy switching** - No existing library, custom logic required:
- Decision tree mapping: regime → strategy selection
- Integration with Phase 5's HMM regime detector
- Logging which strategy is active and why

**Dynamic model selection** - No sklearn equivalent:
- Performance tracking over sliding window
- Best recent performer selection
- Graceful degradation handling

**Observability/transparency layer** - Domain-specific:
- Logging individual model predictions
- DecisionContext correlation tracking
- Feature importance aggregation reporting

---

## 4. Common Pitfalls

### Ensemble Overfitting

**Problem:** Stacking can overfit if not properly cross-validated

**Prevention:**
```python
# ✅ GOOD: Use cross-validation in StackingClassifier
stacking_clf = StackingClassifier(
    estimators=[...],
    cv=5,  # Prevents information leakage
    stack_method='predict_proba'
)

# ❌ BAD: Train meta-model on same data as base models
# This causes severe overfitting and inflated performance metrics
```

**Best practices:**
- Always use `cv` parameter in StackingClassifier
- Use repeated k-fold CV for robust estimates
- Monitor validation set performance separately
- Early stopping for meta-model if using gradient boosting

### Model Correlation (Lack of Diversity)

**Problem:** If all models make similar errors, ensemble provides no benefit

**Detection:**
```python
from sklearn.metrics import matthews_corrcoef

# Check prediction correlation between models
predictions = {
    'lgbm': lgbm_model.predict(X_val),
    'xgb': xgb_model.predict(X_val),
    'rf': rf_model.predict(X_val)
}

# Correlation matrix
correlation = {}
for m1 in predictions:
    for m2 in predictions:
        if m1 < m2:  # Avoid duplicates
            corr = matthews_corrcoef(predictions[m1], predictions[m2])
            correlation[f'{m1}_vs_{m2}'] = corr
            print(f"{m1} vs {m2}: {corr:.3f}")

# Target: correlation < 0.8 (models are sufficiently diverse)
```

**Solutions:**
- Use different model types (gradient boosting vs random forest)
- Vary hyperparameters significantly
- Use different feature subsets per model
- Ensemble benefits from uncorrelated base models

### Regime Detection Bugs

**Problem:** Incorrect regime classification breaks strategy switching

**Prevention:**
```python
# Validate regime detection before using in ensemble
def validate_regime_detection(df):
    """Sanity checks for regime detector output."""
    # Check: regime probabilities sum to 1
    prob_sum = df['regime_prob_0'] + df['regime_prob_1']
    assert np.allclose(prob_sum, 1.0), "Regime probabilities don't sum to 1"

    # Check: current_regime matches highest probability
    expected_regime = df[['regime_prob_0', 'regime_prob_1']].idxmax(axis=1)
    expected_regime = expected_regime.str.replace('regime_prob_', '').astype(int)
    assert (df['current_regime'] == expected_regime).all(), "Regime mismatch"

    # Check: is_low_volatility consistent with regime variances
    # (requires access to regime detector's variance estimates)

    return True
```

**Best practices:**
- Add unit tests for regime → strategy mapping
- Log regime confidence scores (probabilities)
- Validate regime detector output before strategy selection
- Handle edge cases (equal probabilities, regime transitions)

### Hyperparameter Interaction

**Problem:** Individual model hyperparameters were tuned separately, but ensemble performance suffers

**Example:**
- XGBoost tuned for max accuracy → overfits
- LightGBM tuned for speed → underfits
- Random Forest default params → mediocre
- Ensemble combines overfit + underfit + mediocre = poor performance

**Solution:**
```python
# Joint hyperparameter optimization (optional, not in scope boundaries)
from sklearn.model_selection import GridSearchCV

# Define parameter grid for ALL models
param_grid = {
    'lgbm__n_estimators': [100, 200],
    'lgbm__max_depth': [5, 10],
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 5],
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 20]
}

# Search over ensemble combinations
grid_search = GridSearchCV(
    VotingClassifier(estimators=[...]),
    param_grid,
    cv=5,
    scoring='roc_auc'
)
```

**Phase 6 approach:**
- Use reasonable defaults from official docs
- Focus on ensemble architecture, not squeezing every 0.1% accuracy
- Hyperparameter tuning is acceptable but NOT marked out-of-scope

### Random Forest Time Series Issues

**Problem:** Random Forest struggles with trends, cannot extrapolate

**Solutions (from 2025 research):**
- Create lag features manually (already done in Phase 5)
- Apply differencing/transformations (ARIMA-style preprocessing)
- Use walk-forward validation (NOT k-fold CV for time series)
- Leverage Random Forest's noise resilience (financial data is mostly noise)

**Phase 6 benefit:**
- Ensemble with LightGBM/XGBoost compensates for RF's trend weakness
- Model diversity improves robustness

### XGBoost vs LightGBM Trade-offs

**LightGBM strengths:**
- 2-5x faster training
- 40-60% lower memory usage
- Better for high-dimensional, sparse datasets

**XGBoost strengths:**
- "Extra edge in accuracy"
- More robust regularization
- Better for smaller datasets where interpretability matters

**Phase 6 strategy:**
- Keep both models in ensemble (complementary strengths)
- LightGBM for speed, XGBoost for accuracy, Random Forest for diversity
- Ensemble leverages all strengths while mitigating individual weaknesses

---

## 5. Code Examples (From Official Docs)

### XGBoost Basic Setup
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create XGBoost classifier (scikit-learn API)
xgb_model = xgb.XGBClassifier(
    n_estimators=100,      # Number of trees
    max_depth=5,           # Tree depth
    learning_rate=0.1,     # Step size shrinkage
    objective='binary:logistic',  # Binary classification
    eval_metric='logloss',
    random_state=42
)

# Train
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,  # Stop if no improvement
    verbose=False
)

# Predict
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)

# Feature importance
importance = xgb_model.feature_importances_
```

### Ensemble with Regime Switching (Full Example)
```python
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import numpy as np

class ProductionEnsemble:
    """
    Production-ready ensemble with regime-aware strategy switching.

    Integrates with Phase 5's HMM regime detector to automatically
    select ensemble strategy based on current market volatility.
    """

    def __init__(self, n_estimators=100):
        # Initialize base models
        self.base_models = {
            'lgbm': lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                learning_rate=0.1,
                objective='binary:logistic',
                random_state=42
            ),
            'rf': RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=20,
                random_state=42
            )
        }

        # Create ensemble strategies
        self.voting_ensemble = VotingClassifier(
            estimators=list(self.base_models.items()),
            voting='soft'  # Average probabilities
        )

        self.stacking_ensemble = StackingClassifier(
            estimators=list(self.base_models.items()),
            final_estimator=LogisticRegression(),
            cv=5,  # Prevent overfitting
            stack_method='predict_proba'
        )

        self.current_strategy = None
        self.strategy_history = []

    def fit(self, X_train, y_train):
        """Train all ensemble strategies."""
        # Train voting ensemble
        self.voting_ensemble.fit(X_train, y_train)

        # Train stacking ensemble (includes cross-validation)
        self.stacking_ensemble.fit(X_train, y_train)

        # Train base models for dynamic selection
        for name, model in self.base_models.items():
            model.fit(X_train, y_train)

        return self

    def predict_with_regime(self, X, current_regime_info):
        """
        Predict using regime-aware strategy selection.

        Args:
            X: Features for prediction
            current_regime_info: Dict with keys:
                - 'current_regime': 0 or 1
                - 'is_low_volatility': 0 or 1
                - 'regime_prob_0': float [0, 1]
                - 'regime_prob_1': float [0, 1]

        Returns:
            prediction: Binary class prediction
            metadata: Dict with strategy, regime info, individual predictions
        """
        is_low_vol = current_regime_info['is_low_volatility']
        regime = current_regime_info['current_regime']
        regime_prob = current_regime_info[f'regime_prob_{regime}']

        # Get individual model predictions
        individual_predictions = {}
        individual_probabilities = {}
        for name, model in self.base_models.items():
            individual_predictions[name] = model.predict(X)[0]
            individual_probabilities[name] = model.predict_proba(X)[0, 1]

        # Strategy selection based on regime
        if is_low_vol and regime_prob > 0.7:
            # Low volatility, confident → Stacking
            strategy = 'stacking'
            prediction = self.stacking_ensemble.predict(X)[0]
            prediction_proba = self.stacking_ensemble.predict_proba(X)[0, 1]
            reason = f"Low volatility regime (prob={regime_prob:.2f}) → meta-model learns combination"

        elif regime_prob < 0.6:
            # Transitional regime → Dynamic selection (best recent performer)
            strategy = 'dynamic'
            # For simplicity, use model with highest individual probability
            best_model = max(individual_probabilities, key=individual_probabilities.get)
            prediction = individual_predictions[best_model]
            prediction_proba = individual_probabilities[best_model]
            reason = f"Transitional regime (prob={regime_prob:.2f}) → selected {best_model}"

        else:
            # High volatility or default → Voting
            strategy = 'voting'
            prediction = self.voting_ensemble.predict(X)[0]
            prediction_proba = self.voting_ensemble.predict_proba(X)[0, 1]
            reason = f"High volatility regime (prob={regime_prob:.2f}) → majority vote"

        # Track strategy history
        self.current_strategy = strategy
        self.strategy_history.append({
            'strategy': strategy,
            'regime': regime,
            'is_low_vol': is_low_vol,
            'regime_prob': regime_prob
        })

        # Metadata for observability
        metadata = {
            'strategy': strategy,
            'reason': reason,
            'regime': regime,
            'is_low_volatility': bool(is_low_vol),
            'regime_confidence': regime_prob,
            'individual_predictions': individual_predictions,
            'individual_probabilities': individual_probabilities,
            'ensemble_probability': prediction_proba
        }

        return prediction, metadata

    def get_feature_importance(self, feature_names):
        """Aggregate feature importance across all models."""
        importance_dict = {}

        for name, model in self.base_models.items():
            importance_dict[name] = dict(zip(feature_names, model.feature_importances_))

        # Average importance across models
        aggregated = {}
        for feature in feature_names:
            feature_importances = [
                importance_dict[model][feature]
                for model in importance_dict.keys()
            ]
            aggregated[feature] = np.mean(feature_importances)

        return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))

# Usage example
ensemble = ProductionEnsemble(n_estimators=100)
ensemble.fit(X_train, y_train)

# Predict with regime information (from Phase 5's regime detector)
regime_info = {
    'current_regime': 0,
    'is_low_volatility': 1,
    'regime_prob_0': 0.85,
    'regime_prob_1': 0.15
}

prediction, metadata = ensemble.predict_with_regime(X_test[0:1], regime_info)

print(f"Prediction: {prediction}")
print(f"Strategy: {metadata['strategy']}")
print(f"Reason: {metadata['reason']}")
print(f"Individual predictions: {metadata['individual_predictions']}")
print(f"Individual probabilities: {metadata['individual_probabilities']}")
```

---

## 6. Integration with Existing Infrastructure

### Phase 5 Integration Points

**Volatility Regime Detector:**
```python
from trading.features.regime import VolatilityRegimeDetector

# Already implemented in Phase 5
regime_detector = VolatilityRegimeDetector(n_regimes=2, min_periods=100)
df = regime_detector.detect_regimes(df, return_col='close')

# Available columns:
# - 'regime_prob_0': Probability of regime 0
# - 'regime_prob_1': Probability of regime 1
# - 'current_regime': 0 or 1
# - 'is_low_volatility': 0 or 1

# Pass to ensemble
regime_info = {
    'current_regime': df['current_regime'].iloc[-1],
    'is_low_volatility': df['is_low_volatility'].iloc[-1],
    'regime_prob_0': df['regime_prob_0'].iloc[-1],
    'regime_prob_1': df['regime_prob_1'].iloc[-1]
}
```

**Feature Engineering Pipeline:**
```python
from trading.ml.feature_engineering import FeatureEngineer

# Already implemented with 86 features
feature_engineer = FeatureEngineer()
df = await feature_engineer.engineer_features(df, exchange, symbol)

# Features available for all models:
# - 50 original features (momentum, technical, volatility, volume, etc.)
# - 11 sentiment features (Fear & Greed Index)
# - 11 temporal features (sessions, holidays, day-of-week)
# - 4 regime features (HMM probabilities)
# - 6 microstructure features (bid-ask spread, imbalance)
# - 4 price features (returns, log returns, price change)

# Extract feature matrix
feature_columns = feature_engineer.get_feature_columns()
X = df[feature_columns].values
```

### Phase 4 Integration Points

**Structured Logging:**
```python
from trading.logging_config import get_logger, DecisionContext

logger = get_logger(__name__)

# Use DecisionContext for correlation tracking
with DecisionContext() as decision_id:
    prediction, metadata = ensemble.predict_with_regime(X, regime_info)

    logger.info(
        "Ensemble prediction complete",
        extra={
            'decision_id': decision_id,
            'strategy': metadata['strategy'],
            'reason': metadata['reason'],
            'individual_predictions': metadata['individual_predictions'],
            'ensemble_prediction': int(prediction),
            'ensemble_probability': metadata['ensemble_probability']
        }
    )
```

**Exception Handling:**
```python
from trading.exceptions import AgentError, ModelError

try:
    prediction, metadata = ensemble.predict_with_regime(X, regime_info)
except ValueError as e:
    # Regime detection failed or invalid input
    raise AgentError(
        f"Ensemble prediction failed: {e}",
        agent_name="EnsemblePredictor",
        context={'regime_info': regime_info}
    )
except Exception as e:
    # Unexpected model error
    raise ModelError(
        f"Unexpected error in ensemble: {e}",
        model_name="RegimeAwareEnsemble"
    )
```

**Timeout Protection:**
```python
from trading.utils.timeout import with_timeout

class EnsembleAgent:
    @with_timeout(60.0)  # 60 seconds for ensemble prediction
    async def predict(self, X, regime_info):
        """Ensemble prediction with timeout."""
        prediction, metadata = self.ensemble.predict_with_regime(X, regime_info)
        return prediction, metadata
```

---

## 7. Performance Expectations (from 06-CONTEXT.md)

### Target Improvements
- **10-20% improvement** in prediction accuracy vs single LightGBM
- **Higher Sharpe ratio** in backtests due to more robust predictions
- **Reduced losses** during volatile market transitions (regime-aware switching)

### Benchmarking Strategy
```python
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Single model baseline (LightGBM only)
lgbm_baseline = lgb.LGBMClassifier(n_estimators=100)
lgbm_baseline.fit(X_train, y_train)
baseline_accuracy = accuracy_score(y_test, lgbm_baseline.predict(X_test))
baseline_auc = roc_auc_score(y_test, lgbm_baseline.predict_proba(X_test)[:, 1])

# Ensemble performance
ensemble = ProductionEnsemble()
ensemble.fit(X_train, y_train)
ensemble_predictions = []
for i in range(len(X_test)):
    regime_info = extract_regime_info(df_test.iloc[i])
    pred, _ = ensemble.predict_with_regime(X_test[i:i+1], regime_info)
    ensemble_predictions.append(pred)

ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
ensemble_auc = roc_auc_score(y_test, ensemble_predictions)

# Calculate improvement
accuracy_improvement = ((ensemble_accuracy - baseline_accuracy) / baseline_accuracy) * 100
auc_improvement = ((ensemble_auc - baseline_auc) / baseline_auc) * 100

print(f"Accuracy improvement: {accuracy_improvement:.1f}%")
print(f"AUC improvement: {auc_improvement:.1f}%")

# Target: 10-20% improvement
assert accuracy_improvement >= 10, f"Ensemble accuracy improvement too low: {accuracy_improvement:.1f}%"
```

---

## 8. Research Sources

### Official Documentation
- [XGBoost Python API Reference](https://xgboost.readthedocs.io/en/stable/python/python_api.html) - XGBoost 3.1.1 official docs
- [scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html) - VotingClassifier, StackingClassifier 1.8.0 docs
- [Introduction to Model IO — xgboost](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html) - Model serialization best practices

### Ensemble Strategies
- [Ensemble Learning: Stacking, Blending & Voting | Towards Data Science](https://towardsdatascience.com/ensemble-learning-stacking-blending-voting-b37737c4f483/) - Comprehensive comparison
- [Stacking vs Blending vs Voting Ensembles | Medium](https://medium.com/@pacosun/how-models-team-up-to-outsmart-errors-7ea40510c0f3) - Performance trade-offs

### Financial ML Research (2024-2025)
- [Machine Learning for Financial Prediction Under Regime Change Using Technical Analysis: A Systematic Review | ResearchGate](https://www.researchgate.net/publication/371875352_Machine_Learning_for_Financial_Prediction_Under_Regime_Change_Using_Technical_Analysis_A_Systematic_Review) - Regime switching ML
- [A Machine Learning Approach to Regime Modeling - Two Sigma](https://www.twosigma.com/articles/a-machine-learning-approach-to-regime-modeling/) - Industry best practices
- [Chapter 4: Ensemble Learning in Investment | CFA Institute](https://rpc.cfainstitute.org/research/foundation/2025/chapter-4-ensemble-learning-investment) - Ensemble applications in finance

### XGBoost vs LightGBM
- [LightGBM vs XGBoost 2025: Performance Benchmarks | Markaicode](https://markaicode.com/lightgbm-vs-xgboost-2025-performance-benchmarks/) - 2025 benchmarks
- [XGBoost vs LightGBM: How Are They Different | Neptune.ai](https://neptune.ai/blog/xgboost-vs-lightgbm) - Detailed comparison

### Random Forest for Time Series
- [Random Forest for Time Series Forecasting | MachineLearningMastery.com](https://machinelearningmastery.com/random-forest-for-time-series-forecasting/) - Best practices
- [Optimizing Random Forest Models for Stock Market Prediction | ACM](https://dl.acm.org/doi/10.1145/3745238.3745250) - Hyperparameter analysis

### Overfitting Prevention
- [Effective Overfitting Prevention Techniques | NumberAnalytics](https://www.numberanalytics.com/blog/effective-overfitting-prevention-techniques-ml-model-quality) - Cross-validation strategies
- [Cross Validation in Ensemble Techniques | FasterCapital](https://www.fastercapital.com/content/Cross-Validation--Validating-the-Future--Cross-Validation-in-Ensemble-Techniques.html) - Preventing ensemble overfitting

### Feature Importance
- [Synthesizing Explainability Across Multiple ML Models | MDPI](https://www.mdpi.com/1999-4893/18/6/368) - WISFC framework
- [The stability of different aggregation techniques in ensemble feature selection | Journal of Big Data](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00607-1) - Aggregation methods

### Model Serialization
- [How to Save and Load XGBoost Models | StackAbuse](https://stackabuse.com/bytes/how-to-save-and-load-xgboost-models/) - XGBoost persistence
- [Save Machine Learning Model Using Joblib | Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/08/quick-hacks-to-save-machine-learning-model-using-pickle-and-joblib/) - scikit-learn best practices

---

**Research complete.** Ready to create execution plan (06-01-PLAN.md).

---

*Phase: 06-ensemble-model-framework*
*Research gathered: 2025-12-27*
