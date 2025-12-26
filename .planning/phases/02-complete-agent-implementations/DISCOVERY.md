# Phase 2 Discovery: TA-Lib & LightGBM Integration

**Research Date:** 2025-12-26
**Phase:** 02-complete-agent-implementations
**Scope:** Technical indicator calculation and ML model integration

## Research Summary

This discovery confirms the current API patterns for TA-Lib (technical indicators) and LightGBM (ML predictions) integration in Python, both of which are partially imported but not actively used in the codebase.

## TA-Lib Integration (Technical Indicators)

### Current Status in Codebase
- **Imported:** Yes (`trading/agents/quant_analyst.py` imports talib)
- **Used:** No (returns hardcoded neutral signals)
- **Version:** ta-lib-python 0.4.x/0.5.x/0.6.x branches available for numpy 1/2 compatibility

### API Patterns (2025)

**RSI (Relative Strength Index):**
```python
import talib
import numpy as np

# Input: close prices as numpy array
close = np.array([100.0, 101.5, 99.8, ...])  # OHLCV close prices

# Calculate RSI with 14-period default
rsi = talib.RSI(close, timeperiod=14)

# Interpretation:
# RSI > 70 → Overbought (bearish signal)
# RSI < 30 → Oversold (bullish signal)
# RSI 40-60 → Neutral
```

**MACD (Moving Average Convergence Divergence):**
```python
# Returns 3 arrays: MACD line, signal line, histogram
macd, macdsignal, macdhist = talib.MACD(
    close,
    fastperiod=12,    # Fast EMA period
    slowperiod=26,    # Slow EMA period
    signalperiod=9    # Signal line period
)

# Interpretation:
# MACD > signal → Bullish momentum
# MACD < signal → Bearish momentum
# macdhist crossover → Trend change
```

**Bollinger Bands:**
```python
# Returns 3 arrays: upper band, middle band, lower band
upperband, middleband, lowerband = talib.BBANDS(
    close,
    timeperiod=20,    # Moving average period
    nbdevup=2,        # Upper band std deviations
    nbdevdn=2,        # Lower band std deviations
    matype=0          # MA type (0=SMA)
)

# Interpretation:
# Price near upper band → Overbought
# Price near lower band → Oversold
# Band squeeze → Volatility contraction (breakout likely)
```

### Integration Considerations

**Data Format:**
- TA-Lib requires numpy arrays (not lists or pandas Series directly)
- Close prices extracted from OHLCV data: `close_prices = np.array([candle.close for candle in ohlcv])`

**Minimum Data Requirements:**
- RSI: Needs at least 14 periods for default calculation
- MACD: Needs at least 26 periods (slow period)
- Bollinger Bands: Needs at least 20 periods
- **Action:** Verify DataSyncAgent fetches sufficient historical data (currently fetches limit=100, should be safe)

**NaN Handling:**
- Early values may be NaN (insufficient data for calculation)
- **Pattern:** Use `np.isnan()` check or take last valid value with `rsi[-1]` after removing NaNs

**Performance:**
- TA-Lib uses Cython/Numpy → 2-4x faster than SWIG interface
- All calculations are vectorized (fast for 100-1000 data points)

### Recommendation

**Approach:** Direct TA-Lib integration in QuantAnalystAgent

**Implementation:**
1. Extract close prices from OHLCV context as numpy array
2. Calculate RSI, MACD, Bollinger Bands using TA-Lib functions
3. Return structured signals (not just neutral 0.0)
4. Handle NaN values for early periods

**Example Signal Structure:**
```python
return {
    'rsi': {
        'value': rsi[-1],
        'signal': 'overbought' if rsi[-1] > 70 else 'oversold' if rsi[-1] < 30 else 'neutral'
    },
    'macd': {
        'macd': macd[-1],
        'signal': macdsignal[-1],
        'histogram': macdhist[-1],
        'signal': 'bullish' if macd[-1] > macdsignal[-1] else 'bearish'
    },
    'bollinger': {
        'upper': upperband[-1],
        'middle': middleband[-1],
        'lower': lowerband[-1],
        'price_position': 'upper' if close[-1] > middleband[-1] else 'lower'
    }
}
```

## LightGBM Integration (ML Predictions)

### Current Status in Codebase
- **Imported:** Yes (`trading/ml/ensemble.py` imports lightgbm)
- **Used:** No (PredictAgent returns hardcoded 0.0 confidence)
- **Model files:** Not found (no saved models in repo)

### API Patterns (2025)

**Training Pattern:**
```python
import lightgbm as lgb
import numpy as np

# Prepare training data
X_train = np.array([...])  # Features (technical indicators, price data)
y_train = np.array([...])  # Labels (1=price_up, 0=price_down or continuous returns)

# Create LightGBM dataset (memory efficient)
train_data = lgb.Dataset(X_train, label=y_train)

# Configure parameters
params = {
    'objective': 'binary',  # or 'regression' for continuous predictions
    'metric': 'binary_logloss',  # or 'rmse' for regression
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1  # Suppress output
}

# Train with early stopping
bst = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[valid_data],  # Validation set for early stopping
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# Save model
bst.save_model('model.txt')
```

**Prediction Pattern:**
```python
# Load trained model
bst = lgb.Booster(model_file='model.txt')

# Prepare features (same structure as training)
X_test = np.array([[rsi, macd, bb_position, ...]])

# Get prediction (use best iteration from early stopping)
prediction = bst.predict(X_test, num_iteration=bst.best_iteration)[0]

# For binary classification:
# prediction = probability of class 1 (price will go up)
# confidence = abs(prediction - 0.5) * 2  # Scale to [0, 1]

# For regression:
# prediction = expected return
# confidence = model's uncertainty estimate
```

### Integration Considerations

**Features Engineering:**
- **Inputs:** Technical indicators (RSI, MACD, Bollinger Bands from QuantAnalyst)
- **Additional:** Price momentum, volume changes, volatility metrics
- **Structure:** Fixed feature vector (must match training schema)

**Training Data Source:**
- **Option 1:** CCXT historical fetch (on-demand, slower)
- **Option 2:** Pre-downloaded datasets (faster, requires setup)
- **Recommendation:** Start with CCXT historical fetch for flexibility

**Model Lifecycle:**
- **Initial training:** One-time setup, requires historical data
- **Retraining:** Periodic (daily/weekly) with new market data
- **Prediction:** Real-time, uses pre-trained model loaded at startup

**Categorical Features:**
- LightGBM handles categorical features natively (8x faster than one-hot encoding)
- Example: market regime ('bull', 'bear', 'sideways') as categorical
- **Pattern:** Convert to int before creating Dataset

**Performance Optimization:**
- Use `num_threads` parameter for multi-core processing
- LightGBM is memory efficient (discrete bins instead of raw data)
- Early stopping prevents overfitting and reduces training time

**Model Persistence:**
- Save model with `bst.save_model('model.txt')` after training
- Load at PredictAgent initialization: `self.model = lgb.Booster(model_file='...')`
- **Location:** Store in `trading/ml/models/` directory (add to .gitignore)

### Recommendation

**Approach:** Phased LightGBM integration

**Phase 1 (Minimum Viable):**
1. Create simple training script using CCXT historical data
2. Train binary classifier (price_up vs price_down over next N candles)
3. Save model to `trading/ml/models/lgbm_predictor.txt`
4. Load model in PredictAgent.__init__
5. Extract features from context (QuantAnalyst outputs)
6. Return real predictions instead of hardcoded 0.0

**Phase 2 (Future Enhancement):**
- Ensemble with XGBoost/PyTorch models (code already structured for this)
- Periodic retraining automation
- Confidence calibration
- Feature importance analysis

**Example Prediction Flow:**
```python
class PredictAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        # Load pre-trained model at initialization
        model_path = 'trading/ml/models/lgbm_predictor.txt'
        self.model = lgb.Booster(model_file=model_path) if os.path.exists(model_path) else None

    async def execute(self, context):
        if not self.model:
            return {'prediction': 0.0, 'confidence': 0.0}  # Fallback

        # Extract features from context
        quant_signals = context.get('quant_analyst', {})
        features = np.array([[
            quant_signals['rsi']['value'],
            quant_signals['macd']['histogram'],
            # ... more features
        ]])

        # Predict
        prob_up = self.model.predict(features, num_iteration=self.model.best_iteration)[0]

        # Convert to confidence score
        direction = 1 if prob_up > 0.5 else -1
        confidence = abs(prob_up - 0.5) * 2  # Scale to [0, 1]

        return {
            'prediction': direction,
            'confidence': confidence,
            'probability_up': prob_up
        }
```

## Implementation Priority

**Critical Path:**
1. TA-Lib indicators in QuantAnalystAgent (foundation for all other agents)
2. LightGBM model training script (one-time setup)
3. LightGBM predictions in PredictAgent (uses QuantAnalyst outputs)
4. Enhanced Bull/Bear agents using TA-Lib signals (remove 2% momentum hack)

**Dependency Chain:**
- Bull/Bear depend on QuantAnalyst providing real signals
- PredictAgent depends on QuantAnalyst for feature engineering
- All three must be completed together for coherent decisions

## Risks & Mitigations

**Risk 1: Insufficient Training Data**
- **Issue:** LightGBM needs substantial historical data for reliable predictions
- **Mitigation:** Start with at least 1000 historical candles (CCXT fetch)
- **Fallback:** Use simpler rule-based predictions if training fails

**Risk 2: NaN Values in Indicators**
- **Issue:** Early periods don't have enough data for TA-Lib calculations
- **Mitigation:** Check for NaN, use only last valid value, or skip trading if insufficient data

**Risk 3: Model Drift**
- **Issue:** Market conditions change, model becomes stale
- **Mitigation:** Accept initial version won't be perfect, plan for periodic retraining in Phase 4

**Risk 4: Feature Mismatch**
- **Issue:** Prediction features don't match training features (schema drift)
- **Mitigation:** Document feature schema, validate before prediction

## Success Criteria

✅ **Discovery Complete When:**
- [x] TA-Lib API patterns confirmed (RSI, MACD, Bollinger Bands)
- [x] LightGBM training/prediction patterns confirmed
- [x] Integration approach defined (direct TA-Lib, phased LightGBM)
- [x] Risks identified with mitigations
- [x] Code examples provided for both libraries

**Ready to proceed with planning.**

---

## Sources

- [TA-Lib Python Documentation](https://ta-lib.github.io/ta-lib-python/)
- [TA-Lib GitHub Repository](https://github.com/TA-Lib/ta-lib-python)
- [Applying RSI, MACD, and Bollinger Bands with TA-Lib - Sling Academy](https://www.slingacademy.com/article/applying-rsi-macd-and-bollinger-bands-with-ta-lib/)
- [LightGBM Train API Documentation](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html)
- [LightGBM Python Introduction](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html)
- [How to master LightGBM to efficiently make predictions](https://data-ai.theodo.com/en/technical-blog/master-lightgbm-efficiently-to-make-predictions)
