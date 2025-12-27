# Phase 7 Context: Deep Learning Models

**Phase:** 7 of 8 (Deep Learning Models)
**Milestone:** v1.1 Advanced ML & Feature Engineering
**Created:** 2025-12-27
**Depends on:** Phase 6 (Ensemble Model Framework)

---

## Vision Summary

Build a **fully independent deep learning trading strategy** that runs in parallel to the existing ensemble system. The deep learning pipeline will implement both LSTM and Transformer architectures, manage its own portfolio separately, and use a hybrid training approach combining raw price sequences with engineered features.

---

## Core Goals

### 1. Independent Deep Learning Pipeline
- **Architecture**: Separate from existing ensemble framework
- **Operation**: Parallel execution with separate portfolio management
- **Integration**: No direct interaction with ensemble predictions
- **Risk Management**: Independent risk controls and position management

### 2. Dual Architecture Implementation
- **LSTM (Long Short-Term Memory)**:
  - Primary focus on temporal dependencies
  - Proven for financial time-series
  - Well-understood training dynamics

- **Transformer (Attention-based)**:
  - Self-attention mechanisms for complex patterns
  - State-of-art sequence modeling
  - Captures long-range dependencies

### 3. Hybrid Training Data Strategy
- **Raw price sequences**: OHLCV + volume for representation learning
- **Engineered features**: Leverage existing 86-feature pipeline from Phase 5
- **Learning approach**: Let models learn which inputs to prioritize
- **Benefit**: Combines domain knowledge (features) with raw pattern detection

---

## Architectural Decisions

### Independence from Ensemble
**Decision**: Deep learning runs as completely separate strategy
**Rationale**:
- Different training dynamics (epochs, learning rates, GPU requirements)
- Different inference characteristics (batch processing, sequence dependencies)
- Allows independent optimization of both systems
- Enables A/B testing and performance comparison

**Implementation**:
```python
# Existing ensemble system (unchanged)
trading/ml/ensemble/regime_aware_ensemble.py

# New deep learning system (separate)
trading/ml/deep_learning/
  ├── lstm_model.py
  ├── transformer_model.py
  ├── deep_learning_strategy.py
  └── training.py
```

### Parallel Portfolio Management
**Decision**: Separate portfolios, no shared positions
**Rationale**:
- Clear attribution of performance to each strategy
- No interference between strategies
- Independent risk budgets
- Easier to compare and evaluate

**Risk Management**:
- Each strategy has its own max position size
- Each strategy has its own daily drawdown limits
- Each strategy has independent circuit breakers
- No shared risk budget (prevents one strategy from blocking the other)

### Hybrid Training Data
**Decision**: Feed both raw sequences AND engineered features
**Rationale**:
- Raw data: Allows deep learning to discover novel patterns
- Engineered features: Leverages domain expertise from Phase 5
- Models learn optimal weighting of both input types
- Best of both worlds: representation learning + domain knowledge

**Data Pipeline**:
```python
# Raw sequence data (e.g., 100 timesteps of OHLCV)
raw_input: shape (batch, 100, 5)  # 5 = O, H, L, C, V

# Engineered features (86 features from Phase 5)
feature_input: shape (batch, 86)

# Combined input strategy:
# - LSTM: Concatenate features to each timestep
# - Transformer: Separate embedding for sequences and features
```

---

## What's IN SCOPE

### Must Have (Phase 7 Deliverables)
1. ✅ **LSTM implementation**:
   - Multi-layer LSTM architecture
   - Sequence modeling for price prediction
   - Dropout for regularization
   - Training script with validation

2. ✅ **Transformer implementation**:
   - Multi-head self-attention
   - Positional encoding for time-series
   - Feed-forward layers
   - Training script with validation

3. ✅ **Hybrid data pipeline**:
   - Sequence windowing for raw OHLCV
   - Feature extraction from Phase 5 pipeline
   - Combined input preprocessing
   - Train/validation/test splits

4. ✅ **Independent trading strategy**:
   - Separate TradingManager for deep learning
   - Independent position management
   - Independent risk controls
   - Performance tracking and logging

5. ✅ **Model persistence**:
   - Save/load trained models
   - Checkpoint during training
   - Model versioning

6. ✅ **Basic evaluation**:
   - Accuracy, precision, recall
   - Sharpe ratio on validation set
   - Comparison vs random baseline

### Nice to Have (If Time Permits)
- Model interpretation (attention weights visualization)
- Early stopping based on validation loss
- Learning rate scheduling

---

## What's OUT OF SCOPE

### Explicitly NOT Included in Phase 7

1. ❌ **GPU optimization and deployment**:
   - Keep models CPU-compatible for MacBook
   - No CUDA optimizations
   - No distributed training
   - Defer GPU deployment to future phase

2. ❌ **Automated hyperparameter tuning**:
   - Use reasonable default hyperparameters
   - Manual tuning acceptable
   - No Optuna, Ray Tune, or AutoML
   - Defer hyperparameter optimization to Phase 8

3. ❌ **Advanced architectures**:
   - No GRU (Gated Recurrent Units)
   - No TCN (Temporal Convolutional Networks)
   - No hybrid CNN-LSTM architectures
   - Stick to vanilla LSTM and Transformer only

4. ❌ **Real-time inference optimization**:
   - Focus on accuracy, not speed
   - Accept slower inference times
   - No model quantization or pruning
   - Defer optimization to future phase

5. ❌ **Online learning / incremental training**:
   - Batch training only (offline)
   - No streaming updates
   - Retrain periodically as needed

6. ❌ **Multi-symbol / multi-asset training**:
   - Train separate models per symbol (e.g., BTC/USDT)
   - No transfer learning across assets
   - Defer to future phase

---

## Success Criteria

### Functional Requirements (All Must Pass)
1. ✅ **LSTM model trains successfully** on historical data with validation accuracy > 55%
2. ✅ **Transformer model trains successfully** on historical data with validation accuracy > 55%
3. ✅ **Hybrid data pipeline works** - combines raw sequences + 86 features correctly
4. ✅ **Independent strategy executes trades** without interfering with ensemble system
5. ✅ **Model persistence works** - save/load models without errors
6. ✅ **CPU-compatible** - runs on MacBook without GPU

### Performance Requirements (Target Metrics)
1. ✅ **Validation accuracy**: > 55% (better than random 50%)
2. ✅ **Training stability**: Loss decreases over epochs, no divergence
3. ✅ **Inference speed**: < 500ms per prediction on CPU (acceptable for live trading)
4. ✅ **Memory footprint**: < 2GB RAM during training (MacBook compatible)

### Code Quality Requirements
1. ✅ **Follows Phase 4 standards**: Structured logging, custom exceptions, timeouts
2. ✅ **Testable**: Unit tests for models, data pipeline, strategy
3. ✅ **Documented**: Clear docstrings, architecture decisions documented

---

## Integration Points

### With Existing System

**Phase 5 (Feature Engineering)**:
- Import feature extraction functions from `trading/features/`
- Use all 86 features as input to models
- Reuse regime detection for context (optional metadata, not training input)

**Phase 4 (Logging & Exceptions)**:
- Use `trading/logging_config.py` for structured logging
- Add `DeepLearningError` to `trading/exceptions.py`
- Apply timeout decorators to training/inference

**Phase 6 (Ensemble Framework)**:
- **NO direct dependency** - completely separate
- Can compare performance metrics post-hoc
- Potentially ensemble DL predictions with ensemble predictions in future phase (out of scope for Phase 7)

**Risk Management**:
- Reuse `trading/risk/` module for position sizing, drawdown checks
- Independent risk limits for DL strategy
- Separate circuit breakers

---

## Technical Constraints

### MacBook Compatibility (CRITICAL)
- **CPU-only training**: No GPU required
- **Memory limit**: < 2GB RAM during training
- **Storage**: Model checkpoints < 500MB per model
- **Python environment**: Use existing venv, add PyTorch/TensorFlow

### Framework Choice
**Decision needed in Phase 7**:
- **PyTorch** vs **TensorFlow/Keras**
- Recommendation: PyTorch (better for research, easier debugging, MacBook compatible)

### Training Data Requirements
- **Minimum data**: 10,000 candles (historical)
- **Sequence length**: 50-100 timesteps (research needed)
- **Train/val/test split**: 70/15/15 (time-series aware, no shuffle)

### Inference Requirements
- **Latency**: < 500ms per prediction (CPU)
- **Batch size**: 1 (real-time single prediction)
- **Fallback**: Return neutral signal if model fails

---

## Research Topics (To Be Documented in 07-RESEARCH.md)

### LSTM Architecture
- Sequence length for financial data (50, 100, 200 timesteps?)
- Number of LSTM layers (1, 2, 3?)
- Hidden dimensions (64, 128, 256?)
- Dropout rate (0.2, 0.3, 0.5?)
- Bidirectional vs unidirectional
- Stacked LSTM vs single layer with more units

### Transformer Architecture
- Positional encoding strategies for time-series
- Number of attention heads (4, 8, 16?)
- Feed-forward dimension (256, 512, 1024?)
- Number of encoder layers (2, 4, 6?)
- Attention mask for causal predictions
- Sequence length considerations (Transformers scale O(n²) with sequence length)

### Training Strategies
- Loss function (binary cross-entropy, focal loss, custom?)
- Optimizer (Adam, AdamW, SGD with momentum?)
- Learning rate (1e-3, 1e-4, 1e-5?)
- Batch size (32, 64, 128?)
- Number of epochs (50, 100, 200?)
- Early stopping criteria (patience, min delta)
- Label creation (5 candles lookahead like Phase 2, or different horizon?)

### Data Preprocessing
- Normalization strategy (min-max, z-score, robust scaler?)
- Sequence windowing (rolling window, fixed window?)
- Feature scaling (same scaler for raw + engineered, or separate?)
- Handling missing data (forward fill, interpolation, drop?)

### Evaluation Metrics
- Classification metrics (accuracy, precision, recall, F1)
- Financial metrics (Sharpe ratio, Sortino ratio, max drawdown)
- Backtest on validation set (realistic trading simulation)
- Confusion matrix analysis (which predictions are wrong?)

---

## Open Questions (To Be Resolved During Research)

1. **PyTorch vs TensorFlow**: Which framework is better for this use case?
2. **Sequence length**: How many timesteps should LSTM/Transformer see?
3. **Label horizon**: Predict 5 candles ahead (like Phase 2), or different timeframe?
4. **Input combination**: Concatenate features to sequence, or separate encoders?
5. **Model size**: Trade-off between capacity and overfitting on limited data?
6. **Regularization**: Dropout only, or also L2/weight decay?
7. **Attention mask**: Should Transformer use causal mask (no future data)?

---

## Dependencies

### New Requirements
- `torch>=2.0.0` (PyTorch - already in requirements.txt)
- OR `tensorflow>=2.14.0` (TensorFlow - if chosen instead)
- `tensorboard>=2.14.0` (training visualization, optional)

### Existing Dependencies (Reuse)
- `pandas`, `numpy` (data processing)
- `scikit-learn` (metrics, preprocessing)
- `lightgbm`, `xgboost` (for comparison only)
- Phase 5 features (all 86 engineered features)

---

## Files to Create (Preliminary)

```
trading/ml/deep_learning/
├── __init__.py
├── lstm_model.py              # LSTM architecture
├── transformer_model.py       # Transformer architecture
├── data_pipeline.py           # Hybrid data preprocessing
├── training.py                # Training scripts for both models
├── deep_learning_strategy.py  # Independent trading strategy
└── persistence.py             # Model save/load

trading/cli_deep_learning.py   # CLI for DL strategy (optional)

.planning/phases/07-deep-learning-models/
├── 07-CONTEXT.md              # This file
├── 07-RESEARCH.md             # Ecosystem research (next)
└── 07-01-PLAN.md              # Execution plan (after research)
```

---

## Next Steps

1. **Research Phase** (`/gsd:research-phase 7`):
   - PyTorch LSTM API and best practices
   - PyTorch Transformer implementation patterns
   - Financial time-series deep learning papers
   - Sequence length recommendations
   - Training stability techniques
   - Create 07-RESEARCH.md

2. **Planning Phase** (`/gsd:plan-phase 7`):
   - Break into executable tasks
   - Define verification steps
   - Estimate complexity
   - Create 07-01-PLAN.md

3. **Execution Phase** (`/gsd:execute-plan`):
   - Implement LSTM model
   - Implement Transformer model
   - Create hybrid data pipeline
   - Build independent trading strategy
   - Train and evaluate models

---

**Vision gathered:** 2025-12-27
**Status:** Ready for research phase
