# Phase 6: Ensemble Model Framework - Context

**Gathered:** 2025-12-27
**Status:** Ready for research

<vision>
## How This Should Work

The ensemble model framework should support multiple combination strategies (voting, stacking, dynamic selection) and intelligently choose which strategy to use based on current market conditions. The system leverages the volatility regime detector from Phase 5 (HMM) to automatically determine market state, then selects the ensemble strategy that historically performs best in that regime.

When making trading decisions, the ensemble combines predictions from three tree-based models: LightGBM (already integrated), XGBoost (new), and Random Forest (new). Each model trains on the same 86 features from Phase 5 but brings different strengths:
- LightGBM: Fast gradient boosting, handles missing data well
- XGBoost: Robust regularization, excellent accuracy
- Random Forest: Decorrelated trees, good for non-linear patterns

The ensemble adapts to market conditions automatically:
- **Low volatility regime** → Use stacking (meta-model learns optimal combination)
- **High volatility regime** → Use voting (majority rule, more robust to noise)
- **Transitional regime** → Use dynamic selection (pick best recent performer)

This creates a trading system that doesn't rely on a single model's judgment but leverages the wisdom of multiple complementary approaches.

</vision>

<essential>
## What Must Be Nailed

All three pillars are equally critical for Phase 6 success:

- **Ensemble predictions are more accurate than single models** - Must see measurable improvement in prediction accuracy/Sharpe ratio vs using just LightGBM alone. The ensemble should consistently outperform any individual model.

- **Clean abstraction for adding/removing models** - Easy to plug in new models (e.g., add CatBoost later) or remove underperforming ones without rewriting code. Extensibility and maintainability matter as much as performance.

- **Regime-aware strategy switching works correctly** - The system correctly identifies market regimes (using Phase 5's HMM detector) and switches ensemble strategies appropriately. No bugs in the regime detection → strategy selection pipeline.

If any of these three fails, Phase 6 fails. Better accuracy, clean architecture, and correct regime switching are all non-negotiable.

</essential>

<boundaries>
## What's Out of Scope

- **Deep learning models (LSTM, Transformers)** - Phase 6 focuses exclusively on tree-based models (LightGBM, XGBoost, Random Forest). Neural networks are Phase 7 as the roadmap specifies.

- **Real-time model retraining** - Train models once with historical data. Automatic retraining on new data (online learning, incremental updates) is explicitly deferred. Models are static once trained.

</boundaries>

<specifics>
## Specific Ideas

**Comprehensive observability** - Full transparency into the ensemble's decision-making process:

1. **Show individual model predictions + ensemble result**
   - When the ensemble makes a decision, log/display what each model predicted individually
   - Example: "LightGBM: 0.65 buy, XGBoost: 0.55 buy, Random Forest: 0.45 neutral → Ensemble: 0.58 buy"
   - Transparency into the voting/combination process

2. **Track which ensemble strategy is active**
   - Clear visibility into which strategy is being used (voting/stacking/dynamic) and WHY
   - Example: "Using voting strategy - high volatility regime detected (regime_prob_1 = 0.82)"
   - Makes the system explainable and debuggable

3. **Feature importance across all models**
   - Aggregate feature importance from all three models to understand which of the 86 features are driving predictions across the ensemble
   - Helps identify if certain features work better with specific models
   - Informs future feature selection and engineering work

**Integration with existing infrastructure:**
- Leverage the structured JSON logging from Phase 4 for ensemble decision tracking
- Use the DecisionContext correlation IDs to trace ensemble decisions through the full trading pipeline
- Integrate with Phase 5's volatility regime detector (regime.py) for strategy switching
- Maintain the fail-fast error handling patterns established in v1.0

</specifics>

<notes>
## Additional Context

**Design philosophy:**
- Support multiple ensemble strategies but don't over-engineer - start with voting and stacking, add dynamic selection if time permits
- Regime-aware strategy switching is a key differentiator - most ensemble systems use fixed strategies
- Clean architecture is as important as performance - this framework should be extensible for Phase 7 (deep learning) and beyond

**Model diversity:**
- LightGBM, XGBoost, Random Forest chosen for complementary strengths
- Tree-based models work well with the tabular features from Phase 5
- Ensemble should benefit from diversity (each model makes different errors)

**Performance expectations:**
- Target: 10-20% improvement in prediction accuracy vs single LightGBM
- Target: Higher Sharpe ratio in backtests due to more robust predictions
- Regime-aware switching should reduce losses during volatile market transitions

**Hyperparameter tuning:**
- NOT marked as out-of-scope, so some level of hyperparameter optimization is acceptable
- But don't go overboard with AutoML - reasonable defaults are fine
- Focus on getting the ensemble architecture right, not squeezing every 0.1% of accuracy

</notes>

---

*Phase: 06-ensemble-model-framework*
*Context gathered: 2025-12-27*
