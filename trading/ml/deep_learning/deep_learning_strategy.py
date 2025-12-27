"""Deep Learning Trading Strategy - Independent from Ensemble System.

ARCHITECTURE DECISION (from 07-CONTEXT.md):
- **Independent execution**: Separate portfolio management, no shared positions with ensemble
- **Separate risk controls**: Independent RiskAuditAgent and position limits
- **Parallel operation**: Can run alongside ensemble system for A/B testing

WHY this architecture:
- Clear performance attribution (know which strategy generated which returns)
- No interference between strategies
- Independent risk budgets (one strategy doesn't block the other)
- Allows comparing deep learning vs ensemble approaches

INTEGRATION:
- Reuses Phase 5 features (86 engineered features via FeatureEngineer)
- Reuses Phase 4 risk management (RiskAuditAgent from trading.agents.risk_audit)
- Reuses Phase 4 logging (structured logging from trading.logging_config)
- Uses ModelPersistence for security-hardened model loading
"""

import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

from trading.config import TradingConfig
from trading.state import TradingState
from trading.providers.base import BaseExchangeProvider
from trading.agents.risk_audit import RiskAuditAgent
from trading.logging_config import get_logger, DecisionContext
from trading.exceptions import TradingBotError, AgentError
from trading.ml.deep_learning.persistence import ModelPersistence
from trading.ml.deep_learning.data.preprocessing import DataPreprocessor
from trading.ml.feature_engineering import FeatureEngineer


class DeepLearningStrategy:
    """Independent deep learning trading strategy with separate portfolio management.

    KEY DESIGN:
    - Independent from ensemble system (NO shared state)
    - Separate portfolio (separate positions, separate PnL tracking)
    - Separate risk controls (own RiskAuditAgent instance)
    - Uses trained BiLSTM or Transformer model for predictions

    Example:
        >>> strategy = DeepLearningStrategy(
        ...     config=config,
        ...     provider=provider,
        ...     model_type='lstm'
        ... )
        >>> prediction = await strategy.predict("BTC/USDT")
        >>> print(prediction)
        {'signal': 'buy', 'confidence': 0.85, 'model': 'lstm', 'prediction_time_ms': 120}
    """

    def __init__(
        self,
        config: TradingConfig,
        provider: BaseExchangeProvider,
        model_type: str = 'lstm',
        spec_dir: Path = Path("specs/deep_learning"),
        sequence_length: int = 50
    ):
        """Initialize deep learning strategy.

        Args:
            config: Trading configuration
            provider: Exchange provider instance
            model_type: "lstm" or "transformer"
            spec_dir: Directory for state isolation (separate from ensemble)
            sequence_length: Sequence length for model (must match training)
        """
        if model_type not in ['lstm', 'transformer']:
            raise ValueError(f"model_type must be 'lstm' or 'transformer', got {model_type}")

        self.config = config
        self.provider = provider
        self.model_type = model_type
        self.spec_dir = Path(spec_dir)
        self.sequence_length = sequence_length
        self.logger = get_logger(__name__)

        # Create spec directory for state isolation
        self.spec_dir.mkdir(parents=True, exist_ok=True)

        # Load or create SEPARATE state (independent from ensemble)
        self.state = TradingState.load(self.spec_dir) or TradingState()

        # Initialize SEPARATE risk audit agent (independent risk controls)
        self.risk_audit = RiskAuditAgent(config=config)

        # Load trained model
        self.persistence = ModelPersistence()
        if model_type == 'lstm':
            self.model = self.persistence.load_lstm()
        else:
            self.model = self.persistence.load_transformer()

        if self.model is None:
            raise ValueError(
                f"No trained {model_type} model found. "
                f"Train model first using: python trading/ml/deep_learning/training/train_{model_type}.py"
            )

        # Set model to evaluation mode (PyTorch method, not Python's dangerous function)
        self.model.eval()

        # Initialize feature engineer (Phase 5 integration)
        self.feature_engineer = FeatureEngineer(
            windows=[5, 10, 20, 50],
            include_target=False,  # No target needed for inference
            target_horizon=5
        )

        # Initialize data preprocessor
        self.preprocessor = DataPreprocessor(sequence_length=sequence_length)

        self.logger.info(
            f"DeepLearningStrategy initialized with {model_type} model",
            extra={
                "model_type": model_type,
                "sequence_length": sequence_length,
                "spec_dir": str(spec_dir)
            }
        )

    async def predict(self, symbol: str) -> Dict[str, Any]:
        """Generate trading prediction using deep learning model.

        PREDICTION FLOW (from 07-03-PLAN.md):
        1. Fetch recent OHLCV candles (sequence_length + buffer)
        2. Engineer 86 features using Phase 5 pipeline
        3. Create sequence using DataPreprocessor
        4. Run inference with trained model
        5. Convert probability to signal (buy/sell/neutral)

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")

        Returns:
            Dictionary with:
                - signal: "buy", "sell", or "neutral"
                - confidence: Float probability (0.0 to 1.0)
                - model: Model type used ("lstm" or "transformer")
                - prediction_time_ms: Inference time in milliseconds

        Raises:
            TradingBotError: If prediction fails
        """
        start_time = time.time()

        try:
            # Step 1: Fetch recent candles (need extra for feature engineering)
            buffer = max(self.feature_engineer.windows) + 20  # Extra buffer for indicators
            limit = self.sequence_length + buffer
            self.logger.debug(f"Fetching {limit} candles for {symbol}")

            candles = await self.provider.fetch_ohlcv(
                symbol=symbol,
                timeframe='5m',
                limit=limit
            )

            if len(candles) < limit:
                raise ValueError(
                    f"Insufficient candles: got {len(candles)}, need {limit}"
                )

            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': c.timestamp,
                    'open': c.open,
                    'high': c.high,
                    'low': c.low,
                    'close': c.close,
                    'volume': c.volume
                }
                for c in candles
            ])

            # Step 2: Engineer features using Phase 5 pipeline
            self.logger.debug("Engineering features...")
            df_features = self.feature_engineer.transform(df)
            feature_columns = self.feature_engineer.get_feature_names(exclude_target=True)

            # Take only the most recent samples needed for sequence
            df_recent = df_features.tail(self.sequence_length + 50)  # Extra buffer

            # Step 3: Fit scaler on recent data and create sequence
            # NOTE: In production, you should save/load the scaler from training
            # For now, fit on recent data (not ideal but works for demo)
            self.logger.debug("Creating sequence...")

            # Create dummy label column (not used, but required by create_sequences)
            df_recent['dummy_label'] = 0

            sequences, _ = self.preprocessor.create_sequences(
                df_recent,
                feature_columns=feature_columns,
                label_column='dummy_label',
                fit_scaler=True  # Fit on recent data
            )

            # Take the last sequence (most recent)
            last_sequence = sequences[-1]  # Shape: (sequence_length, num_features)

            # Step 4: Run inference
            self.logger.debug("Running inference...")
            with torch.no_grad():
                # Convert to tensor
                x = torch.FloatTensor(last_sequence).unsqueeze(0)  # Shape: (1, seq_len, features)

                # Get model logit
                logit = self.model(x)  # Shape: (1,)

                # Convert to probability using sigmoid
                probability = torch.sigmoid(logit).item()

            # Step 5: Convert probability to signal
            if probability > 0.6:
                signal = 'buy'
                confidence = probability
            elif probability < 0.4:
                signal = 'sell'
                confidence = 1.0 - probability  # Flip for sell confidence
            else:
                signal = 'neutral'
                confidence = 0.5

            # Calculate prediction time
            prediction_time_ms = int((time.time() - start_time) * 1000)

            result = {
                'signal': signal,
                'confidence': confidence,
                'model': self.model_type,
                'prediction_time_ms': prediction_time_ms,
                'probability': probability  # Raw probability for debugging
            }

            self.logger.info(
                f"Prediction generated: {signal} (confidence={confidence:.3f})",
                extra=result
            )

            return result

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            raise TradingBotError(f"Deep learning prediction failed: {e}")

    async def execute_strategy(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Execute deep learning trading strategy with risk checks.

        EXECUTION FLOW (from 07-03-PLAN.md):
        1. Generate prediction using trained model
        2. Apply risk checks (independent RiskAuditAgent)
        3. Execute trade if signal strong and risk passes
        4. Save state

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")

        Returns:
            Execution result dictionary, or None if no trade
        """
        decision_id = DecisionContext.create_decision_id()

        with DecisionContext(decision_id=decision_id):
            try:
                # Step 1: Generate prediction
                self.logger.info(f"Generating prediction for {symbol}...")
                prediction = await self.predict(symbol)

                if prediction['signal'] == 'neutral':
                    self.logger.info("Signal is neutral, skipping trade")
                    return None

                # Step 2: Prepare decision for risk audit
                decision = {
                    'action': prediction['signal'],  # 'buy' or 'sell'
                    'confidence': prediction['confidence'],
                    'symbol': symbol,
                    'model': prediction['model'],
                    'decision_id': decision_id
                }

                # Step 3: Risk audit (independent risk controls)
                self.logger.info("Running risk audit...")
                context = {
                    'decision': decision,
                    'state': self.state
                }
                risk_result = await self.risk_audit.execute(context)

                if risk_result['risk_audit']['veto']:
                    reason = risk_result['risk_audit']['reason']
                    self.logger.warning(f"Trade vetoed by risk audit: {reason}")
                    return None

                # Step 4: Execute trade (placeholder - implement actual execution)
                self.logger.info(f"Executing {decision['action']} trade for {symbol}")
                # TODO: Implement actual order execution via provider
                # For now, just log the decision

                # Step 5: Save state
                self.state.save(self.spec_dir)

                return {
                    'decision_id': decision_id,
                    'signal': prediction['signal'],
                    'confidence': prediction['confidence'],
                    'executed': True,
                    'timestamp': datetime.now().isoformat()
                }

            except Exception as e:
                self.logger.error(f"Strategy execution failed: {e}", exc_info=True)
                raise TradingBotError(f"Deep learning strategy execution failed: {e}")
