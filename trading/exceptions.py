"""
Custom exception hierarchy for LLM-TradeBot.

All trading bot exceptions derive from TradingBotError for easy catching.
Organized by domain: Configuration, API, Risk, Agent, State.
"""


class TradingBotError(Exception):
    """Base exception for all trading bot errors."""
    pass


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(TradingBotError):
    """Configuration-related errors (env vars, settings)."""
    pass


class MissingCredentialError(ConfigurationError):
    """Required API credential not found in environment."""
    pass


class InvalidConfigValueError(ConfigurationError):
    """Configuration value invalid or out of range."""
    pass


# ============================================================================
# API Errors (Exchange, Sentiment, External Services)
# ============================================================================

class APIError(TradingBotError):
    """External API communication errors."""
    pass


class ExchangeConnectionError(APIError):
    """Cannot connect to exchange API."""
    pass


class RateLimitExceededError(APIError):
    """Exchange rate limit exceeded."""
    pass


class OrderRejectedError(APIError):
    """Exchange rejected order (insufficient balance, invalid params, etc.)."""
    pass


class InvalidSymbolError(APIError):
    """Trading symbol not supported by exchange."""
    pass


# ============================================================================
# Risk Management Errors (Safety Violations)
# ============================================================================

class RiskViolationError(TradingBotError):
    """Risk management constraint violations."""
    pass


class InsufficientBalanceError(RiskViolationError):
    """Insufficient balance to execute trade."""
    pass


class PositionLimitExceededError(RiskViolationError):
    """Position would exceed max position size."""
    pass


class DailyDrawdownExceededError(RiskViolationError):
    """Daily drawdown limit exceeded."""
    pass


class CircuitBreakerTrippedError(RiskViolationError):
    """Circuit breaker activated due to excessive losses."""
    pass


class ConfidenceThresholdError(RiskViolationError):
    """Decision confidence below minimum threshold."""
    pass


# ============================================================================
# Agent Errors (Analysis, Prediction, Decision)
# ============================================================================

class AgentError(TradingBotError):
    """Agent execution errors."""
    pass


class InvalidIndicatorDataError(AgentError):
    """Indicator data is invalid, incomplete, or missing."""
    pass


class ModelPredictionError(AgentError):
    """ML model prediction failed."""
    pass


class AgentTimeoutError(AgentError):
    """Agent execution exceeded timeout."""
    pass


class InsufficientMarketDataError(AgentError):
    """Not enough market data for analysis (need 100+ candles)."""
    pass


# ============================================================================
# ML Model Errors
# ============================================================================

class ModelError(TradingBotError):
    """Machine learning model errors."""

    def __init__(self, message: str, model_name: str = None, context: dict = None):
        """
        Initialize model error with context.

        Args:
            message: Error message
            model_name: Name of the model that failed
            context: Additional context dictionary
        """
        self.model_name = model_name
        self.context = context or {}

        full_message = message
        if model_name:
            full_message = f"[{model_name}] {message}"

        super().__init__(full_message)


class ModelNotFittedError(ModelError):
    """Model must be fitted before making predictions."""
    pass


class FeatureMismatchError(ModelError):
    """Feature count or order mismatch between training and prediction."""
    pass


class ModelLoadError(ModelError):
    """Failed to load model from disk."""
    pass


class ModelSaveError(ModelError):
    """Failed to save model to disk."""
    pass


# ============================================================================
# State Persistence Errors
# ============================================================================

class StateError(TradingBotError):
    """State persistence errors."""
    pass


class StateCorruptedError(StateError):
    """State file corrupted or invalid JSON."""
    pass


class StateSaveFailedError(StateError):
    """Failed to save state to disk."""
    pass


class StateLoadFailedError(StateError):
    """Failed to load state from disk."""
    pass
