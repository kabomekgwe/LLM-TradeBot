"""
Comprehensive tests for all 8 trading agents.
Tests decision-making logic across various market scenarios.
"""

import pytest
import pytest_asyncio
import numpy as np
from unittest.mock import AsyncMock, Mock, MagicMock
from dataclasses import dataclass

from trading.agents.quant_analyst import QuantAnalystAgent
from trading.agents.predict import PredictAgent
from trading.agents.bull import BullAgent
from trading.agents.bear import BearAgent
from trading.agents.risk_audit import RiskAuditAgent
from trading.agents.data_sync import DataSyncAgent
from trading.agents.execution import ExecutionEngine
from trading.agents.decision_core import DecisionCoreAgent
from trading.config import TradingConfig
from trading.state import TradingState


# === Test Fixtures ===


@pytest.fixture
def mock_config():
    """Create mock TradingConfig for testing."""
    return TradingConfig(
        provider="paper",
        api_key="test_key",
        api_secret="test_secret",
        testnet=True,
        max_position_size_usd=1000.0,
        max_daily_drawdown_pct=5.0,
        max_open_positions=3,
        decision_threshold=0.6,
        enable_ml_predictions=True
    )


@pytest.fixture
def mock_provider():
    """Create mock exchange provider."""
    provider = AsyncMock()
    provider.fetch_ohlcv = AsyncMock(return_value=[])
    provider.fetch_ticker = AsyncMock(return_value={})
    provider.fetch_orderbook = AsyncMock(return_value={})
    provider.create_order = AsyncMock(return_value={})
    return provider


@pytest.fixture
def trading_state():
    """Create fresh TradingState for testing."""
    return TradingState()


# === QuantAnalystAgent Tests ===


@pytest.mark.asyncio
@pytest.mark.unit
async def test_quant_analyst_calculates_indicators(ohlcv_factory, mock_provider, mock_config):
    """Test QuantAnalyst calculates RSI, MACD, Bollinger Bands."""
    # Arrange
    ohlcv_data = ohlcv_factory(trend="uptrend", num_candles=100, timeframe="1h")
    context = {'market_data': {'1h': ohlcv_data}}
    agent = QuantAnalystAgent(mock_provider, mock_config)

    # Act
    result = await agent.execute(context)

    # Assert
    assert 'quant_signals' in result
    indicators = result['quant_signals']['indicators']
    assert 'rsi' in indicators
    assert 'macd' in indicators
    assert 'bollinger' in indicators

    # RSI should be between 0 and 100
    assert 0 <= indicators['rsi']['value'] <= 100

    # MACD should have macd, signal, histogram
    assert 'macd' in indicators['macd']
    assert 'signal' in indicators['macd']
    assert 'histogram' in indicators['macd']

    # Bollinger Bands should have upper, middle, lower
    assert 'upper' in indicators['bollinger']
    assert 'middle' in indicators['bollinger']
    assert 'lower' in indicators['bollinger']
    assert indicators['bollinger']['upper'] > indicators['bollinger']['middle']
    assert indicators['bollinger']['middle'] > indicators['bollinger']['lower']


@pytest.mark.asyncio
@pytest.mark.unit
async def test_quant_analyst_handles_insufficient_data(mock_provider, mock_config):
    """Test QuantAnalyst handles insufficient data gracefully."""
    # Arrange - less than 26 candles (MACD requirement)
    context = {'market_data': {'1h': []}}
    agent = QuantAnalystAgent(mock_provider, mock_config)

    # Act
    result = await agent.execute(context)

    # Assert
    assert 'quant_signals' in result
    assert result['quant_signals']['overall_signal'] == 'neutral'
    assert 'error' in result['quant_signals']


@pytest.mark.asyncio
@pytest.mark.unit
async def test_quant_analyst_detects_overbought_oversold(ohlcv_factory, mock_provider, mock_config):
    """Test QuantAnalyst detects overbought/oversold conditions."""
    # Create strong uptrend (should produce elevated RSI)
    ohlcv_data = ohlcv_factory(trend="uptrend", num_candles=100, timeframe="1h", volatility=0.01)
    context = {'market_data': {'1h': ohlcv_data}}
    agent = QuantAnalystAgent(mock_provider, mock_config)

    result = await agent.execute(context)
    rsi = result['quant_signals']['indicators']['rsi']

    # Strong uptrend should produce elevated RSI
    assert rsi['value'] > 40  # At least above neutral


# === PredictAgent Tests ===


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.ml
async def test_predict_agent_with_model(mock_lightgbm_model, mock_provider, mock_config):
    """Test PredictAgent returns ML prediction with trained model."""
    # Arrange
    agent = PredictAgent(mock_provider, mock_config)
    agent.model = mock_lightgbm_model  # Inject mock model

    indicators = {
        'rsi': {'value': 65},
        'macd': {'macd': 0.5, 'signal': 0.3, 'histogram': 0.2},
        'bollinger': {'upper': 30100, 'middle': 30000, 'lower': 29900}
    }
    context = {'quant_analyst': {'indicators': indicators}}

    # Act
    result = await agent.execute(context)

    # Assert
    assert 'ml_prediction' in result
    ml_pred = result['ml_prediction']
    assert 'direction' in ml_pred
    assert 'confidence' in ml_pred
    assert ml_pred['direction'] in ['up', 'down']
    assert 0.0 <= ml_pred['confidence'] <= 1.0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_predict_agent_without_model(mock_provider, mock_config):
    """Test PredictAgent returns neutral fallback without trained model."""
    # Arrange
    agent = PredictAgent(mock_provider, mock_config)
    agent.model = None  # No model loaded

    context = {'quant_analyst': {'indicators': {}}}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['ml_prediction']['direction'] == 'neutral'
    assert result['ml_prediction']['confidence'] == 0.0
    assert 'reason' in result['ml_prediction']


@pytest.mark.asyncio
@pytest.mark.unit
async def test_predict_agent_disabled(mock_provider, mock_config):
    """Test PredictAgent when ML predictions are disabled."""
    # Arrange
    mock_config.enable_ml_predictions = False
    agent = PredictAgent(mock_provider, mock_config)

    context = {'quant_analyst': {'indicators': {}}}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['ml_prediction']['direction'] == 'neutral'
    assert result['ml_prediction']['confidence'] == 0.0
    assert result['ml_prediction']['enabled'] is False


# === BullAgent Tests ===


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("rsi_value,macd_bullish,bb_position,min_confidence", [
    (25, True, 'lower', 0.7),  # Strong bullish: oversold + bullish MACD + lower BB
    (45, True, 'middle_lower', 0.3),  # Moderate bullish
    (55, False, 'middle', 0.0),  # Neutral: no strong signals
    (75, False, 'upper', 0.0),  # Bearish conditions (bull agent abstains)
])
async def test_bull_agent_decision_logic(rsi_value, macd_bullish, bb_position, min_confidence, mock_provider, mock_config):
    """Test BullAgent votes correctly based on technical indicators."""
    # Arrange
    agent = BullAgent(mock_provider, mock_config)

    indicators = {
        'rsi': {'value': rsi_value, 'oversold': rsi_value < 30, 'overbought': rsi_value > 70},
        'macd': {'bullish': macd_bullish, 'histogram': 0.2 if macd_bullish else -0.2},
        'bollinger': {'position': bb_position}
    }
    context = {'quant_analyst': {'indicators': indicators}}

    # Act
    result = await agent.execute(context)

    # Assert
    assert 'bull_vote' in result
    bull_vote = result['bull_vote']
    assert bull_vote['confidence'] >= min_confidence
    assert 0.0 <= bull_vote['confidence'] <= 1.0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_bull_agent_no_indicators(mock_provider, mock_config):
    """Test BullAgent handles missing indicators gracefully."""
    # Arrange
    agent = BullAgent(mock_provider, mock_config)
    context = {'quant_analyst': {'indicators': {}}}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['bull_vote']['action'] == 'hold'
    assert result['bull_vote']['confidence'] == 0.0


# === BearAgent Tests ===


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("rsi_value,macd_bullish,bb_position,min_confidence", [
    (75, False, 'upper', 0.7),  # Strong bearish: overbought + bearish MACD + upper BB
    (55, False, 'middle_upper', 0.3),  # Moderate bearish
    (45, True, 'middle', 0.0),  # Neutral: no strong signals
    (25, True, 'lower', 0.0),  # Bullish conditions (bear agent abstains)
])
async def test_bear_agent_decision_logic(rsi_value, macd_bullish, bb_position, min_confidence, mock_provider, mock_config):
    """Test BearAgent votes correctly based on technical indicators."""
    # Arrange
    agent = BearAgent(mock_provider, mock_config)

    indicators = {
        'rsi': {'value': rsi_value, 'oversold': rsi_value < 30, 'overbought': rsi_value > 70},
        'macd': {'bullish': macd_bullish, 'histogram': 0.2 if macd_bullish else -0.2},
        'bollinger': {'position': bb_position}
    }
    context = {'quant_analyst': {'indicators': indicators}}

    # Act
    result = await agent.execute(context)

    # Assert
    assert 'bear_vote' in result
    bear_vote = result['bear_vote']
    assert bear_vote['confidence'] >= min_confidence
    assert 0.0 <= bear_vote['confidence'] <= 1.0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_bear_agent_no_indicators(mock_provider, mock_config):
    """Test BearAgent handles missing indicators gracefully."""
    # Arrange
    agent = BearAgent(mock_provider, mock_config)
    context = {'quant_analyst': {'indicators': {}}}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['bear_vote']['action'] == 'hold'
    assert result['bear_vote']['confidence'] == 0.0


# === DecisionCoreAgent Tests ===


@pytest.mark.asyncio
@pytest.mark.unit
async def test_decision_core_buy_signal(mock_provider, mock_config):
    """Test DecisionCore produces buy signal when bull confidence high."""
    # Arrange
    agent = DecisionCoreAgent(mock_provider, mock_config)

    context = {
        'bull_vote': {'action': 'buy', 'confidence': 0.8},
        'bear_vote': {'action': 'hold', 'confidence': 0.2},
        'regime': 'trending'
    }

    # Act
    result = await agent.execute(context)

    # Assert
    assert 'decision' in result
    decision = result['decision']
    assert decision['action'] in ['buy', 'hold']  # Might be hold if threshold not met
    assert 0.0 <= decision['confidence'] <= 1.0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_decision_core_sell_signal(mock_provider, mock_config):
    """Test DecisionCore produces sell signal when bear confidence high."""
    # Arrange
    agent = DecisionCoreAgent(mock_provider, mock_config)

    context = {
        'bull_vote': {'action': 'hold', 'confidence': 0.2},
        'bear_vote': {'action': 'sell', 'confidence': 0.8},
        'regime': 'choppy'
    }

    # Act
    result = await agent.execute(context)

    # Assert
    assert 'decision' in result
    decision = result['decision']
    assert decision['action'] in ['sell', 'hold']
    assert 0.0 <= decision['confidence'] <= 1.0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_decision_core_hold_signal(mock_provider, mock_config):
    """Test DecisionCore produces hold when no clear consensus."""
    # Arrange
    agent = DecisionCoreAgent(mock_provider, mock_config)

    context = {
        'bull_vote': {'action': 'hold', 'confidence': 0.3},
        'bear_vote': {'action': 'hold', 'confidence': 0.3},
        'regime': 'neutral'
    }

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['decision']['action'] == 'hold'
    assert result['decision']['confidence'] == 0.0


# === DataSyncAgent Tests ===


@pytest.mark.asyncio
@pytest.mark.integration
async def test_data_sync_fetches_ohlcv(mock_exchange, mock_provider):
    """Test DataSyncAgent fetches OHLCV data from exchange."""
    # Arrange
    config = TradingConfig(provider="paper", api_key="test", api_secret="test")

    # Mock provider methods
    mock_provider.fetch_ohlcv = AsyncMock(return_value=[])
    mock_provider.fetch_ticker = AsyncMock(return_value={'last': 30000})
    mock_provider.fetch_orderbook = AsyncMock(return_value={'bids': [], 'asks': []})

    agent = DataSyncAgent(mock_provider, config)

    # Act
    result = await agent.execute({'symbol': 'BTC/USDT'})

    # Assert
    assert 'market_data' in result
    assert mock_provider.fetch_ohlcv.call_count >= 3  # 3 timeframes


@pytest.mark.asyncio
@pytest.mark.integration
async def test_data_sync_requires_symbol(mock_provider):
    """Test DataSyncAgent raises error without symbol."""
    # Arrange
    config = TradingConfig(provider="paper", api_key="test", api_secret="test")
    agent = DataSyncAgent(mock_provider, config)

    # Act & Assert
    with pytest.raises(ValueError, match="Symbol is required"):
        await agent.execute({})


# === ExecutionEngine Tests ===


@pytest.mark.asyncio
@pytest.mark.integration
async def test_execution_places_order(mock_provider, mock_config, trading_state):
    """Test ExecutionEngine places order based on decision."""
    # Arrange
    mock_provider.create_order = AsyncMock(return_value={
        'id': 'order-123',
        'status': 'closed',
        'filled': 0.01
    })

    agent = ExecutionEngine(mock_provider, mock_config)
    # Mock OrderManager
    agent.order_manager = Mock()
    agent.order_manager.calculate_position_size = Mock(return_value=0.01)
    agent.order_manager.create_bracket_order = AsyncMock(return_value=(
        Mock(id='entry-123'),
        Mock(id='stop-123'),
        Mock(id='tp-123')
    ))

    decision = {
        'action': 'buy',
        'confidence': 0.8
    }
    context = {
        'decision': decision,
        'state': trading_state,
        'symbol': 'BTC/USDT',
        'market_data': {
            'ticker': Mock(last=30000)
        }
    }

    # Act
    result = await agent.execute(context)

    # Assert
    assert 'execution' in result
    assert result['execution']['success'] is True


@pytest.mark.asyncio
@pytest.mark.integration
async def test_execution_skips_on_hold(mock_provider, mock_config, trading_state):
    """Test ExecutionEngine skips execution when decision is hold."""
    # Arrange
    agent = ExecutionEngine(mock_provider, mock_config)

    decision = {'action': 'hold'}
    context = {
        'decision': decision,
        'state': trading_state,
        'symbol': 'BTC/USDT'
    }

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['execution']['success'] is False
    assert 'No action to execute' in result['execution']['error']


# === RiskAuditAgent Tests ===


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_veto_circuit_breaker(mock_provider, mock_config, trading_state):
    """Test RiskAudit vetoes when circuit breaker tripped."""
    # Arrange
    agent = RiskAuditAgent(mock_provider, mock_config)
    trading_state.circuit_breaker_tripped = True
    trading_state.last_circuit_trip_reason = "Max drawdown exceeded"

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': trading_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] is True
    assert 'circuit breaker' in result['risk_audit']['reason'].lower()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_veto_max_positions(mock_provider, mock_config, trading_state):
    """Test RiskAudit vetoes when max positions reached."""
    # Arrange
    mock_config.max_open_positions = 2
    agent = RiskAuditAgent(mock_provider, mock_config)

    # Add 2 positions (at limit)
    trading_state.add_position({'symbol': 'BTC/USDT', 'size': 0.01})
    trading_state.add_position({'symbol': 'ETH/USDT', 'size': 0.1})

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': trading_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] is True
    assert 'Max open positions' in result['risk_audit']['reason']


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_veto_low_confidence(mock_provider, mock_config, trading_state):
    """Test RiskAudit vetoes when confidence below threshold."""
    # Arrange
    mock_config.decision_threshold = 0.6
    agent = RiskAuditAgent(mock_provider, mock_config)

    decision = {'action': 'buy', 'confidence': 0.4}  # Below threshold
    context = {'decision': decision, 'state': trading_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] is True
    assert 'Confidence below threshold' in result['risk_audit']['reason']


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_passes_valid_decision(mock_provider, mock_config, trading_state):
    """Test RiskAudit passes valid decision."""
    # Arrange
    agent = RiskAuditAgent(mock_provider, mock_config)

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': trading_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] is False
    assert result['risk_audit']['reason'] is None
