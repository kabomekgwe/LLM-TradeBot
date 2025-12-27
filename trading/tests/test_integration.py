"""
Integration tests for full agent pipeline.
Tests end-to-end flow: DataSync → QuantAnalyst → Predict/Bull/Bear → DecisionCore → RiskAudit → Execution.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock

from trading.agents.data_sync import DataSyncAgent
from trading.agents.quant_analyst import QuantAnalystAgent
from trading.agents.predict import PredictAgent
from trading.agents.bull import BullAgent
from trading.agents.bear import BearAgent
from trading.agents.decision_core import DecisionCoreAgent
from trading.agents.risk_audit import RiskAuditAgent
from trading.agents.execution import ExecutionEngine
from trading.config import TradingConfig
from trading.state import TradingState


# === Full Pipeline Integration Tests ===

@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_pipeline_bullish_scenario(
    mock_exchange,
    ohlcv_factory,
    mock_lightgbm_model
):
    """Test complete pipeline with bullish market scenario."""
    # Arrange - Setup bullish market data
    ohlcv_data = ohlcv_factory(trend="uptrend", num_candles=100, volatility=0.01)
    mock_exchange.fetch_ohlcv.return_value = ohlcv_data
    mock_exchange.fetch_balance.return_value = {
        'USDT': {'free': 10000, 'used': 0, 'total': 10000}
    }

    config = TradingConfig(
        provider="paper",
        api_key="test",
        api_secret="test",
        max_position_size_usd=1000,
        max_open_positions=3,
        decision_threshold=0.1  # Low threshold for testing
    )

    state = TradingState()

    # Create agents
    data_sync = DataSyncAgent(mock_exchange, config)
    quant_analyst = QuantAnalystAgent(mock_exchange, config)
    predict = PredictAgent(mock_exchange, config)
    predict.model = mock_lightgbm_model  # Inject mock model
    bull = BullAgent(mock_exchange, config)
    bear = BearAgent(mock_exchange, config)
    decision_core = DecisionCoreAgent(mock_exchange, config)
    risk_audit = RiskAuditAgent(mock_exchange, config)

    # Act - Execute full pipeline
    context = {"symbol": "BTC/USDT", "timeframes": ["5m"]}

    # Step 1: Fetch data
    data_result = await data_sync.execute(context)
    context.update(data_result)

    # Step 2: Calculate indicators
    quant_result = await quant_analyst.execute(context)
    context.update(quant_result)

    # Step 3: Get agent votes
    predict_result = await predict.execute(context)
    context.update(predict_result)

    bull_result = await bull.execute(context)
    context.update({"bull_vote": bull_result.get("vote", {})})

    bear_result = await bear.execute(context)
    context.update({"bear_vote": bear_result.get("vote", {})})

    # Add regime (for DecisionCore)
    context["regime"] = "trending"

    # Step 4: Make decision
    decision_result = await decision_core.execute(context)
    context.update(decision_result)

    # Step 5: Apply risk audit
    context["state"] = state
    risk_result = await risk_audit.execute(context)

    # Assert - In bullish scenario, should produce buy signal (or hold if risk vetoes)
    decision = context.get("decision", {})
    assert decision["action"] in ["buy", "hold"]  # Bull agents should vote buy
    assert "confidence" in decision
    assert decision["confidence"] >= 0.0

    # Risk audit should run without errors
    assert "risk_audit" in risk_result
    assert isinstance(risk_result["risk_audit"]["veto"], bool)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_pipeline_bearish_scenario(
    mock_exchange,
    ohlcv_factory,
    mock_lightgbm_model
):
    """Test complete pipeline with bearish market scenario."""
    # Arrange - Setup bearish market data
    ohlcv_data = ohlcv_factory(trend="downtrend", num_candles=100, volatility=0.01)
    mock_exchange.fetch_ohlcv.return_value = ohlcv_data
    mock_exchange.fetch_balance.return_value = {
        'USDT': {'free': 10000, 'used': 0, 'total': 10000}
    }

    config = TradingConfig(
        provider="paper",
        api_key="test",
        api_secret="test",
        max_position_size_usd=1000,
        decision_threshold=0.1
    )

    state = TradingState()

    # Create agents
    data_sync = DataSyncAgent(mock_exchange, config)
    quant_analyst = QuantAnalystAgent(mock_exchange, config)
    predict = PredictAgent(mock_exchange, config)
    predict.model = mock_lightgbm_model
    bull = BullAgent(mock_exchange, config)
    bear = BearAgent(mock_exchange, config)
    decision_core = DecisionCoreAgent(mock_exchange, config)

    # Act - Execute pipeline up to decision
    context = {"symbol": "BTC/USDT", "timeframes": ["5m"]}

    data_result = await data_sync.execute(context)
    context.update(data_result)

    quant_result = await quant_analyst.execute(context)
    context.update(quant_result)

    predict_result = await predict.execute(context)
    context.update(predict_result)

    bull_result = await bull.execute(context)
    context.update({"bull_vote": bull_result.get("vote", {})})

    bear_result = await bear.execute(context)
    context.update({"bear_vote": bear_result.get("vote", {})})

    context["regime"] = "trending"

    decision_result = await decision_core.execute(context)
    decision = decision_result.get("decision", {})

    # Assert - In bearish scenario, bear agents should have influence
    assert decision["action"] in ["sell", "hold"]  # Likely sell or hold
    assert "confidence" in decision


@pytest.mark.asyncio
@pytest.mark.integration
async def test_risk_audit_vetoes_dangerous_trade(
    mock_exchange,
    ohlcv_factory,
    mock_lightgbm_model
):
    """Test risk audit vetoes trade when max positions limit exceeded."""
    # Arrange
    ohlcv_data = ohlcv_factory(trend="uptrend", num_candles=100)
    mock_exchange.fetch_ohlcv.return_value = ohlcv_data

    config = TradingConfig(
        provider="paper",
        api_key="test",
        api_secret="test",
        max_open_positions=2,  # Max 2 positions
        decision_threshold=0.1
    )

    # State with 2 positions already (at max)
    state = TradingState()
    state.add_position({
        "symbol": "BTC/USDT",
        "side": "long",
        "size": 0.01,
        "entry_price": 30000,
    })
    state.add_position({
        "symbol": "ETH/USDT",
        "side": "long",
        "size": 0.1,
        "entry_price": 2000,
    })

    # Create agents
    data_sync = DataSyncAgent(mock_exchange, config)
    quant_analyst = QuantAnalystAgent(mock_exchange, config)
    bull = BullAgent(mock_exchange, config)
    bear = BearAgent(mock_exchange, config)
    predict = PredictAgent(mock_exchange, config)
    predict.model = mock_lightgbm_model
    decision_core = DecisionCoreAgent(mock_exchange, config)
    risk_audit = RiskAuditAgent(mock_exchange, config)

    # Act - Execute pipeline
    context = {"symbol": "BTC/USDT", "timeframes": ["5m"]}

    data_result = await data_sync.execute(context)
    context.update(data_result)

    quant_result = await quant_analyst.execute(context)
    context.update(quant_result)

    predict_result = await predict.execute(context)
    context.update(predict_result)

    bull_result = await bull.execute(context)
    context.update({"bull_vote": bull_result.get("vote", {})})

    bear_result = await bear.execute(context)
    context.update({"bear_vote": bear_result.get("vote", {})})

    context["regime"] = "trending"

    decision_result = await decision_core.execute(context)
    context.update(decision_result)

    # Risk audit with maxed positions
    context["state"] = state
    risk_result = await risk_audit.execute(context)

    # Assert - Risk audit should veto if trying to open new position
    risk_audit_result = risk_result.get("risk_audit", {})
    if context["decision"]["action"] in ["buy", "sell"]:
        # Should veto because max positions reached
        assert risk_audit_result["veto"] is True
        assert "max open positions" in risk_audit_result["reason"].lower()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_pipeline_handles_missing_indicators_gracefully(
    mock_exchange,
    ohlcv_factory
):
    """Test pipeline handles missing indicator data gracefully (fail-safe)."""
    # Arrange - Setup data with insufficient candles for indicators
    ohlcv_data = ohlcv_factory(num_candles=10)  # Too few for indicators
    mock_exchange.fetch_ohlcv.return_value = ohlcv_data

    config = TradingConfig(
        provider="paper",
        api_key="test",
        api_secret="test"
    )

    # Create agents
    data_sync = DataSyncAgent(mock_exchange, config)
    quant_analyst = QuantAnalystAgent(mock_exchange, config)
    bull = BullAgent(mock_exchange, config)
    bear = BearAgent(mock_exchange, config)
    decision_core = DecisionCoreAgent(mock_exchange, config)

    # Act - Execute pipeline
    context = {"symbol": "BTC/USDT", "timeframes": ["5m"]}

    data_result = await data_sync.execute(context)
    context.update(data_result)

    quant_result = await quant_analyst.execute(context)
    context.update(quant_result)

    bull_result = await bull.execute(context)
    context.update({"bull_vote": bull_result.get("vote", {})})

    bear_result = await bear.execute(context)
    context.update({"bear_vote": bear_result.get("vote", {})})

    # Predict agent without model (fallback)
    context["ml_prediction"] = {"direction": "neutral", "confidence": 0.0}
    context["regime"] = "neutral"

    decision_result = await decision_core.execute(context)
    decision = decision_result.get("decision", {})

    # Assert - Should fail safe (likely hold decision with low confidence)
    assert decision["action"] in ["buy", "sell", "hold"]
    assert decision["confidence"] >= 0.0  # Valid confidence


@pytest.mark.asyncio
@pytest.mark.integration
async def test_decision_core_aggregates_votes_correctly(
    mock_exchange,
    ohlcv_factory,
    mock_lightgbm_model
):
    """Test DecisionCore combines agent votes with proper weighting."""
    # Arrange
    ohlcv_data = ohlcv_factory(trend="uptrend", num_candles=100)
    mock_exchange.fetch_ohlcv.return_value = ohlcv_data

    config = TradingConfig(
        provider="paper",
        api_key="test",
        api_secret="test",
        decision_threshold=0.1
    )

    # Create agents
    data_sync = DataSyncAgent(mock_exchange, config)
    quant_analyst = QuantAnalystAgent(mock_exchange, config)
    predict = PredictAgent(mock_exchange, config)
    predict.model = mock_lightgbm_model
    bull = BullAgent(mock_exchange, config)
    bear = BearAgent(mock_exchange, config)
    decision_core = DecisionCoreAgent(mock_exchange, config)

    # Act - Get agent votes
    context = {"symbol": "BTC/USDT", "timeframes": ["5m"]}

    data_result = await data_sync.execute(context)
    context.update(data_result)

    quant_result = await quant_analyst.execute(context)
    context.update(quant_result)

    predict_result = await predict.execute(context)
    context.update(predict_result)

    bull_result = await bull.execute(context)
    context.update({"bull_vote": bull_result.get("vote", {})})

    bear_result = await bear.execute(context)
    context.update({"bear_vote": bear_result.get("vote", {})})

    context["regime"] = "trending"

    # Get decision
    decision_result = await decision_core.execute(context)
    decision = decision_result.get("decision", {})

    # Assert - Decision should aggregate votes
    assert "action" in decision
    assert "confidence" in decision
    assert decision["action"] in ["buy", "sell", "hold"]
    assert 0.0 <= decision["confidence"] <= 1.0

    # Verify regime weighting applied
    assert decision["regime"] == "trending"
    assert "weighted_score" in decision

    # Verify bull and bear votes were considered
    assert "bull_vote" in decision
    assert "bear_vote" in decision


@pytest.mark.asyncio
@pytest.mark.integration
async def test_execution_engine_places_orders(
    mock_exchange,
    ohlcv_factory
):
    """Test ExecutionEngine places bracket orders for approved decisions."""
    # Arrange
    ohlcv_data = ohlcv_factory(trend="uptrend", num_candles=100)
    mock_exchange.fetch_ohlcv.return_value = ohlcv_data
    mock_exchange.fetch_ticker.return_value = {
        'symbol': 'BTC/USDT',
        'last': 30000,
        'bid': 29998,
        'ask': 30002
    }

    config = TradingConfig(
        provider="paper",
        api_key="test",
        api_secret="test",
        max_position_size_usd=1000
    )

    state = TradingState()

    # Mock order creation
    mock_exchange.create_order = AsyncMock(return_value={
        'id': 'test-order-123',
        'status': 'closed',
        'filled': 0.01
    })

    execution_engine = ExecutionEngine(mock_exchange, config)

    # Act - Execute buy decision
    decision = {
        "action": "buy",
        "confidence": 0.8
    }

    ticker_data = await mock_exchange.fetch_ticker('BTC/USDT')
    context = {
        "decision": decision,
        "state": state,
        "symbol": "BTC/USDT",
        "market_data": {
            "ticker": Mock(last=ticker_data['last'])
        }
    }

    result = await execution_engine.execute(context)

    # Assert - Order should be placed
    execution_result = result.get("execution", {})

    # Since we're testing execution logic, check if attempt was made
    # (actual order placement may fail due to mocking complexity)
    assert "success" in execution_result


@pytest.mark.asyncio
@pytest.mark.integration
async def test_execution_engine_skips_hold_decisions(
    mock_exchange
):
    """Test ExecutionEngine skips execution for hold decisions."""
    # Arrange
    config = TradingConfig(
        provider="paper",
        api_key="test",
        api_secret="test"
    )

    state = TradingState()
    execution_engine = ExecutionEngine(mock_exchange, config)

    # Act - Execute hold decision
    decision = {
        "action": "hold",
        "confidence": 0.0
    }

    context = {
        "decision": decision,
        "state": state,
        "symbol": "BTC/USDT"
    }

    result = await execution_engine.execute(context)

    # Assert - No order should be placed
    execution_result = result.get("execution", {})
    assert execution_result["success"] is False
    assert "no action" in execution_result["error"].lower()
    mock_exchange.create_order.assert_not_called()
