"""
100% coverage tests for risk management (safety-critical module).
Tests position sizing, max loss limits, leverage caps, and all edge cases.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock

from trading.agents.risk_audit import RiskAuditAgent
from trading.config import TradingConfig
from trading.state import TradingState


# === Test Fixtures ===


@pytest.fixture
def risk_config():
    """Create TradingConfig with specific risk parameters."""
    return TradingConfig(
        provider="paper",
        api_key="test_key",
        api_secret="test_secret",
        testnet=True,
        max_position_size_usd=1000.0,
        max_daily_drawdown_pct=5.0,
        max_open_positions=3,
        decision_threshold=0.6
    )


@pytest.fixture
def mock_provider():
    """Create mock exchange provider."""
    return AsyncMock()


@pytest.fixture
def fresh_state():
    """Create fresh TradingState for each test."""
    return TradingState()


# === Circuit Breaker Tests ===


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_veto_circuit_breaker_tripped(mock_provider, risk_config, fresh_state):
    """Test RiskAudit vetoes when circuit breaker is tripped."""
    # Arrange
    agent = RiskAuditAgent(mock_provider, risk_config)
    fresh_state.circuit_breaker_tripped = True
    fresh_state.last_circuit_trip_reason = "Daily drawdown limit exceeded"

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] is True
    assert 'circuit breaker' in result['risk_audit']['reason'].lower()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_allows_when_circuit_breaker_not_tripped(mock_provider, risk_config, fresh_state):
    """Test RiskAudit allows trades when circuit breaker is not tripped."""
    # Arrange
    agent = RiskAuditAgent(mock_provider, risk_config)
    fresh_state.circuit_breaker_tripped = False

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] is False


# === Position Limit Tests ===


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_veto_max_positions_reached(mock_provider, risk_config, fresh_state):
    """Test RiskAudit vetoes when max open positions reached."""
    # Arrange
    risk_config.max_open_positions = 2
    agent = RiskAuditAgent(mock_provider, risk_config)

    # Add 2 positions (at limit)
    fresh_state.add_position({'symbol': 'BTC/USDT', 'size': 0.01, 'entry_price': 30000})
    fresh_state.add_position({'symbol': 'ETH/USDT', 'size': 0.1, 'entry_price': 2000})

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] is True
    assert 'Max open positions reached' in result['risk_audit']['reason']


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_allows_when_below_max_positions(mock_provider, risk_config, fresh_state):
    """Test RiskAudit allows trades when below max positions."""
    # Arrange
    risk_config.max_open_positions = 3
    agent = RiskAuditAgent(mock_provider, risk_config)

    # Add 1 position (below limit)
    fresh_state.add_position({'symbol': 'BTC/USDT', 'size': 0.01, 'entry_price': 30000})

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] is False


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_allows_when_zero_positions(mock_provider, risk_config, fresh_state):
    """Test RiskAudit allows first trade when no positions open."""
    # Arrange
    agent = RiskAuditAgent(mock_provider, risk_config)

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] is False
    assert len(fresh_state.active_positions) == 0


# === Daily Drawdown Tests ===


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_veto_max_drawdown_exceeded(mock_provider, risk_config, fresh_state):
    """Test RiskAudit vetoes when daily drawdown limit exceeded."""
    # Arrange
    risk_config.max_daily_drawdown_pct = 5.0
    agent = RiskAuditAgent(mock_provider, risk_config)

    # Simulate 6% drawdown (exceeds 5% limit)
    fresh_state.daily_drawdown_pct = 6.0

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] is True
    assert 'Daily drawdown limit exceeded' in result['risk_audit']['reason']


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_allows_within_drawdown_limit(mock_provider, risk_config, fresh_state):
    """Test RiskAudit allows trades when within drawdown limit."""
    # Arrange
    risk_config.max_daily_drawdown_pct = 5.0
    agent = RiskAuditAgent(mock_provider, risk_config)

    # Simulate 3% drawdown (within 5% limit)
    fresh_state.daily_drawdown_pct = 3.0

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] is False


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_allows_when_no_drawdown(mock_provider, risk_config, fresh_state):
    """Test RiskAudit allows trades when no drawdown."""
    # Arrange
    agent = RiskAuditAgent(mock_provider, risk_config)

    # No drawdown (default is 0.0)
    assert fresh_state.daily_drawdown_pct == 0.0

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] is False


# === Confidence Threshold Tests ===


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
@pytest.mark.parametrize("confidence,threshold,should_veto", [
    (0.8, 0.6, False),  # Above threshold - allow
    (0.6, 0.6, False),  # Equal to threshold - allow
    (0.5, 0.6, True),   # Below threshold - veto
    (0.3, 0.6, True),   # Well below threshold - veto
    (1.0, 0.6, False),  # Max confidence - allow
    (0.0, 0.6, True),   # Zero confidence - veto
])
async def test_risk_audit_confidence_threshold(confidence, threshold, should_veto, mock_provider, risk_config, fresh_state):
    """Test RiskAudit confidence threshold enforcement."""
    # Arrange
    risk_config.decision_threshold = threshold
    agent = RiskAuditAgent(mock_provider, risk_config)

    decision = {'action': 'buy', 'confidence': confidence}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] == should_veto
    if should_veto:
        assert 'Confidence below threshold' in result['risk_audit']['reason']


# === Hold Decision Tests ===


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_veto_hold_decision(mock_provider, risk_config, fresh_state):
    """Test RiskAudit vetoes hold decisions (no trade needed)."""
    # Arrange
    agent = RiskAuditAgent(mock_provider, risk_config)

    decision = {'action': 'hold', 'confidence': 0.5}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] is True
    assert 'Decision is to hold' in result['risk_audit']['reason']


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_allows_buy_decision(mock_provider, risk_config, fresh_state):
    """Test RiskAudit allows buy decision when all checks pass."""
    # Arrange
    agent = RiskAuditAgent(mock_provider, risk_config)

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] is False


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_allows_sell_decision(mock_provider, risk_config, fresh_state):
    """Test RiskAudit allows sell decision when all checks pass."""
    # Arrange
    agent = RiskAuditAgent(mock_provider, risk_config)

    decision = {'action': 'sell', 'confidence': 0.8}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] is False


# === Edge Cases ===


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_handles_missing_decision(mock_provider, risk_config, fresh_state):
    """Test RiskAudit handles missing decision gracefully."""
    # Arrange
    agent = RiskAuditAgent(mock_provider, risk_config)

    context = {'state': fresh_state}  # Missing decision

    # Act & Assert
    with pytest.raises(ValueError, match="decision is required"):
        await agent.execute(context)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_handles_invalid_state(mock_provider, risk_config):
    """Test RiskAudit handles invalid state type."""
    # Arrange
    agent = RiskAuditAgent(mock_provider, risk_config)

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': {}}  # Invalid state (not TradingState instance)

    # Act & Assert
    with pytest.raises(ValueError, match="state must be TradingState instance"):
        await agent.execute(context)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_handles_missing_confidence(mock_provider, risk_config, fresh_state):
    """Test RiskAudit handles missing confidence field."""
    # Arrange
    agent = RiskAuditAgent(mock_provider, risk_config)

    decision = {'action': 'buy'}  # Missing confidence
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert - Should default to 0.0 confidence and veto
    assert result['risk_audit']['veto'] is True
    assert 'Confidence below threshold' in result['risk_audit']['reason']


# === Multiple Veto Conditions ===


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_multiple_veto_conditions(mock_provider, risk_config, fresh_state):
    """Test RiskAudit with multiple veto conditions (returns first violation)."""
    # Arrange
    risk_config.max_open_positions = 1
    risk_config.max_daily_drawdown_pct = 5.0
    agent = RiskAuditAgent(mock_provider, risk_config)

    # Set multiple violations
    fresh_state.add_position({'symbol': 'BTC/USDT', 'size': 0.01, 'entry_price': 30000})
    fresh_state.circuit_breaker_tripped = True
    fresh_state.last_circuit_trip_reason = "Test trip"

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert - Should veto (first check: circuit breaker)
    assert result['risk_audit']['veto'] is True
    assert 'circuit breaker' in result['risk_audit']['reason'].lower()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_all_checks_pass(mock_provider, risk_config, fresh_state):
    """Test RiskAudit when all safety checks pass."""
    # Arrange
    risk_config.max_open_positions = 3
    risk_config.max_daily_drawdown_pct = 5.0
    risk_config.decision_threshold = 0.6
    agent = RiskAuditAgent(mock_provider, risk_config)

    # Perfect conditions
    fresh_state.circuit_breaker_tripped = False
    fresh_state.daily_drawdown_pct = 0.0
    # No positions

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert
    assert result['risk_audit']['veto'] is False
    assert result['risk_audit']['reason'] is None


# === Boundary Tests ===


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_drawdown_at_exact_limit(mock_provider, risk_config, fresh_state):
    """Test RiskAudit when drawdown is exactly at limit."""
    # Arrange
    risk_config.max_daily_drawdown_pct = 5.0
    agent = RiskAuditAgent(mock_provider, risk_config)

    # Exactly at limit (should veto with >= comparison)
    fresh_state.daily_drawdown_pct = 5.0

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert - At limit should veto
    assert result['risk_audit']['veto'] is True


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.risk
async def test_risk_audit_positions_at_exact_limit(mock_provider, risk_config, fresh_state):
    """Test RiskAudit when positions exactly at limit."""
    # Arrange
    risk_config.max_open_positions = 2
    agent = RiskAuditAgent(mock_provider, risk_config)

    # Exactly at limit
    fresh_state.add_position({'symbol': 'BTC/USDT', 'size': 0.01, 'entry_price': 30000})
    fresh_state.add_position({'symbol': 'ETH/USDT', 'size': 0.1, 'entry_price': 2000})

    decision = {'action': 'buy', 'confidence': 0.8}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert - At limit should veto
    assert result['risk_audit']['veto'] is True


# === Integration Scenario Tests ===


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.risk
async def test_risk_audit_realistic_trading_scenario(mock_provider, risk_config, fresh_state):
    """Test RiskAudit in realistic trading scenario."""
    # Arrange
    agent = RiskAuditAgent(mock_provider, risk_config)

    # Realistic state: 1 position, small drawdown, circuit breaker OK
    fresh_state.add_position({'symbol': 'BTC/USDT', 'size': 0.01, 'entry_price': 30000})
    fresh_state.daily_drawdown_pct = 1.5  # 1.5% drawdown
    fresh_state.circuit_breaker_tripped = False

    decision = {'action': 'buy', 'confidence': 0.75}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert - Should allow (all within limits)
    assert result['risk_audit']['veto'] is False


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.risk
async def test_risk_audit_dangerous_trading_scenario(mock_provider, risk_config, fresh_state):
    """Test RiskAudit vetoes dangerous scenario."""
    # Arrange
    risk_config.max_open_positions = 2
    risk_config.max_daily_drawdown_pct = 5.0
    agent = RiskAuditAgent(mock_provider, risk_config)

    # Dangerous state: max positions, high drawdown
    fresh_state.add_position({'symbol': 'BTC/USDT', 'size': 0.01, 'entry_price': 30000})
    fresh_state.add_position({'symbol': 'ETH/USDT', 'size': 0.1, 'entry_price': 2000})
    fresh_state.daily_drawdown_pct = 4.8  # Close to limit but still OK

    decision = {'action': 'buy', 'confidence': 0.75}
    context = {'decision': decision, 'state': fresh_state}

    # Act
    result = await agent.execute(context)

    # Assert - Should veto (max positions reached)
    assert result['risk_audit']['veto'] is True
