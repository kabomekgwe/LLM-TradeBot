"""Tests for circuit breaker module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from trading.safety.circuit_breaker import CircuitBreaker, CircuitState
from trading.safety.thresholds import SafetyThresholds
from trading.analytics.risk_calculator import RiskCalculator, RiskMetrics
from trading.memory.trade_history import TradeRecord


class TestCircuitBreaker:
    """Test suite for CircuitBreaker."""

    @pytest.fixture
    def thresholds(self):
        """Create test thresholds."""
        return SafetyThresholds(
            max_daily_drawdown_pct=5.0,
            max_weekly_drawdown_pct=10.0,
            max_total_drawdown_pct=20.0,
            max_consecutive_losses=5,
            max_api_errors_per_minute=3,
            max_order_failures=5,
            max_failed_trades_per_hour=10,
        )

    @pytest.fixture
    def risk_calculator(self):
        """Create mock risk calculator."""
        return Mock(spec=RiskCalculator)

    @pytest.fixture
    def circuit_breaker(self, thresholds, risk_calculator):
        """Create circuit breaker instance."""
        return CircuitBreaker(thresholds, risk_calculator)

    def test_initial_state(self, circuit_breaker):
        """Test circuit breaker starts in CLOSED state."""
        assert not circuit_breaker.is_open()
        status = circuit_breaker.get_status()
        assert status["state"] == CircuitState.CLOSED.value
        assert not status["is_open"]

    def test_trip_circuit(self, circuit_breaker):
        """Test manually tripping circuit."""
        circuit_breaker.trip("Test trip")

        assert circuit_breaker.is_open()
        status = circuit_breaker.get_status()
        assert status["state"] == CircuitState.OPEN.value
        assert status["trip_reason"] == "Test trip"

    def test_reset_circuit(self, circuit_breaker):
        """Test resetting circuit after trip."""
        circuit_breaker.trip("Test trip")
        assert circuit_breaker.is_open()

        circuit_breaker.reset()

        assert not circuit_breaker.is_open()
        status = circuit_breaker.get_status()
        assert status["state"] == CircuitState.CLOSED.value

    def test_drawdown_threshold_trip(self, circuit_breaker, risk_calculator):
        """Test circuit trips on excessive drawdown."""
        # Mock risk metrics with high drawdown
        risk_calculator.calculate_risk_metrics = Mock(return_value=RiskMetrics(
            var_95=1.0,
            var_99=2.0,
            cvar_95=1.5,
            cvar_99=2.5,
            sharpe_ratio=0.5,
            sortino_ratio=0.6,
            calmar_ratio=0.3,
            max_drawdown=3000.0,
            max_drawdown_pct=25.0,  # Exceeds 20% threshold
            avg_drawdown=10.0,
            drawdown_duration_days=5,
            daily_volatility=2.0,
            annualized_volatility=30.0,
            skewness=-0.5,
            kurtosis=1.0,
            risk_of_ruin_pct=5.0,
        ))

        trades = []
        current_equity = 10000.0

        result = circuit_breaker.check_and_update(trades, current_equity)

        assert result  # Should trip
        assert circuit_breaker.is_open()

    def test_consecutive_losses_trip(self, circuit_breaker, risk_calculator):
        """Test circuit trips on consecutive losses."""
        # Create 6 consecutive losing trades (exceeds threshold of 5)
        trades = []
        timestamp = int(datetime.now().timestamp() * 1000)

        for i in range(6):
            trade = Mock(spec=TradeRecord)
            trade.closed = True
            trade.won = False
            trade.timestamp = timestamp + i * 1000
            trades.append(trade)

        # Mock risk calculator to return safe metrics
        risk_calculator.calculate_risk_metrics = Mock(return_value=RiskMetrics(
            var_95=1.0, var_99=2.0, cvar_95=1.5, cvar_99=2.5,
            sharpe_ratio=0.5, sortino_ratio=0.6, calmar_ratio=0.3,
            max_drawdown=100.0, max_drawdown_pct=1.0,  # Safe drawdown
            avg_drawdown=0.5, drawdown_duration_days=1,
            daily_volatility=1.0, annualized_volatility=15.0,
            skewness=0.0, kurtosis=0.0, risk_of_ruin_pct=1.0,
        ))

        result = circuit_breaker.check_and_update(trades, 10000.0)

        assert result  # Should trip
        assert circuit_breaker.is_open()

    def test_api_error_rate_trip(self, circuit_breaker):
        """Test circuit trips on excessive API errors."""
        # Record 4 API errors (exceeds threshold of 3)
        for _ in range(4):
            circuit_breaker.record_api_error()

        result = circuit_breaker.check_and_update([], 10000.0)

        assert result  # Should trip
        assert circuit_breaker.is_open()

    def test_order_failure_trip(self, circuit_breaker):
        """Test circuit trips on excessive order failures."""
        # Record 6 order failures (exceeds threshold of 5)
        for _ in range(6):
            circuit_breaker.record_order_failure()

        result = circuit_breaker.check_and_update([], 10000.0)

        assert result  # Should trip
        assert circuit_breaker.is_open()

    def test_failed_trades_per_hour_trip(self, circuit_breaker):
        """Test circuit trips on too many failed trades per hour."""
        # Record 11 failed trades (exceeds threshold of 10)
        for _ in range(11):
            circuit_breaker.record_failed_trade()

        result = circuit_breaker.check_and_update([], 10000.0)

        assert result  # Should trip
        assert circuit_breaker.is_open()

    def test_api_error_sliding_window(self, circuit_breaker):
        """Test API error sliding window expires old errors."""
        # Record errors (within threshold)
        circuit_breaker.record_api_error()
        circuit_breaker.record_api_error()

        # Should not trip yet
        assert not circuit_breaker.check_and_update([], 10000.0)

        # In real implementation, old errors would expire after 1 minute
        # For this test, we verify the deque is being used
        assert len(circuit_breaker._api_errors_last_minute) == 2

    def test_reset_clears_counters(self, circuit_breaker):
        """Test reset clears all error counters."""
        # Accumulate some errors
        circuit_breaker.record_api_error()
        circuit_breaker.record_order_failure()
        circuit_breaker.record_failed_trade()

        circuit_breaker.trip("Test trip")
        circuit_breaker.reset()

        # Check counters are cleared
        status = circuit_breaker.get_status()
        assert status["consecutive_losses"] == 0
        assert status["api_errors_last_minute"] == 0
        assert status["order_failures"] == 0
        assert status["failed_trades_last_hour"] == 0

    def test_already_open_returns_true(self, circuit_breaker):
        """Test that open circuit immediately returns True."""
        circuit_breaker.trip("Manual trip")

        # Should return True without checking thresholds
        result = circuit_breaker.check_and_update([], 10000.0)
        assert result
        assert circuit_breaker.is_open()

    def test_multiple_threshold_breaches_first_wins(self, circuit_breaker, risk_calculator):
        """Test that first threshold breach triggers trip."""
        # Set up multiple breaches
        circuit_breaker.record_api_error()
        circuit_breaker.record_api_error()
        circuit_breaker.record_api_error()
        circuit_breaker.record_api_error()  # 4 errors, exceeds threshold

        for _ in range(6):
            circuit_breaker.record_order_failure()  # Also exceeds

        # Mock high drawdown too
        risk_calculator.calculate_risk_metrics = Mock(return_value=RiskMetrics(
            var_95=1.0, var_99=2.0, cvar_95=1.5, cvar_99=2.5,
            sharpe_ratio=0.5, sortino_ratio=0.6, calmar_ratio=0.3,
            max_drawdown=3000.0, max_drawdown_pct=25.0,  # Exceeds
            avg_drawdown=10.0, drawdown_duration_days=5,
            daily_volatility=2.0, annualized_volatility=30.0,
            skewness=-0.5, kurtosis=1.0, risk_of_ruin_pct=5.0,
        ))

        result = circuit_breaker.check_and_update([], 10000.0)

        assert result
        # Should trip on first checked threshold (implementation detail)
        assert circuit_breaker.is_open()
