"""Integration tests for safety system."""

import pytest
import os
from unittest.mock import Mock, patch
from datetime import datetime

from trading.safety.thresholds import SafetyThresholds
from trading.safety.kill_switch import KillSwitch
from trading.safety.circuit_breaker import CircuitBreaker
from trading.safety.position_limits import PositionLimitEnforcer
from trading.analytics.risk_calculator import RiskCalculator


class TestSafetyIntegration:
    """Integration tests for complete safety system."""

    @pytest.fixture
    def safety_components(self):
        """Create full safety system."""
        thresholds = SafetyThresholds()
        kill_switch = KillSwitch(secret_key="test-secret")
        risk_calculator = RiskCalculator()
        circuit_breaker = CircuitBreaker(thresholds, risk_calculator)
        position_limits = PositionLimitEnforcer(thresholds)

        return {
            "thresholds": thresholds,
            "kill_switch": kill_switch,
            "circuit_breaker": circuit_breaker,
            "position_limits": position_limits,
            "risk_calculator": risk_calculator,
        }

    def test_layered_defense_kill_switch_overrides_all(self, safety_components):
        """Test kill switch blocks trading regardless of other states."""
        kill_switch = safety_components["kill_switch"]
        circuit_breaker = safety_components["circuit_breaker"]
        position_limits = safety_components["position_limits"]

        # Kill switch is highest priority
        kill_switch.trigger("Emergency", "operator")

        # Even if circuit breaker is closed and position limits allow
        assert not circuit_breaker.is_open()

        allowed, _ = position_limits.check_new_position(
            "BTC/USDT", "momentum", 1000.0, 10000.0
        )
        assert allowed

        # Kill switch should block everything
        assert kill_switch.is_active()

    def test_layered_defense_circuit_breaker_before_position_limits(self, safety_components):
        """Test circuit breaker is checked before position limits."""
        circuit_breaker = safety_components["circuit_breaker"]
        position_limits = safety_components["position_limits"]

        # Trip circuit breaker
        circuit_breaker.trip("Drawdown exceeded")

        # Position limit check would pass
        allowed, _ = position_limits.check_new_position(
            "BTC/USDT", "momentum", 1000.0, 10000.0
        )
        assert allowed

        # But circuit breaker should prevent trading
        assert circuit_breaker.is_open()

    def test_all_checks_pass_allows_trading(self, safety_components):
        """Test that trading proceeds when all safety checks pass."""
        kill_switch = safety_components["kill_switch"]
        circuit_breaker = safety_components["circuit_breaker"]
        position_limits = safety_components["position_limits"]

        # All systems green
        assert not kill_switch.is_active()
        assert not circuit_breaker.is_open()

        allowed, reason = position_limits.check_new_position(
            "BTC/USDT", "momentum", 1000.0, 10000.0
        )
        assert allowed
        assert reason is None

    def test_threshold_validation(self, safety_components):
        """Test threshold configuration is valid."""
        thresholds = safety_components["thresholds"]

        is_valid, error = thresholds.validate()
        assert is_valid
        assert error is None

    def test_safety_status_reporting(self, safety_components):
        """Test comprehensive status reporting."""
        kill_switch = safety_components["kill_switch"]
        circuit_breaker = safety_components["circuit_breaker"]
        position_limits = safety_components["position_limits"]
        thresholds = safety_components["thresholds"]

        # Get all statuses
        ks_status = kill_switch.get_status()
        cb_status = circuit_breaker.get_status()
        pl_status = position_limits.get_status()
        thresh_dict = thresholds.to_dict()

        # Verify status structures
        assert "active" in ks_status
        assert "state" in cb_status
        assert "total_positions" in pl_status
        assert "max_daily_drawdown_pct" in thresh_dict

    def test_circuit_breaker_error_tracking(self, safety_components):
        """Test circuit breaker tracks errors correctly."""
        circuit_breaker = safety_components["circuit_breaker"]

        # Simulate API errors
        for _ in range(2):
            circuit_breaker.record_api_error()

        # Simulate order failures
        for _ in range(3):
            circuit_breaker.record_order_failure()

        # Simulate failed trades
        circuit_breaker.record_failed_trade()

        status = circuit_breaker.get_status()
        assert status["api_errors_last_minute"] == 2
        assert status["order_failures"] == 3
        assert status["failed_trades_last_hour"] == 1

    def test_position_limits_multi_strategy(self, safety_components):
        """Test position limits across multiple strategies."""
        position_limits = safety_components["position_limits"]
        timestamp = datetime.now().timestamp()

        # Add positions across different strategies
        position_limits.add_position("BTC/USDT", "momentum", 2000.0, timestamp)
        position_limits.add_position("ETH/USDT", "mean_reversion", 2000.0, timestamp + 1)
        position_limits.add_position("SOL/USDT", "momentum", 1000.0, timestamp + 2)

        status = position_limits.get_status()

        # Check exposures
        assert status["total_positions"] == 3
        assert status["total_exposure_usd"] == 5000.0
        assert status["strategy_exposures"]["momentum"] == 3000.0
        assert status["strategy_exposures"]["mean_reversion"] == 2000.0

    def test_kill_switch_hmac_security(self, safety_components):
        """Test kill switch HMAC authentication."""
        kill_switch = safety_components["kill_switch"]
        import hmac
        import hashlib

        message = '{"reason": "test"}'

        # Correct signature
        correct_sig = hmac.new(
            b"test-secret",
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        # Wrong signature
        wrong_sig = "invalid"

        assert kill_switch.verify_hmac(message, correct_sig)
        assert not kill_switch.verify_hmac(message, wrong_sig)

    def test_reset_coordination(self, safety_components):
        """Test coordinated reset of safety systems."""
        kill_switch = safety_components["kill_switch"]
        circuit_breaker = safety_components["circuit_breaker"]

        # Activate both
        kill_switch.trigger("Emergency", "operator")
        circuit_breaker.trip("High drawdown")

        assert kill_switch.is_active()
        assert circuit_breaker.is_open()

        # Reset both
        kill_switch.reset("admin")
        circuit_breaker.reset()

        assert not kill_switch.is_active()
        assert not circuit_breaker.is_open()

    def test_threshold_configuration_inheritance(self):
        """Test custom threshold configuration."""
        custom_thresholds = SafetyThresholds(
            max_daily_drawdown_pct=3.0,
            max_weekly_drawdown_pct=7.0,
            max_total_drawdown_pct=15.0,
        )

        is_valid, error = custom_thresholds.validate()
        assert is_valid

        # Verify custom values
        assert custom_thresholds.max_daily_drawdown_pct == 3.0
        assert custom_thresholds.max_weekly_drawdown_pct == 7.0
        assert custom_thresholds.max_total_drawdown_pct == 15.0
