"""Tests for position limit enforcer."""

import pytest
from datetime import datetime

from trading.safety.position_limits import PositionLimitEnforcer
from trading.safety.thresholds import SafetyThresholds


class TestPositionLimitEnforcer:
    """Test suite for PositionLimitEnforcer."""

    @pytest.fixture
    def thresholds(self):
        """Create test thresholds."""
        return SafetyThresholds(
            max_position_pct_per_symbol=0.3,  # 30%
            max_position_pct_per_strategy=0.6,  # 60%
            max_total_exposure_pct=0.9,  # 90%
            max_concurrent_positions=10,
        )

    @pytest.fixture
    def enforcer(self, thresholds):
        """Create position limit enforcer."""
        return PositionLimitEnforcer(thresholds)

    def test_initial_state(self, enforcer):
        """Test enforcer starts with no positions."""
        status = enforcer.get_status()
        assert status["total_positions"] == 0
        assert status["total_exposure_usd"] == 0.0
        assert len(status["strategy_exposures"]) == 0
        assert len(status["symbol_exposures"]) == 0

    def test_per_symbol_limit_pass(self, enforcer):
        """Test position under per-symbol limit is approved."""
        # 20% of portfolio ($2000 out of $10000)
        allowed, reason = enforcer.check_new_position(
            symbol="BTC/USDT",
            strategy="momentum",
            position_size_usd=2000.0,
            current_portfolio_value=10000.0,
        )

        assert allowed
        assert reason is None

    def test_per_symbol_limit_fail(self, enforcer):
        """Test position exceeding per-symbol limit is rejected."""
        # 40% of portfolio ($4000 out of $10000) - exceeds 30% limit
        allowed, reason = enforcer.check_new_position(
            symbol="BTC/USDT",
            strategy="momentum",
            position_size_usd=4000.0,
            current_portfolio_value=10000.0,
        )

        assert not allowed
        assert "Per-symbol limit exceeded" in reason
        assert "BTC/USDT" in reason

    def test_per_strategy_limit_pass(self, enforcer):
        """Test strategy exposure under limit is approved."""
        # Add first position (20% to momentum strategy)
        enforcer.add_position("BTC/USDT", "momentum", 2000.0, datetime.now().timestamp())

        # Add second position (30% more to momentum = 50% total, under 60% limit)
        allowed, reason = enforcer.check_new_position(
            symbol="ETH/USDT",
            strategy="momentum",
            position_size_usd=3000.0,
            current_portfolio_value=10000.0,
        )

        assert allowed
        assert reason is None

    def test_per_strategy_limit_fail(self, enforcer):
        """Test strategy exposure exceeding limit is rejected."""
        # Add first position (40% to momentum)
        enforcer.add_position("BTC/USDT", "momentum", 4000.0, datetime.now().timestamp())

        # Try to add 30% more (total 70%, exceeds 60% limit)
        allowed, reason = enforcer.check_new_position(
            symbol="ETH/USDT",
            strategy="momentum",
            position_size_usd=3000.0,
            current_portfolio_value=10000.0,
        )

        assert not allowed
        assert "Per-strategy limit exceeded" in reason
        assert "momentum" in reason

    def test_portfolio_wide_limit_pass(self, enforcer):
        """Test total exposure under limit is approved."""
        # Add positions totaling 80% (under 90% limit)
        enforcer.add_position("BTC/USDT", "momentum", 3000.0, datetime.now().timestamp())
        enforcer.add_position("ETH/USDT", "mean_reversion", 3000.0, datetime.now().timestamp())
        enforcer.add_position("SOL/USDT", "momentum", 2000.0, datetime.now().timestamp())

        # Try to add 5% more (total 85%, under 90%)
        allowed, reason = enforcer.check_new_position(
            symbol="ADA/USDT",
            strategy="mean_reversion",
            position_size_usd=500.0,
            current_portfolio_value=10000.0,
        )

        assert allowed
        assert reason is None

    def test_portfolio_wide_limit_fail(self, enforcer):
        """Test total exposure exceeding limit is rejected."""
        # Add positions totaling 80%
        enforcer.add_position("BTC/USDT", "momentum", 3000.0, datetime.now().timestamp())
        enforcer.add_position("ETH/USDT", "mean_reversion", 3000.0, datetime.now().timestamp())
        enforcer.add_position("SOL/USDT", "momentum", 2000.0, datetime.now().timestamp())

        # Try to add 15% more (total 95%, exceeds 90%)
        allowed, reason = enforcer.check_new_position(
            symbol="ADA/USDT",
            strategy="mean_reversion",
            position_size_usd=1500.0,
            current_portfolio_value=10000.0,
        )

        assert not allowed
        assert "Total exposure limit exceeded" in reason

    def test_max_concurrent_positions_pass(self, enforcer):
        """Test position count under limit is approved."""
        # Add 9 positions (under 10 limit)
        symbols = ["BTC", "ETH", "SOL", "ADA", "DOT", "LINK", "AVAX", "MATIC", "UNI"]
        for i, symbol in enumerate(symbols):
            enforcer.add_position(
                f"{symbol}/USDT",
                "momentum",
                100.0,  # Small positions to avoid other limits
                datetime.now().timestamp() + i,
            )

        # Try to add 10th position (exactly at limit)
        allowed, reason = enforcer.check_new_position(
            symbol="ATOM/USDT",
            strategy="momentum",
            position_size_usd=100.0,
            current_portfolio_value=10000.0,
        )

        assert allowed
        assert reason is None

    def test_max_concurrent_positions_fail(self, enforcer):
        """Test position count exceeding limit is rejected."""
        # Add 10 positions (at limit)
        symbols = ["BTC", "ETH", "SOL", "ADA", "DOT", "LINK", "AVAX", "MATIC", "UNI", "ATOM"]
        for i, symbol in enumerate(symbols):
            enforcer.add_position(
                f"{symbol}/USDT",
                "momentum",
                100.0,
                datetime.now().timestamp() + i,
            )

        # Try to add 11th position
        allowed, reason = enforcer.check_new_position(
            symbol="FTM/USDT",
            strategy="momentum",
            position_size_usd=100.0,
            current_portfolio_value=10000.0,
        )

        assert not allowed
        assert "Max concurrent positions limit exceeded" in reason

    def test_add_position_tracking(self, enforcer):
        """Test position is tracked after adding."""
        enforcer.add_position("BTC/USDT", "momentum", 1000.0, datetime.now().timestamp())

        status = enforcer.get_status()
        assert status["total_positions"] == 1
        assert status["total_exposure_usd"] == 1000.0
        assert "momentum" in status["strategy_exposures"]
        assert status["strategy_exposures"]["momentum"] == 1000.0
        assert "BTC/USDT" in status["symbol_exposures"]
        assert status["symbol_exposures"]["BTC/USDT"] == 1000.0

    def test_remove_position_tracking(self, enforcer):
        """Test position is removed from tracking."""
        timestamp = datetime.now().timestamp()
        enforcer.add_position("BTC/USDT", "momentum", 1000.0, timestamp)

        enforcer.remove_position("BTC/USDT", "momentum", 1000.0)

        status = enforcer.get_status()
        assert status["total_positions"] == 0
        assert status["total_exposure_usd"] == 0.0
        assert "momentum" not in status["strategy_exposures"]
        assert "BTC/USDT" not in status["symbol_exposures"]

    def test_multiple_positions_same_symbol(self, enforcer):
        """Test multiple positions for same symbol accumulate exposure."""
        # Add two positions for BTC/USDT (different strategies)
        enforcer.add_position("BTC/USDT", "momentum", 1000.0, datetime.now().timestamp())
        enforcer.add_position("BTC/USDT", "mean_reversion", 1000.0, datetime.now().timestamp())

        # Try to add third position - should check total symbol exposure
        # Total would be 3000/10000 = 30%, exactly at limit
        allowed, reason = enforcer.check_new_position(
            symbol="BTC/USDT",
            strategy="scalping",
            position_size_usd=1000.0,
            current_portfolio_value=10000.0,
        )

        # Should be allowed (exactly at 30% limit)
        assert allowed

    def test_layer_check_order(self, enforcer):
        """Test that all four layers are checked in correct order."""
        # Set up to violate per-symbol limit first
        allowed, reason = enforcer.check_new_position(
            symbol="BTC/USDT",
            strategy="momentum",
            position_size_usd=4000.0,  # 40%, violates 30% per-symbol limit
            current_portfolio_value=10000.0,
        )

        # Should fail on per-symbol check (first layer)
        assert not allowed
        assert "Per-symbol" in reason

    def test_invalid_inputs(self, enforcer):
        """Test validation of invalid inputs."""
        # Zero portfolio value
        allowed, reason = enforcer.check_new_position(
            symbol="BTC/USDT",
            strategy="momentum",
            position_size_usd=1000.0,
            current_portfolio_value=0.0,
        )
        assert not allowed

        # Negative position size
        allowed, reason = enforcer.check_new_position(
            symbol="BTC/USDT",
            strategy="momentum",
            position_size_usd=-1000.0,
            current_portfolio_value=10000.0,
        )
        assert not allowed
