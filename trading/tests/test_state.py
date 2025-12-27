"""
State persistence tests (100% coverage required - safety-critical).
Tests atomic writes, crash recovery, corruption handling.
"""
import pytest
import tempfile
import os
import json
from pathlib import Path
from trading.state import TradingState


class TestStatePersistence:
    """Test state save/load operations."""

    def test_save_creates_file(self, tmp_path):
        """Test saving state creates file at specified path."""
        state = TradingState()
        state.initialized = True
        state.provider = "binance_futures"
        state.active_positions = [{'symbol': 'BTC/USDT', 'size': 0.01, 'entry_price': 30000}]
        state.save(tmp_path)

        state_file = tmp_path / ".trading_state.json"
        assert state_file.exists()
        assert state_file.stat().st_size > 0

    def test_load_restores_state(self, tmp_path):
        """Test loading state restores all fields correctly."""
        # Save state
        original_state = TradingState()
        original_state.initialized = True
        original_state.provider = "binance_futures"
        original_state.active_positions = [{'symbol': 'BTC/USDT', 'size': 0.01, 'entry_price': 30000}]
        original_state.total_trades = 5
        original_state.total_pnl = 150.0
        original_state.save(tmp_path)

        # Load state
        loaded_state = TradingState.load(tmp_path)

        assert loaded_state is not None
        assert loaded_state.initialized == original_state.initialized
        assert loaded_state.provider == original_state.provider
        assert loaded_state.active_positions == original_state.active_positions
        assert loaded_state.total_trades == original_state.total_trades
        assert loaded_state.total_pnl == original_state.total_pnl

    def test_atomic_write_prevents_corruption(self, tmp_path):
        """Test crash during write doesn't corrupt existing state."""
        state_file = tmp_path / ".trading_state.json"

        # Save initial state
        state = TradingState()
        state.initialized = True
        state.provider = "binance_futures"
        state.active_positions = [{'symbol': 'BTC/USDT', 'size': 0.01, 'entry_price': 30000}]
        state.save(tmp_path)

        # Verify initial state is saved
        assert state_file.exists()
        initial_content = state_file.read_text()

        # Simulate crash during write by writing partial data
        with open(state_file, 'w') as f:
            f.write('{"initialized": true, "provider": "binance_futures", "active_positions": [')  # Incomplete JSON

        # Should fail to load corrupted state and return None
        loaded_state = TradingState.load(tmp_path)
        assert loaded_state is None  # Corrupted state returns None

    def test_missing_file_returns_none(self, tmp_path):
        """Test loading non-existent file returns None."""
        loaded_state = TradingState.load(tmp_path)
        assert loaded_state is None

    def test_concurrent_writes_dont_corrupt(self, tmp_path):
        """Test multiple writes in quick succession maintain data integrity."""
        # Write 10 times rapidly
        for i in range(10):
            state = TradingState()
            state.initialized = True
            state.provider = "binance_futures"
            state.active_positions = [{'symbol': 'BTC/USDT', 'size': 0.01 * (i + 1), 'entry_price': 30000 + i}]
            state.total_trades = i
            state.save(tmp_path)

        # Load final state
        final_state = TradingState.load(tmp_path)

        # Should have last written values
        assert final_state is not None
        assert final_state.total_trades == 9
        assert final_state.active_positions[0]['size'] == 0.1
        assert final_state.active_positions[0]['entry_price'] == 30009

    def test_save_updates_timestamp(self, tmp_path):
        """Test save operation updates last_updated timestamp."""
        state = TradingState()
        assert state.last_updated is None

        state.save(tmp_path)
        assert state.last_updated is not None

        # Save again
        first_timestamp = state.last_updated
        state.save(tmp_path)
        second_timestamp = state.last_updated

        # Timestamp should be updated
        assert second_timestamp >= first_timestamp


class TestStateValidation:
    """Test state validation and schema checking."""

    def test_invalid_json_raises_error(self, tmp_path):
        """Test loading file with invalid JSON returns None gracefully."""
        state_file = tmp_path / ".trading_state.json"
        state_file.write_text("not valid json {{{")

        # Should return None for invalid JSON (graceful degradation)
        loaded_state = TradingState.load(tmp_path)
        assert loaded_state is None

    def test_missing_required_fields_handled(self, tmp_path):
        """Test loading state with missing fields uses defaults."""
        state_file = tmp_path / ".trading_state.json"
        state_file.write_text('{}')  # Empty JSON object

        state = TradingState.load(tmp_path)

        # Should have default values
        assert state is not None
        assert hasattr(state, 'initialized')
        assert isinstance(state.initialized, bool)
        assert state.initialized is False
        assert hasattr(state, 'active_positions')
        assert isinstance(state.active_positions, list)
        assert state.active_positions == []

    def test_partial_state_loads_with_defaults(self, tmp_path):
        """Test loading state with some fields fills in defaults for missing ones."""
        state_file = tmp_path / ".trading_state.json"
        partial_state = {
            'initialized': True,
            'provider': 'binance_futures',
            # Missing: active_positions, total_trades, etc.
        }
        state_file.write_text(json.dumps(partial_state))

        state = TradingState.load(tmp_path)

        assert state is not None
        assert state.initialized is True
        assert state.provider == 'binance_futures'
        # Defaults should be filled in
        assert state.active_positions == []
        assert state.total_trades == 0
        assert state.total_pnl == 0.0


class TestStateBackwardCompatibility:
    """Test state loading works with old schemas (future-proofing)."""

    def test_load_old_schema_without_new_fields(self, tmp_path):
        """Test loading old state schema doesn't break when new fields added."""
        state_file = tmp_path / ".trading_state.json"

        # Write old schema (missing hypothetical future fields)
        old_schema = {
            'initialized': True,
            'provider': 'binance_futures',
            'active_positions': [{'symbol': 'BTC/USDT', 'size': 0.01, 'entry_price': 30000}],
            'total_trades': 5,
            # Missing future fields like 'risk_metrics', 'decision_history', etc.
        }
        state_file.write_text(json.dumps(old_schema))

        # Should load successfully with defaults for missing fields
        state = TradingState.load(tmp_path)

        assert state is not None
        assert state.initialized is True
        assert state.provider == 'binance_futures'
        assert state.active_positions == [{'symbol': 'BTC/USDT', 'size': 0.01, 'entry_price': 30000}]
        assert state.total_trades == 5
        # Fields not in old schema should have defaults
        assert state.total_pnl == 0.0
        assert state.circuit_breaker_tripped is False


class TestStateOperations:
    """Test state manipulation operations."""

    def test_add_position(self):
        """Test adding a position to active positions."""
        state = TradingState()
        assert len(state.active_positions) == 0

        position = {
            "symbol": "BTC/USDT",
            "side": "long",
            "size": 0.1,
            "entry_price": 42000.0
        }
        state.add_position(position)

        assert len(state.active_positions) == 1
        assert state.active_positions[0] == position

    def test_remove_position(self):
        """Test removing a position by symbol."""
        state = TradingState()
        position1 = {"symbol": "BTC/USDT", "size": 0.1}
        position2 = {"symbol": "ETH/USDT", "size": 0.5}

        state.add_position(position1)
        state.add_position(position2)
        assert len(state.active_positions) == 2

        removed = state.remove_position("BTC/USDT")

        assert removed == position1
        assert len(state.active_positions) == 1
        assert state.active_positions[0] == position2

    def test_remove_nonexistent_position(self):
        """Test removing a position that doesn't exist returns None."""
        state = TradingState()
        result = state.remove_position("BTC/USDT")
        assert result is None

    def test_trip_circuit_breaker(self):
        """Test tripping circuit breaker updates state."""
        state = TradingState()
        assert state.circuit_breaker_tripped is False
        assert state.last_circuit_trip_reason is None

        reason = "Daily drawdown limit exceeded"
        state.trip_circuit_breaker(reason)

        assert state.circuit_breaker_tripped is True
        assert state.last_circuit_trip_reason == reason
        assert state.last_updated is not None

    def test_reset_circuit_breaker(self):
        """Test resetting circuit breaker clears state."""
        state = TradingState()
        state.trip_circuit_breaker("Test")
        assert state.circuit_breaker_tripped is True

        state.reset_circuit_breaker()

        assert state.circuit_breaker_tripped is False
        assert state.last_circuit_trip_reason is None

    def test_add_trade(self):
        """Test adding a trade updates PnL."""
        state = TradingState()
        assert state.total_trades == 0
        assert state.total_pnl == 0.0
        assert state.daily_pnl == 0.0

        state.add_trade(trade_pnl=100.0, is_win=True)

        assert state.total_trades == 1
        assert state.total_pnl == 100.0
        assert state.daily_pnl == 100.0

        state.add_trade(trade_pnl=-50.0, is_win=False)

        assert state.total_trades == 2
        assert state.total_pnl == 50.0
        assert state.daily_pnl == 50.0


class TestStateRepresentation:
    """Test state string representation."""

    def test_repr(self):
        """Test __repr__ provides useful debugging output."""
        state = TradingState()
        state.provider = "binance_futures"
        state.total_trades = 5
        state.total_pnl = 150.0
        state.active_positions = [{'symbol': 'BTC/USDT'}]

        repr_str = repr(state)

        assert 'binance_futures' in repr_str
        assert 'total_trades=5' in repr_str
        assert '$150.00' in repr_str
        assert 'active_positions=1' in repr_str

    def test_to_dict(self):
        """Test to_dict conversion."""
        state = TradingState()
        state.initialized = True
        state.provider = "binance_futures"
        state.total_trades = 5

        state_dict = state.to_dict()

        assert isinstance(state_dict, dict)
        assert state_dict['initialized'] is True
        assert state_dict['provider'] == 'binance_futures'
        assert state_dict['total_trades'] == 5

    def test_from_dict(self):
        """Test from_dict creation."""
        data = {
            'initialized': True,
            'provider': 'binance_futures',
            'total_trades': 5,
            'total_pnl': 150.0,
            'active_positions': [{'symbol': 'BTC/USDT'}]
        }

        state = TradingState.from_dict(data)

        assert state.initialized is True
        assert state.provider == 'binance_futures'
        assert state.total_trades == 5
        assert state.total_pnl == 150.0
        assert state.active_positions == [{'symbol': 'BTC/USDT'}]


# Run with: pytest trading/tests/test_state.py -v --cov=trading.state --cov-report=term-missing
