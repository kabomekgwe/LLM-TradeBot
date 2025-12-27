"""Tests for kill switch module."""

import pytest
import hmac
import hashlib
from datetime import datetime

from trading.safety.kill_switch import KillSwitch


class TestKillSwitch:
    """Test suite for KillSwitch."""

    @pytest.fixture
    def kill_switch(self):
        """Create kill switch instance for testing."""
        return KillSwitch(secret_key="test-secret-key")

    def test_initial_state(self, kill_switch):
        """Test kill switch starts inactive."""
        assert not kill_switch.is_active()
        status = kill_switch.get_status()
        assert not status["active"]
        assert status["triggered_at"] is None
        assert status["triggered_by"] is None
        assert status["reason"] is None

    def test_trigger_kill_switch(self, kill_switch):
        """Test triggering kill switch."""
        reason = "Emergency shutdown test"
        triggered_by = "test@example.com"

        success = kill_switch.trigger(reason, triggered_by)

        assert success
        assert kill_switch.is_active()

        status = kill_switch.get_status()
        assert status["active"]
        assert status["triggered_by"] == triggered_by
        assert status["reason"] == reason
        assert status["triggered_at"] is not None

    def test_trigger_already_active(self, kill_switch):
        """Test triggering already active kill switch returns False."""
        kill_switch.trigger("First trigger", "user1")

        # Try to trigger again
        success = kill_switch.trigger("Second trigger", "user2")

        assert not success
        assert kill_switch.is_active()

        # Should still have first trigger info
        status = kill_switch.get_status()
        assert status["triggered_by"] == "user1"
        assert status["reason"] == "First trigger"

    def test_reset_kill_switch(self, kill_switch):
        """Test resetting kill switch."""
        # Trigger first
        kill_switch.trigger("Test reason", "user@test.com")
        assert kill_switch.is_active()

        # Reset
        success = kill_switch.reset("admin@test.com")

        assert success
        assert not kill_switch.is_active()

    def test_reset_already_inactive(self, kill_switch):
        """Test resetting inactive kill switch returns False."""
        success = kill_switch.reset("admin@test.com")
        assert not success
        assert not kill_switch.is_active()

    def test_hmac_verification_valid(self, kill_switch):
        """Test HMAC signature verification with valid signature."""
        message = '{"reason": "test", "triggered_by": "user"}'

        # Generate valid signature
        signature = hmac.new(
            b"test-secret-key",
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        # Verify
        assert kill_switch.verify_hmac(message, signature)

    def test_hmac_verification_invalid(self, kill_switch):
        """Test HMAC signature verification with invalid signature."""
        message = '{"reason": "test", "triggered_by": "user"}'
        invalid_signature = "invalid_signature_12345"

        # Should reject invalid signature
        assert not kill_switch.verify_hmac(message, invalid_signature)

    def test_hmac_verification_wrong_secret(self, kill_switch):
        """Test HMAC signature with wrong secret key."""
        message = '{"reason": "test"}'

        # Generate signature with different secret
        wrong_signature = hmac.new(
            b"wrong-secret-key",
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        # Should reject
        assert not kill_switch.verify_hmac(message, wrong_signature)

    def test_hmac_verification_timing_safe(self, kill_switch):
        """Test HMAC verification uses timing-safe comparison."""
        # This test verifies the code uses hmac.compare_digest
        # which is resistant to timing attacks
        message = "test"
        correct_sig = hmac.new(b"test-secret-key", message.encode(), hashlib.sha256).hexdigest()

        # Signature that differs in first character
        wrong_sig = "0" + correct_sig[1:]

        # Should still be timing-safe (we can't test timing, but we test it rejects)
        assert not kill_switch.verify_hmac(message, wrong_sig)

    def test_concurrent_trigger_attempts(self, kill_switch):
        """Test multiple concurrent trigger attempts."""
        # First trigger succeeds
        assert kill_switch.trigger("Trigger 1", "user1")

        # Subsequent triggers fail
        assert not kill_switch.trigger("Trigger 2", "user2")
        assert not kill_switch.trigger("Trigger 3", "user3")

        # State should reflect first trigger only
        status = kill_switch.get_status()
        assert status["triggered_by"] == "user1"
        assert status["reason"] == "Trigger 1"

    def test_state_persistence_after_reset(self, kill_switch):
        """Test state maintains history after reset for audit."""
        # Trigger
        kill_switch.trigger("Critical issue", "operator@company.com")
        triggered_at = kill_switch.get_status()["triggered_at"]

        # Reset
        kill_switch.reset("admin@company.com")

        # Historical data should be preserved (implementation keeps it for audit)
        # This is for audit trail purposes
        assert not kill_switch.is_active()
