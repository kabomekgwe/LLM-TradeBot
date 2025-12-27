"""Kill Switch - Emergency shutdown mechanism with HMAC authentication.

Provides immediate manual override to stop ALL trading when critical
issues are detected by human operators.
"""

import hmac
import hashlib
import logging
from typing import Optional
from datetime import datetime


class KillSwitch:
    """Emergency kill switch for immediate trading halt.

    Provides secure webhook endpoint with HMAC-SHA256 authentication
    to prevent unauthorized emergency shutdowns.

    Features:
    - HMAC-SHA256 signature verification (timing-attack safe)
    - State tracking (active/inactive)
    - Audit trail (triggered_by, reason, timestamp)
    - Manual reset only (no automatic recovery)

    Example:
        >>> kill_switch = KillSwitch(secret_key="your-secret-key")
        >>> kill_switch.trigger("Runaway losses detected", "operator@company.com")
        >>> if kill_switch.is_active():
        ...     print("Trading halted!")
    """

    def __init__(self, secret_key: str):
        """Initialize kill switch.

        Args:
            secret_key: Secret key for HMAC signature verification
        """
        self.logger = logging.getLogger(__name__)

        self._active = False
        self._triggered_at: Optional[datetime] = None
        self._triggered_by: Optional[str] = None
        self._reason: Optional[str] = None
        self._secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key

        self.logger.info("Kill switch initialized")

    def trigger(self, reason: str, triggered_by: str) -> bool:
        """Activate kill switch - stops ALL trading immediately.

        Args:
            reason: Reason for emergency shutdown
            triggered_by: Identifier of person/system triggering (email, username, etc.)

        Returns:
            True if successfully triggered
        """
        if self._active:
            self.logger.warning(
                "kill_switch_already_active",
                extra={
                    "triggered_at": self._triggered_at.isoformat() if self._triggered_at else None,
                    "triggered_by": self._triggered_by,
                    "reason": self._reason,
                }
            )
            return False

        # Activate kill switch
        self._active = True
        self._triggered_at = datetime.now()
        self._triggered_by = triggered_by
        self._reason = reason

        self.logger.critical(
            "KILL_SWITCH_TRIGGERED",
            extra={
                "triggered_at": self._triggered_at.isoformat(),
                "triggered_by": triggered_by,
                "reason": reason,
            }
        )

        return True

    def is_active(self) -> bool:
        """Check if kill switch is currently active.

        Returns:
            True if kill switch is active (trading halted)
        """
        return self._active

    def reset(self, reset_by: str) -> bool:
        """Deactivate kill switch - requires manual confirmation.

        Args:
            reset_by: Identifier of person resetting the kill switch

        Returns:
            True if successfully reset
        """
        if not self._active:
            self.logger.warning("kill_switch_already_inactive")
            return False

        # Log reset event BEFORE deactivating
        self.logger.warning(
            "KILL_SWITCH_RESET",
            extra={
                "reset_at": datetime.now().isoformat(),
                "reset_by": reset_by,
                "was_triggered_at": self._triggered_at.isoformat() if self._triggered_at else None,
                "was_triggered_by": self._triggered_by,
                "reason": self._reason,
            }
        )

        # Reset state
        self._active = False
        # Keep historical data for audit trail
        # self._triggered_at, self._triggered_by, self._reason remain for logging

        return True

    def verify_hmac(self, message: str, signature: str) -> bool:
        """Verify HMAC-SHA256 signature for webhook security.

        Uses timing-attack-safe comparison to prevent signature guessing.

        Args:
            message: Message that was signed (typically JSON body)
            signature: HMAC signature to verify (hex string)

        Returns:
            True if signature is valid

        Example:
            >>> message = '{"reason": "test", "triggered_by": "admin"}'
            >>> signature = hmac.new(secret_key, message.encode(), hashlib.sha256).hexdigest()
            >>> kill_switch.verify_hmac(message, signature)
            True
        """
        try:
            # Calculate expected signature
            message_bytes = message.encode() if isinstance(message, str) else message
            expected_signature = hmac.new(
                self._secret_key,
                message_bytes,
                hashlib.sha256
            ).hexdigest()

            # Timing-attack-safe comparison
            return hmac.compare_digest(expected_signature, signature)

        except Exception as e:
            self.logger.error(
                "hmac_verification_failed",
                extra={"error": str(e)}
            )
            return False

    def get_status(self) -> dict:
        """Get current kill switch status.

        Returns:
            Dictionary with status information
        """
        return {
            "active": self._active,
            "triggered_at": self._triggered_at.isoformat() if self._triggered_at else None,
            "triggered_by": self._triggered_by,
            "reason": self._reason,
        }
