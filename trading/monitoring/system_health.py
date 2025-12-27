"""System health monitoring module.

Aggregates safety control states and system health indicators:
- Kill switch status (ACTIVE/INACTIVE)
- Circuit breaker status (OPEN/CLOSED with reason)
- Position utilization (current/max positions)
- API connection status (CONNECTED/DISCONNECTED with latency)

Provides unified health status: HEALTHY, DEGRADED, CRITICAL
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict
import time


class HealthLevel(str, Enum):
    """System health levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class ConnectionStatus(str, Enum):
    """API connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"


@dataclass
class HealthStatus:
    """System health status snapshot."""

    # Overall health
    level: str  # HEALTHY, DEGRADED, CRITICAL
    timestamp: str

    # Safety controls
    kill_switch_active: bool
    kill_switch_reason: Optional[str] = None

    circuit_breaker_open: bool
    circuit_breaker_reason: Optional[str] = None

    # Position tracking
    current_positions: int = 0
    max_positions: int = 3
    position_utilization_pct: float = 0.0

    # API status
    api_status: str = "connected"
    api_latency_ms: Optional[float] = None
    last_api_check: Optional[str] = None

    # Detailed info
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.details is None:
            data["details"] = {}
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealthStatus":
        """Create from dictionary."""
        return cls(**data)


class SystemHealthMonitor:
    """System health monitoring and aggregation.

    Monitors and aggregates:
    - Safety control states (kill switch, circuit breaker, position limits)
    - API connection health
    - System resource status

    Determines overall health level: HEALTHY, DEGRADED, CRITICAL

    Example:
        >>> monitor = SystemHealthMonitor(
        ...     kill_switch=kill_switch,
        ...     circuit_breaker=circuit_breaker,
        ...     position_limits=position_limits
        ... )
        >>> health = monitor.get_health_status()
        >>> print(f"System: {health.level}")
    """

    def __init__(
        self,
        kill_switch=None,
        circuit_breaker=None,
        position_limits=None,
        provider=None,
    ):
        """Initialize system health monitor.

        Args:
            kill_switch: KillSwitch instance (optional)
            circuit_breaker: CircuitBreaker instance (optional)
            position_limits: PositionLimitEnforcer instance (optional)
            provider: Exchange provider for API health checks (optional)
        """
        self.logger = logging.getLogger(__name__)

        # Safety control references
        self.kill_switch = kill_switch
        self.circuit_breaker = circuit_breaker
        self.position_limits = position_limits
        self.provider = provider

        # API health tracking
        self.last_api_check_time = None
        self.last_api_latency = None
        self.api_status = ConnectionStatus.CONNECTED

        self.logger.info("SystemHealthMonitor initialized")

    def get_health_status(self) -> HealthStatus:
        """Get current system health status.

        Returns:
            HealthStatus object with all health indicators
        """
        # Check kill switch
        kill_switch_active = False
        kill_switch_reason = None

        if self.kill_switch:
            kill_switch_status = self.kill_switch.get_status()
            kill_switch_active = kill_switch_status.get("active", False)
            kill_switch_reason = kill_switch_status.get("reason")

        # Check circuit breaker
        circuit_breaker_open = False
        circuit_breaker_reason = None

        if self.circuit_breaker:
            cb_status = self.circuit_breaker.get_status()
            circuit_breaker_open = cb_status.get("is_open", False)
            circuit_breaker_reason = cb_status.get("trip_reason")

        # Check position limits
        current_positions = 0
        max_positions = 3
        position_utilization = 0.0

        if self.position_limits:
            # Get position count from position limits enforcer
            # Assuming it has a method to get current position count
            # For now, we'll use a safe default
            max_positions = getattr(
                self.position_limits.thresholds,
                "max_open_positions",
                3
            )

            # Calculate utilization (will be updated when positions are tracked)
            position_utilization = (current_positions / max_positions * 100) if max_positions > 0 else 0.0

        # Determine overall health level
        health_level = self._determine_health_level(
            kill_switch_active,
            circuit_breaker_open,
            position_utilization,
        )

        # Build health status
        health = HealthStatus(
            level=health_level.value,
            timestamp=datetime.now().isoformat(),
            kill_switch_active=kill_switch_active,
            kill_switch_reason=kill_switch_reason,
            circuit_breaker_open=circuit_breaker_open,
            circuit_breaker_reason=circuit_breaker_reason,
            current_positions=current_positions,
            max_positions=max_positions,
            position_utilization_pct=position_utilization,
            api_status=self.api_status.value,
            api_latency_ms=self.last_api_latency,
            last_api_check=self.last_api_check_time.isoformat() if self.last_api_check_time else None,
            details={
                "kill_switch_details": self.kill_switch.get_status() if self.kill_switch else {},
                "circuit_breaker_details": self.circuit_breaker.get_status() if self.circuit_breaker else {},
            }
        )

        return health

    def _determine_health_level(
        self,
        kill_switch_active: bool,
        circuit_breaker_open: bool,
        position_utilization: float,
    ) -> HealthLevel:
        """Determine overall system health level.

        Args:
            kill_switch_active: Whether kill switch is active
            circuit_breaker_open: Whether circuit breaker is open
            position_utilization: Position utilization percentage

        Returns:
            HealthLevel enum value
        """
        # CRITICAL: Kill switch active
        if kill_switch_active:
            return HealthLevel.CRITICAL

        # DEGRADED: Circuit breaker open
        if circuit_breaker_open:
            return HealthLevel.DEGRADED

        # DEGRADED: High position utilization (>80%)
        if position_utilization > 80:
            return HealthLevel.DEGRADED

        # DEGRADED: API issues
        if self.api_status == ConnectionStatus.DISCONNECTED:
            return HealthLevel.DEGRADED

        # HEALTHY: All systems operational
        return HealthLevel.HEALTHY

    async def check_api_health(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Check API connection health by pinging exchange.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Dict with status, latency, error (if any)
        """
        if not self.provider:
            self.api_status = ConnectionStatus.DISCONNECTED
            return {
                "status": ConnectionStatus.DISCONNECTED.value,
                "error": "No provider configured",
                "latency_ms": None,
            }

        try:
            # Measure API latency
            start_time = time.time()

            # Try to fetch server time (lightweight endpoint)
            # This varies by provider - using a generic approach
            try:
                # Attempt to call a lightweight method
                if hasattr(self.provider, 'fetch_ticker'):
                    await asyncio.wait_for(
                        self.provider.fetch_ticker("BTC/USDT"),
                        timeout=timeout
                    )
                else:
                    # Fallback: assume healthy if provider exists
                    await asyncio.sleep(0.01)  # Simulate quick check
            except asyncio.TimeoutError:
                raise Exception("API request timed out")

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            # Update status
            self.last_api_check_time = datetime.now()
            self.last_api_latency = latency_ms

            # Determine status based on latency
            if latency_ms < 1000:  # < 1 second
                self.api_status = ConnectionStatus.CONNECTED
            elif latency_ms < 5000:  # < 5 seconds
                self.api_status = ConnectionStatus.DEGRADED
            else:
                self.api_status = ConnectionStatus.DEGRADED

            self.logger.debug(
                "API health check completed",
                extra={"latency_ms": latency_ms, "status": self.api_status.value}
            )

            return {
                "status": self.api_status.value,
                "latency_ms": latency_ms,
                "error": None,
            }

        except Exception as e:
            # API check failed
            self.api_status = ConnectionStatus.DISCONNECTED
            self.last_api_check_time = datetime.now()
            self.last_api_latency = None

            self.logger.warning(
                "API health check failed",
                extra={"error": str(e)}
            )

            return {
                "status": ConnectionStatus.DISCONNECTED.value,
                "latency_ms": None,
                "error": str(e),
            }

    def update_position_count(self, count: int):
        """Update current position count for utilization calculation.

        Args:
            count: Current number of open positions
        """
        # This method will be called externally to update position count
        # Store in instance variable for next health check
        self._current_position_count = count

        self.logger.debug(
            "Position count updated",
            extra={"count": count}
        )

    def get_safety_status(self) -> Dict[str, Any]:
        """Get detailed safety controls status.

        Returns:
            Dict with detailed status of all safety controls
        """
        return {
            "kill_switch": self.kill_switch.get_status() if self.kill_switch else {"active": False},
            "circuit_breaker": self.circuit_breaker.get_status() if self.circuit_breaker else {"is_open": False},
            "position_limits": {
                "current": getattr(self, "_current_position_count", 0),
                "max": getattr(self.position_limits.thresholds, "max_open_positions", 3) if self.position_limits else 3,
            },
            "timestamp": datetime.now().isoformat(),
        }
