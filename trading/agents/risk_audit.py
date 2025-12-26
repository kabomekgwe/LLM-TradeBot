"""RiskAuditAgent - Safety layer with veto power.

This is Agent #7 in the 8-agent system.
Independent safety validation with authority to veto risky decisions.
"""

from typing import Any

from ..state import TradingState
from .base_agent import BaseAgent


class RiskAuditAgent(BaseAgent):
    """Risk audit agent with veto power.

    Final safety check before order execution. Can veto trades that
    violate risk limits or safety constraints.
    """

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Audit trading decision for risk compliance.

        Args:
            context: Must contain:
                - "decision": From DecisionCoreAgent
                - "state": Current TradingState

        Returns:
            Context updated with "risk_audit" containing:
                - "veto": Boolean (True if trade should be blocked)
                - "reason": String explanation if vetoed

        Example:
            >>> result = await agent.execute(context)
            >>> result["risk_audit"]["veto"]
            False
        """
        decision = context.get("decision", {})
        state = context.get("state")

        if not decision:
            raise ValueError("decision is required in context")

        if not isinstance(state, TradingState):
            raise ValueError("state must be TradingState instance")

        self.log_decision("Auditing decision for risk compliance")

        # Check veto conditions
        should_veto, veto_reason = await self._check_veto_conditions(decision, state)

        if should_veto:
            self.log_decision(f"VETO: {veto_reason}", level="warning")
        else:
            self.log_decision("Risk audit passed")

        return {
            "risk_audit": {
                "veto": should_veto,
                "reason": veto_reason if should_veto else None,
            }
        }

    async def _check_veto_conditions(
        self, decision: dict, state: TradingState
    ) -> tuple[bool, str]:
        """Check all veto conditions.

        Returns:
            (should_veto, reason) tuple
        """
        # Veto if circuit breaker is tripped
        if state.circuit_breaker_tripped:
            return True, f"Circuit breaker tripped: {state.last_circuit_trip_reason}"

        # Veto if max positions reached
        if len(state.active_positions) >= self.config.max_open_positions:
            return True, f"Max open positions reached ({self.config.max_open_positions})"

        # Veto if daily drawdown exceeded
        if state.daily_drawdown_pct >= self.config.max_daily_drawdown_pct:
            return True, f"Daily drawdown limit exceeded ({state.daily_drawdown_pct:.2f}%)"

        # Veto if confidence too low
        confidence = decision.get("confidence", 0.0)
        if confidence < self.config.decision_threshold:
            return True, f"Confidence below threshold ({confidence:.2f} < {self.config.decision_threshold})"

        # Veto if action is hold (no trade needed)
        if decision.get("action") == "hold":
            return True, "Decision is to hold (no trade)"

        # All checks passed
        return False, ""
