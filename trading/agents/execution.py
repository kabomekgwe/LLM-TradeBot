"""ExecutionEngine - Order execution and position management.

This is Agent #8 in the 8-agent system.
Handles precise order execution and lifecycle tracking with advanced order types.
"""

import asyncio
from typing import Any, Optional

from ..state import TradingState
from .base_agent import BaseAgent
from ..orders import OrderManager, BracketOrder


class ExecutionEngine(BaseAgent):
    """Execution engine - handles order placement and tracking.

    Final agent in the pipeline. Executes approved trading decisions
    with advanced order types (bracket orders, stop-loss, take-profit).
    """

    def __init__(self, *args, **kwargs):
        """Initialize execution engine with OrderManager."""
        super().__init__(*args, **kwargs)
        self.order_manager: Optional[OrderManager] = None

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute trading decision.

        Args:
            context: Must contain:
                - "decision": From DecisionCoreAgent
                - "state": Current TradingState
                - "symbol": Trading symbol

        Returns:
            Context updated with "execution" containing:
                - "order": Order object if executed
                - "success": Boolean
                - "error": Error message if failed

        Example:
            >>> result = await agent.execute(context)
            >>> result["execution"]["success"]
            True
        """
        decision = context.get("decision", {})
        state = context.get("state")
        symbol = context.get("symbol")

        if not decision or not symbol:
            raise ValueError("decision and symbol are required in context")

        if not isinstance(state, TradingState):
            raise ValueError("state must be TradingState instance")

        action = decision.get("action")
        if action not in ("buy", "sell"):
            # No action to take (hold)
            return {"execution": {"success": False, "error": "No action to execute"}}

        self.log_decision(f"Executing {action} order for {symbol}")

        try:
            # Initialize OrderManager if not already done
            if self.order_manager is None:
                self.order_manager = OrderManager(self.provider, self.config)

            # Get current price
            ticker = context.get("market_data", {}).get("ticker")
            if not ticker:
                raise ValueError("Ticker data not available")

            current_price = ticker.last

            # Calculate position size based on risk
            # Use 2% default stop-loss distance
            stop_loss_pct = 0.02  # 2% stop
            take_profit_pct = 0.04  # 4% take profit (2:1 reward:risk)

            if action == "buy":
                stop_loss_price = current_price * (1 - stop_loss_pct)
                take_profit_price = current_price * (1 + take_profit_pct)
            else:  # sell
                stop_loss_price = current_price * (1 + stop_loss_pct)
                take_profit_price = current_price * (1 - take_profit_pct)

            # Calculate position size based on max risk
            max_risk_usd = min(
                self.config.max_position_size_usd * 0.02,  # 2% of max position
                100  # Max $100 risk per trade
            )

            amount = self.order_manager.calculate_position_size(
                entry_price=current_price,
                stop_loss_price=stop_loss_price,
                risk_amount_usd=max_risk_usd,
            )

            # Apply confidence scaling
            confidence = decision.get("confidence", 0.5)
            amount = amount * confidence

            # Create bracket order (entry + stop loss + take profit) with timeout
            # 15s timeout for order placement (no retry to avoid duplicate orders)
            entry_order, stop_order, tp_order = await asyncio.wait_for(
                self.order_manager.create_bracket_order(
                    symbol=symbol,
                    side=action,
                    amount=amount,
                    entry_price=current_price,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    order_type="market",  # Market order for immediate execution
                ),
                timeout=15.0
            )

            self.log_decision(
                f"Bracket order executed:\n"
                f"  Entry: {entry_order.id} - {action} {amount:.4f} @ ${current_price:.2f}\n"
                f"  Stop Loss: {stop_order.id} @ ${stop_loss_price:.2f}\n"
                f"  Take Profit: {tp_order.id} @ ${take_profit_price:.2f}\n"
                f"  Risk: ${max_risk_usd:.2f}, Reward: ${max_risk_usd * 2:.2f}"
            )

            # Update state with new position
            state.add_position({
                "symbol": symbol,
                "side": "long" if action == "buy" else "short",
                "size": amount,
                "entry_price": current_price,
                "order_id": entry_order.id,
                "stop_loss_order_id": stop_order.id,
                "take_profit_order_id": tp_order.id,
                "risk_usd": max_risk_usd,
            })

            return {
                "execution": {
                    "success": True,
                    "order": entry_order,
                    "stop_loss_order": stop_order,
                    "take_profit_order": tp_order,
                    "amount": amount,
                    "price": current_price,
                    "stop_loss_price": stop_loss_price,
                    "take_profit_price": take_profit_price,
                    "risk_usd": max_risk_usd,
                    "reward_usd": max_risk_usd * 2,
                }
            }

        except asyncio.TimeoutError:
            self.log_decision(
                "order_placement_timeout",
                level="error",
                symbol=symbol,
                action=action,
                timeout_seconds=15,
            )
            return {
                "execution": {
                    "success": False,
                    "error": "Order placement exceeded 15s timeout",
                }
            }
        except Exception as e:
            self.log_decision(
                "execution_failed",
                level="error",
                symbol=symbol,
                action=action,
                error=str(e),
                error_type=type(e).__name__,
            )
            return {
                "execution": {
                    "success": False,
                    "error": str(e),
                }
            }
