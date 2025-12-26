"""Trading manager - Main orchestrator for the 8-agent trading system.

This module provides the TradingManager class that coordinates all agents
and manages the trading lifecycle.
"""

import logging
from pathlib import Path
from typing import Optional

from .config import TradingConfig
from .state import TradingState
from .providers.factory import create_provider
from .providers.base import BaseExchangeProvider

# Import all 8 agents
from .agents.data_sync import DataSyncAgent
from .agents.quant_analyst import QuantAnalystAgent
from .agents.predict import PredictAgent
from .agents.bull import BullAgent
from .agents.bear import BearAgent
from .agents.decision_core import DecisionCoreAgent
from .agents.risk_audit import RiskAuditAgent
from .agents.execution import ExecutionEngine


class TradingManager:
    """Main orchestrator for trading operations.

    Coordinates the 8-agent adversarial decision system following
    the pattern established by LinearManager in the Linear integration.

    The trading loop follows this sequence:
    1. DataSync → Fetch market data
    2. QuantAnalyst → Generate technical signals
    3. PredictAgent → ML price forecasting
    4. Bull + Bear → Adversarial analysis (parallel)
    5. DecisionCore → Aggregate votes with regime weighting
    6. RiskAudit → Safety validation with veto power
    7. ExecutionEngine → Place orders
    8. State persistence

    Example:
        >>> manager = TradingManager(Path("specs/001/"), provider="binance_futures")
        >>> if manager.enabled:
        ...     result = await manager.run_trading_loop("BTC/USDT")
    """

    def __init__(self, spec_dir: Path, provider: Optional[str] = None):
        """Initialize trading manager.

        Args:
            spec_dir: Path to spec directory for state isolation
            provider: Provider name (if None, reads from TRADING_PROVIDER env var)

        Raises:
            ValueError: If configuration is invalid
        """
        self.spec_dir = spec_dir
        self.logger = logging.getLogger(__name__)

        # Load configuration from environment
        self.config = TradingConfig.from_env(provider)

        # Load or create state
        self.state = TradingState.load(spec_dir) or TradingState()

        # Check if trading is enabled
        if not self.config.is_valid():
            self.logger.warning(
                f"Trading disabled: Invalid configuration for provider {self.config.provider}"
            )
            self.enabled = False
            return

        # Validate risk parameters
        is_valid, error_msg = self.config.validate_risk_parameters()
        if not is_valid:
            self.logger.error(f"Invalid risk parameters: {error_msg}")
            self.enabled = False
            return

        self.enabled = True

        # Initialize provider
        try:
            self.provider: BaseExchangeProvider = create_provider(self.config)
            self.logger.info(
                f"Trading manager initialized with {self.provider.get_provider_name()}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize provider: {e}")
            self.enabled = False
            return

        # Initialize all 8 agents
        self.agents = {
            "data_sync": DataSyncAgent(self.provider, self.config),
            "quant_analyst": QuantAnalystAgent(self.provider, self.config),
            "predict": PredictAgent(self.provider, self.config),
            "bull": BullAgent(self.provider, self.config),
            "bear": BearAgent(self.provider, self.config),
            "decision_core": DecisionCoreAgent(self.provider, self.config),
            "risk_audit": RiskAuditAgent(self.provider, self.config),
            "execution": ExecutionEngine(self.provider, self.config),
        }

        # Initialize state if first run
        if not self.state.initialized:
            self.state.initialized = True
            self.state.provider = self.config.provider
            from datetime import datetime
            self.state.created_at = datetime.now().isoformat()
            self.state.save(spec_dir)

    async def run_trading_loop(self, symbol: str = "BTC/USDT") -> dict:
        """Execute the complete 8-agent trading loop.

        This is the heart of the trading system. Runs all 8 agents in sequence
        and returns the final execution result.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")

        Returns:
            Dict with execution result containing:
                - "success": Boolean
                - "action": "buy", "sell", or "hold"
                - "order": Order object if executed
                - "veto_reason": String if vetoed

        Example:
            >>> result = await manager.run_trading_loop("BTC/USDT")
            >>> if result["success"]:
            ...     print(f"Executed: {result['action']}")

        Raises:
            RuntimeError: If trading is not enabled or circuit breaker is tripped
        """
        if not self.enabled:
            raise RuntimeError("Trading is not enabled - check configuration")

        if self.state.circuit_breaker_tripped:
            raise RuntimeError(
                f"Circuit breaker tripped: {self.state.last_circuit_trip_reason}. "
                "Reset manually to resume trading."
            )

        self.logger.info(f"=== Starting trading loop for {symbol} ===")

        # Initialize context with symbol and state
        context = {
            "symbol": symbol,
            "state": self.state,
        }

        try:
            # Step 1: Fetch market data
            self.logger.info("[1/8] DataSyncAgent - Fetching market data")
            context.update(await self.agents["data_sync"].execute(context))

            # Step 2: Technical analysis
            self.logger.info("[2/8] QuantAnalystAgent - Generating technical signals")
            context.update(await self.agents["quant_analyst"].execute(context))

            # Step 3: ML prediction
            self.logger.info("[3/8] PredictAgent - Running ML forecast")
            context.update(await self.agents["predict"].execute(context))

            # Step 4: Adversarial analysis (Bull + Bear in sequence)
            self.logger.info("[4/8] BullAgent - Analyzing bullish signals")
            bull_result = await self.agents["bull"].execute(context)
            context.update(bull_result)

            self.logger.info("[5/8] BearAgent - Analyzing bearish signals")
            bear_result = await self.agents["bear"].execute(context)
            context.update(bear_result)

            # Step 5: Decision aggregation
            self.logger.info("[6/8] DecisionCoreAgent - Aggregating votes")
            context.update(await self.agents["decision_core"].execute(context))

            # Step 6: Risk veto
            self.logger.info("[7/8] RiskAuditAgent - Safety validation")
            context.update(await self.agents["risk_audit"].execute(context))

            # Check if vetoed
            risk_audit = context.get("risk_audit", {})
            if risk_audit.get("veto", False):
                veto_reason = risk_audit.get("reason", "Unknown reason")
                self.logger.warning(f"Trade vetoed by RiskAuditAgent: {veto_reason}")

                return {
                    "success": False,
                    "action": "hold",
                    "veto": True,
                    "veto_reason": veto_reason,
                }

            # Step 7: Execution
            self.logger.info("[8/8] ExecutionEngine - Placing orders")
            context.update(await self.agents["execution"].execute(context))

            # Extract execution result
            execution = context.get("execution", {})

            if execution.get("success"):
                # Update state
                self.state.total_trades += 1
                self.state.save(self.spec_dir)

                decision = context.get("decision", {})
                self.logger.info(
                    f"=== Trading loop completed successfully: "
                    f"{decision.get('action')} @ ${execution.get('price', 0):.2f} ==="
                )

                return {
                    "success": True,
                    "action": decision.get("action"),
                    "order": execution.get("order"),
                    "price": execution.get("price"),
                    "amount": execution.get("amount"),
                    "confidence": decision.get("confidence"),
                }
            else:
                self.logger.warning(f"Execution failed: {execution.get('error')}")

                return {
                    "success": False,
                    "action": "hold",
                    "error": execution.get("error"),
                }

        except Exception as e:
            self.logger.error(f"Trading loop error: {e}", exc_info=True)

            return {
                "success": False,
                "action": "hold",
                "error": str(e),
            }

    async def get_positions(self) -> list[dict]:
        """Get current open positions.

        Returns:
            List of position dicts from provider

        Example:
            >>> positions = await manager.get_positions()
            >>> len(positions)
            2
        """
        if not self.enabled:
            return []

        try:
            positions = await self.provider.fetch_positions()
            return [
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                for pos in positions
            ]
        except Exception as e:
            self.logger.error(f"Failed to fetch positions: {e}")
            return []

    async def get_balance(self) -> Optional[dict]:
        """Get account balance.

        Returns:
            Balance dict with free, used, total

        Example:
            >>> balance = await manager.get_balance()
            >>> balance["total"]
            10000.0
        """
        if not self.enabled:
            return None

        try:
            balance = await self.provider.fetch_balance()
            return {
                "currency": balance.currency,
                "free": balance.free,
                "used": balance.used,
                "total": balance.total,
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch balance: {e}")
            return None

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (manual action required).

        Example:
            >>> manager.state.circuit_breaker_tripped
            True
            >>> manager.reset_circuit_breaker()
            >>> manager.state.circuit_breaker_tripped
            False
        """
        self.state.reset_circuit_breaker()
        self.state.save(self.spec_dir)
        self.logger.info("Circuit breaker reset")

    async def close(self):
        """Close provider connection and cleanup resources.

        Call this when done with the manager.

        Example:
            >>> manager = TradingManager(spec_dir)
            >>> # ... use manager ...
            >>> await manager.close()
        """
        if hasattr(self.provider, 'close'):
            await self.provider.close()
        self.logger.info("Trading manager closed")
