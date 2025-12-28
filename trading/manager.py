"""Trading manager - Main orchestrator for the 8-agent trading system.

This module provides the TradingManager class that coordinates all agents
and manages the trading lifecycle.
"""

import os
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime

from .config import TradingConfig
from .state import TradingState
from .providers.factory import create_provider
from .providers.base import BaseExchangeProvider
from .logging_config import get_logger, DecisionContext
from .exceptions import (
    AgentError,
    APIError,
    RiskViolationError,
    StateError,
    ConfigurationError,
    TradingBotError,
)

# Import safety controls
from .safety.thresholds import SafetyThresholds
from .safety.kill_switch import KillSwitch
from .safety.circuit_breaker import CircuitBreaker
from .safety.position_limits import PositionLimitEnforcer
from .analytics.risk_calculator import RiskCalculator
from .memory.trade_history import TradeJournal

# Import monitoring infrastructure (Phase 10)
from .monitoring.metrics_tracker import MetricsTracker
from .monitoring.system_health import SystemHealthMonitor
from .monitoring.alert_manager import AlertManager

# Import notifications
from .notifications.manager import NotificationManager

# Import graceful shutdown handler (Phase 11)
from .utils.shutdown import GracefulShutdownHandler

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
        self.logger = get_logger(__name__)

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
                "provider_initialized",
                extra={"provider": self.provider.get_provider_name()}
            )
        except (ConfigurationError, ValueError) as e:
            # Configuration errors - fatal, can't proceed
            self.logger.error(
                "provider_initialization_failed",
                extra={"error": str(e), "provider": self.config.provider},
            )
            self.enabled = False
            return
        except Exception as e:
            # Unexpected errors - log with traceback
            self.logger.critical(
                "provider_initialization_unexpected_error",
                extra={"error": str(e), "error_type": type(e).__name__},
                exc_info=True,
            )
            self.enabled = False
            return

        # Initialize safety components
        from .config.secrets import SecretsManager
        self.thresholds = SafetyThresholds()
        kill_switch_secret = SecretsManager.get_kill_switch_secret() or "default-secret-change-me"
        self.kill_switch = KillSwitch(secret_key=kill_switch_secret)
        self.risk_calculator = RiskCalculator()
        self.trade_journal = TradeJournal(spec_dir / ".trade_journal.db")
        self.circuit_breaker = CircuitBreaker(self.thresholds, self.risk_calculator)
        self.position_limits = PositionLimitEnforcer(self.thresholds)

        # Initialize notification manager
        self.notification_manager = NotificationManager(self.config)

        # Initialize monitoring infrastructure (Phase 10)
        self.metrics_tracker = MetricsTracker(initial_equity=10000.0)
        self.health_monitor = SystemHealthMonitor(
            kill_switch=self.kill_switch,
            circuit_breaker=self.circuit_breaker,
            position_limits=self.position_limits,
            provider=self.provider,
        )
        self.alert_manager = AlertManager(
            notification_manager=self.notification_manager,
            config=self.config,
        )

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

        # Initialize graceful shutdown handler (Phase 11 Task 2)
        self.shutdown_handler = GracefulShutdownHandler(
            manager=self,
            timeout=30
        )
        self.shutdown_handler.register()
        self.logger.info("graceful_shutdown_handler_initialized")

        # Initialize state if first run
        if not self.state.initialized:
            self.state.initialized = True
            self.state.provider = self.config.provider
            from datetime import datetime
            self.state.created_at = datetime.now().isoformat()
            self.state.save(spec_dir)

        self.logger.info(
            "safety_systems_initialized",
            extra={
                "kill_switch": "active",
                "circuit_breaker": "armed",
                "position_limits": "enforced",
            }
        )

    async def run_trading_loop(self, symbol: str = "BTC/USDT") -> dict:
        """Execute the complete 8-agent trading loop with safety controls.

        This is the heart of the trading system. Runs all 8 agents in sequence
        with safety checkpoints at each critical stage.

        Safety checkpoints (in order):
        1. Kill switch check (highest priority - immediate halt)
        2. Circuit breaker check (auto-pause on threshold breach)
        3. Position limit check (before order placement)

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
            RuntimeError: If trading is not enabled or safety controls block trading
        """
        if not self.enabled:
            raise RuntimeError("Trading is not enabled - check configuration")

        # SAFETY CHECKPOINT 1: Kill switch check (highest priority)
        if self.kill_switch.is_active():
            self.logger.critical(
                "kill_switch_blocked_trading",
                extra={
                    "kill_switch_status": self.kill_switch.get_status(),
                }
            )

            # MONITORING: Check alerts for kill switch activation (Phase 10 Task 3)
            await self.alert_manager.check_and_send_alerts(
                metrics=self.metrics_tracker.get_current_metrics().to_dict(),
                health=self.health_monitor.get_health_status().to_dict(),
            )

            raise RuntimeError(
                "KILL SWITCH ACTIVE - ALL TRADING STOPPED. "
                f"Reason: {self.kill_switch.get_status()['reason']}"
            )

        # SAFETY CHECKPOINT 2: Circuit breaker check
        if self.circuit_breaker.is_open():
            self.logger.warning(
                "circuit_breaker_blocked_trading",
                extra={
                    "circuit_breaker_status": self.circuit_breaker.get_status(),
                }
            )

            # MONITORING: Check alerts for circuit breaker trip (Phase 10 Task 3)
            await self.alert_manager.check_and_send_alerts(
                metrics=self.metrics_tracker.get_current_metrics().to_dict(),
                health=self.health_monitor.get_health_status().to_dict(),
            )

            raise RuntimeError(
                f"Circuit breaker open - trading paused. "
                f"Reason: {self.circuit_breaker.get_status()['trip_reason']}"
            )

        # Legacy state check (backwards compatibility)
        if self.state.circuit_breaker_tripped:
            raise RuntimeError(
                f"Circuit breaker tripped: {self.state.last_circuit_trip_reason}. "
                "Reset manually to resume trading."
            )

        # Generate decision ID for tracing
        decision_id = str(uuid.uuid4())
        DecisionContext.set_decision_id(decision_id)

        self.logger.info(
            "decision_start",
            extra={
                **DecisionContext.get_extra(),
                "symbol": symbol,
            }
        )

        # Initialize context with symbol and state
        context = {
            "symbol": symbol,
            "state": self.state,
        }

        try:
            # Step 1: Fetch market data
            self.logger.info(
                "agent_start",
                extra={**DecisionContext.get_extra(), "agent": "DataSyncAgent", "step": "1/8"}
            )
            context.update(await self.agents["data_sync"].execute(context))

            # Step 2: Technical analysis
            self.logger.info(
                "agent_start",
                extra={**DecisionContext.get_extra(), "agent": "QuantAnalystAgent", "step": "2/8"}
            )
            context.update(await self.agents["quant_analyst"].execute(context))

            # Step 3: ML prediction
            self.logger.info(
                "agent_start",
                extra={**DecisionContext.get_extra(), "agent": "PredictAgent", "step": "3/8"}
            )
            context.update(await self.agents["predict"].execute(context))

            # Step 4: Adversarial analysis (Bull + Bear in sequence)
            self.logger.info(
                "agent_start",
                extra={**DecisionContext.get_extra(), "agent": "BullAgent", "step": "4/8"}
            )
            bull_result = await self.agents["bull"].execute(context)
            context.update(bull_result)

            self.logger.info(
                "agent_start",
                extra={**DecisionContext.get_extra(), "agent": "BearAgent", "step": "5/8"}
            )
            bear_result = await self.agents["bear"].execute(context)
            context.update(bear_result)

            # Step 5: Decision aggregation
            self.logger.info(
                "agent_start",
                extra={**DecisionContext.get_extra(), "agent": "DecisionCoreAgent", "step": "6/8"}
            )
            context.update(await self.agents["decision_core"].execute(context))

            # Step 6: Risk veto
            self.logger.info(
                "agent_start",
                extra={**DecisionContext.get_extra(), "agent": "RiskAuditAgent", "step": "7/8"}
            )
            context.update(await self.agents["risk_audit"].execute(context))

            # Check if vetoed
            risk_audit = context.get("risk_audit", {})
            if risk_audit.get("veto", False):
                veto_reason = risk_audit.get("reason", "Unknown reason")
                self.logger.warning(
                    "decision_vetoed",
                    extra={
                        **DecisionContext.get_extra(),
                        "veto_reason": veto_reason,
                    }
                )

                # Clear decision context
                DecisionContext.clear()

                return {
                    "success": False,
                    "action": "hold",
                    "veto": True,
                    "veto_reason": veto_reason,
                }

            # Step 7: Execution
            self.logger.info(
                "agent_start",
                extra={**DecisionContext.get_extra(), "agent": "ExecutionEngine", "step": "8/8"}
            )
            context.update(await self.agents["execution"].execute(context))

            # Extract execution result
            execution = context.get("execution", {})

            if execution.get("success"):
                # Update state
                self.state.total_trades += 1
                self.state.save(self.spec_dir)

                decision = context.get("decision", {})
                self.logger.info(
                    "decision_executed",
                    extra={
                        **DecisionContext.get_extra(),
                        "action": decision.get("action"),
                        "price": execution.get("price"),
                        "amount": execution.get("amount"),
                        "confidence": decision.get("confidence"),
                    }
                )

                # SAFETY CHECKPOINT 4: Update circuit breaker after successful trade
                # Get trade history from journal and check thresholds
                trades = self.trade_journal.get_all_trades()
                balance = await self.get_balance()
                current_equity = balance.get("total", 10000.0) if balance else 10000.0

                if self.circuit_breaker.check_and_update(trades, current_equity):
                    self.logger.critical(
                        "circuit_breaker_tripped_after_trade",
                        extra={
                            "circuit_status": self.circuit_breaker.get_status(),
                        }
                    )
                    # Circuit breaker has tripped - trading will be blocked next iteration

                # MONITORING: Update real-time metrics (Phase 10 Task 1)
                # Create trade data dict for metrics tracker
                trade_data = {
                    "realized_pnl": execution.get("pnl", 0.0),
                    "pnl_pct": execution.get("pnl_pct", 0.0),
                    "won": execution.get("pnl", 0.0) > 0,
                    "closed": True,
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "symbol": symbol,
                    "side": decision.get("action"),
                    "entry_price": execution.get("price", 0.0),
                    "amount": execution.get("amount", 0.0),
                }

                # Update metrics tracker
                updated_metrics = self.metrics_tracker.update_trade(trade_data)

                # TODO: Broadcast metrics update via dashboard server (if available)
                # This will be connected when dashboard server integration is added
                self.logger.debug(
                    "metrics_updated",
                    extra={
                        "sharpe": updated_metrics.sharpe_ratio,
                        "drawdown": updated_metrics.current_drawdown,
                        "win_rate": updated_metrics.win_rate,
                    }
                )

                # MONITORING: Check alert triggers after trade execution (Phase 10 Task 3)
                await self.alert_manager.check_and_send_alerts(
                    metrics=updated_metrics.to_dict(),
                    health=self.health_monitor.get_health_status().to_dict(),
                )

                # Clear decision context
                DecisionContext.clear()

                return {
                    "success": True,
                    "action": decision.get("action"),
                    "order": execution.get("order"),
                    "price": execution.get("price"),
                    "amount": execution.get("amount"),
                    "confidence": decision.get("confidence"),
                }
            else:
                self.logger.warning(
                    "execution_failed",
                    extra={
                        **DecisionContext.get_extra(),
                        "error": execution.get("error"),
                    }
                )

                # Clear decision context
                DecisionContext.clear()

                return {
                    "success": False,
                    "action": "hold",
                    "error": execution.get("error"),
                }

        except AgentError as e:
            # Agent execution failures - log and skip this iteration
            self.logger.error(
                "agent_execution_failed",
                extra={
                    **DecisionContext.get_extra(),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            DecisionContext.clear()
            return {
                "success": False,
                "action": "hold",
                "error": f"Agent error: {e}",
            }
        except APIError as e:
            # API errors - log and record in circuit breaker
            self.circuit_breaker.record_api_error()
            self.logger.warning(
                "api_error_in_trading_loop",
                extra={
                    **DecisionContext.get_extra(),
                    "error": str(e),
                    "circuit_status": self.circuit_breaker.get_status(),
                },
            )
            DecisionContext.clear()
            return {
                "success": False,
                "action": "hold",
                "error": f"API error: {e}",
            }
        except RiskViolationError as e:
            # Risk veto - log and continue (trade rejected)
            self.logger.info(
                "risk_veto_in_trading_loop",
                extra={
                    **DecisionContext.get_extra(),
                    "reason": str(e),
                },
            )
            DecisionContext.clear()
            return {
                "success": False,
                "action": "hold",
                "veto": True,
                "veto_reason": str(e),
            }
        except StateError as e:
            # State errors - critical, might need to stop
            self.logger.critical(
                "state_error_in_trading_loop",
                extra={
                    **DecisionContext.get_extra(),
                    "error": str(e),
                },
                exc_info=True,
            )
            # Try to save current state before continuing
            try:
                self.state.save(self.spec_dir)
            except Exception:
                pass  # Best effort
            DecisionContext.clear()
            return {
                "success": False,
                "action": "hold",
                "error": f"State error: {e}",
            }
        except TradingBotError as e:
            # Known errors - log and continue
            self.logger.error(
                "trading_error",
                extra={
                    **DecisionContext.get_extra(),
                    "error_type": type(e).__name__,
                    "message": str(e),
                },
            )
            DecisionContext.clear()
            return {
                "success": False,
                "action": "hold",
                "error": str(e),
            }
        except Exception as e:
            # Unexpected errors - log with traceback, continue with caution
            self.logger.critical(
                "unexpected_trading_error",
                extra={
                    **DecisionContext.get_extra(),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            DecisionContext.clear()
            return {
                "success": False,
                "action": "hold",
                "error": f"Unexpected error: {e}",
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
        except APIError as e:
            self.logger.error(
                "fetch_positions_failed",
                extra={"error": str(e)},
            )
            return []
        except Exception as e:
            self.logger.critical(
                "fetch_positions_unexpected_error",
                extra={"error": str(e), "error_type": type(e).__name__},
                exc_info=True,
            )
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
        except APIError as e:
            self.logger.error(
                "fetch_balance_failed",
                extra={"error": str(e)},
            )
            return None
        except Exception as e:
            self.logger.critical(
                "fetch_balance_unexpected_error",
                extra={"error": str(e), "error_type": type(e).__name__},
                exc_info=True,
            )
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
