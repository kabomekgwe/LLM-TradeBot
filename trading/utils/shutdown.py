"""Graceful Shutdown Handler - Clean shutdown for Docker containers.

This module provides graceful shutdown functionality for the trading bot,
ensuring all positions are closed safely when Docker sends SIGTERM.

Docker stop sends SIGTERM signal followed by SIGKILL after grace period.
This handler intercepts SIGTERM and performs clean shutdown operations.
"""

import signal
import sys
import asyncio
import logging
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..manager import TradingManager
    from ..state import TradingState
    from ..providers.base import BaseExchangeProvider
    from ..monitoring.alert_manager import AlertManager


class GracefulShutdownHandler:
    """Handles graceful shutdown of trading bot on SIGTERM/SIGINT.

    Registers signal handlers for SIGTERM (Docker stop) and SIGINT (Ctrl+C).
    On signal received:
    1. Cancel all pending orders
    2. Close all open positions (market orders)
    3. Save current state to disk
    4. Send shutdown notification
    5. Exit gracefully

    Example:
        >>> handler = GracefulShutdownHandler(
        ...     manager=trading_manager,
        ...     timeout=30
        ... )
        >>> handler.register()
        >>> # Handler is now active, will trigger on SIGTERM/SIGINT
    """

    def __init__(
        self,
        manager: Optional["TradingManager"] = None,
        state: Optional["TradingState"] = None,
        provider: Optional["BaseExchangeProvider"] = None,
        alert_manager: Optional["AlertManager"] = None,
        timeout: int = 30,
    ):
        """Initialize graceful shutdown handler.

        Args:
            manager: TradingManager instance (provides access to all components)
            state: TradingState instance for saving state
            provider: Exchange provider for canceling orders/closing positions
            alert_manager: AlertManager for sending shutdown notifications
            timeout: Maximum seconds to wait for clean shutdown (default: 30)
        """
        self.manager = manager
        self.state = state
        self.provider = provider
        self.alert_manager = alert_manager
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        # Shutdown state
        self._shutdown_initiated = False
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Extract components from manager if provided
        if self.manager:
            self.state = self.state or getattr(manager, 'state', None)
            self.provider = self.provider or getattr(manager, 'provider', None)
            self.alert_manager = self.alert_manager or getattr(manager, 'alert_manager', None)

    def register(self):
        """Register signal handlers for SIGTERM and SIGINT.

        Call this during TradingManager initialization.
        """
        # Register SIGTERM (Docker stop)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Register SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self._signal_handler)

        self.logger.info(
            "graceful_shutdown_handler_registered",
            extra={
                "signals": ["SIGTERM", "SIGINT"],
                "timeout": self.timeout,
            }
        )

    def _signal_handler(self, signum: int, frame):
        """Handle shutdown signal.

        Args:
            signum: Signal number (SIGTERM=15, SIGINT=2)
            frame: Current stack frame
        """
        signal_name = signal.Signals(signum).name
        self.logger.critical(
            "shutdown_signal_received",
            extra={
                "signal": signal_name,
                "signal_number": signum,
                "timestamp": datetime.now().isoformat(),
            }
        )

        if self._shutdown_initiated:
            self.logger.warning("Shutdown already in progress, forcing exit...")
            sys.exit(1)

        self._shutdown_initiated = True

        # Run async shutdown in event loop
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run shutdown coroutine
            loop.run_until_complete(self._shutdown())

            # Clean exit
            self.logger.info("Graceful shutdown complete")
            sys.exit(0)

        except Exception as e:
            self.logger.critical(
                "shutdown_failed",
                extra={"error": str(e)},
                exc_info=True
            )
            sys.exit(1)

    async def _shutdown(self):
        """Perform graceful shutdown operations.

        This is the core shutdown logic:
        1. Pause trading (prevent new signals)
        2. Cancel all pending orders
        3. Close all open positions
        4. Save state
        5. Send notification
        """
        shutdown_start = datetime.now()

        try:
            # Step 1: Pause trading (if manager available)
            if self.manager:
                self.manager.enabled = False
                self.logger.info("Trading paused - no new signals will be processed")

            # Step 2: Cancel all pending orders
            await self._cancel_pending_orders()

            # Step 3: Close all open positions
            await self._close_open_positions()

            # Step 4: Save state
            self._save_state()

            # Step 5: Send shutdown notification
            await self._send_shutdown_notification(success=True)

            shutdown_duration = (datetime.now() - shutdown_start).total_seconds()
            self.logger.info(
                "shutdown_completed",
                extra={"duration_seconds": shutdown_duration}
            )

        except asyncio.TimeoutError:
            self.logger.error(
                "shutdown_timeout",
                extra={"timeout_seconds": self.timeout}
            )
            await self._send_shutdown_notification(success=False, reason="Timeout")
            raise

        except Exception as e:
            self.logger.critical(
                "shutdown_error",
                extra={"error": str(e)},
                exc_info=True
            )
            await self._send_shutdown_notification(success=False, reason=str(e))
            raise

    async def _cancel_pending_orders(self):
        """Cancel all pending orders on the exchange.

        Uses provider to fetch and cancel all open orders.
        """
        if not self.provider:
            self.logger.warning("No provider available - skipping order cancellation")
            return

        try:
            self.logger.info("Canceling all pending orders...")

            # Fetch open orders
            try:
                open_orders = await asyncio.wait_for(
                    self.provider.fetch_open_orders(),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                self.logger.error("Timeout fetching open orders")
                return
            except Exception as e:
                self.logger.error(f"Failed to fetch open orders: {e}")
                return

            if not open_orders:
                self.logger.info("No pending orders to cancel")
                return

            self.logger.info(f"Found {len(open_orders)} pending orders")

            # Cancel each order
            canceled_count = 0
            for order in open_orders:
                try:
                    await asyncio.wait_for(
                        self.provider.cancel_order(order.id, order.symbol),
                        timeout=5.0
                    )
                    canceled_count += 1
                    self.logger.info(f"Canceled order {order.id} for {order.symbol}")
                except asyncio.TimeoutError:
                    self.logger.error(f"Timeout canceling order {order.id}")
                except Exception as e:
                    self.logger.error(f"Failed to cancel order {order.id}: {e}")

            self.logger.info(
                "order_cancellation_complete",
                extra={
                    "total_orders": len(open_orders),
                    "canceled": canceled_count,
                    "failed": len(open_orders) - canceled_count,
                }
            )

        except Exception as e:
            self.logger.error(f"Error in cancel_pending_orders: {e}", exc_info=True)

    async def _close_open_positions(self):
        """Close all open positions using market orders.

        Fetches all open positions and closes them immediately with market orders
        to ensure positions don't remain open after shutdown.
        """
        if not self.provider:
            self.logger.warning("No provider available - skipping position closure")
            return

        try:
            self.logger.info("Closing all open positions...")

            # Fetch open positions
            try:
                positions = await asyncio.wait_for(
                    self.provider.fetch_positions(),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                self.logger.error("Timeout fetching positions")
                return
            except Exception as e:
                self.logger.error(f"Failed to fetch positions: {e}")
                return

            # Filter to only open positions (size > 0)
            open_positions = [pos for pos in positions if abs(pos.size) > 0]

            if not open_positions:
                self.logger.info("No open positions to close")
                return

            self.logger.info(f"Found {len(open_positions)} open positions")

            # Close each position
            closed_count = 0
            for position in open_positions:
                try:
                    # Determine close side (opposite of position side)
                    close_side = "sell" if position.side == "long" else "buy"

                    self.logger.info(
                        f"Closing {position.side} position: {position.symbol} "
                        f"(size: {position.size}, entry: {position.entry_price})"
                    )

                    # Place market order to close position
                    await asyncio.wait_for(
                        self.provider.place_order(
                            symbol=position.symbol,
                            side=close_side,
                            order_type="market",
                            amount=abs(position.size),
                        ),
                        timeout=10.0
                    )

                    closed_count += 1
                    self.logger.info(f"Closed position for {position.symbol}")

                except asyncio.TimeoutError:
                    self.logger.error(f"Timeout closing position {position.symbol}")
                except Exception as e:
                    self.logger.error(f"Failed to close position {position.symbol}: {e}")

            self.logger.info(
                "position_closure_complete",
                extra={
                    "total_positions": len(open_positions),
                    "closed": closed_count,
                    "failed": len(open_positions) - closed_count,
                }
            )

        except Exception as e:
            self.logger.error(f"Error in close_open_positions: {e}", exc_info=True)

    def _save_state(self):
        """Save current trading state to disk.

        Persists TradingState so bot can resume from last known state.
        """
        if not self.state:
            self.logger.warning("No state available - skipping state save")
            return

        try:
            self.logger.info("Saving trading state...")

            # Save state (assumes TradingState has save method)
            if hasattr(self.state, 'save') and self.manager:
                self.state.save(self.manager.spec_dir)
                self.logger.info("Trading state saved successfully")
            else:
                self.logger.warning("State save method not available")

        except Exception as e:
            self.logger.error(f"Failed to save state: {e}", exc_info=True)

    async def _send_shutdown_notification(self, success: bool, reason: Optional[str] = None):
        """Send shutdown notification via alert manager.

        Args:
            success: Whether shutdown completed successfully
            reason: Optional failure reason if not successful
        """
        if not self.alert_manager:
            return

        try:
            status = "completed" if success else "failed"
            message = f"Trading bot shutdown {status}"
            if reason:
                message += f" - {reason}"

            self.logger.info(f"Sending shutdown notification: {message}")

            # Send alert (use send_alert or equivalent method)
            if hasattr(self.alert_manager, 'send_alert'):
                await asyncio.wait_for(
                    self.alert_manager.send_alert(
                        message=message,
                        priority="high",
                        metadata={
                            "shutdown_status": status,
                            "reason": reason,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                    timeout=5.0
                )
            else:
                self.logger.warning("Alert manager send_alert method not available")

        except asyncio.TimeoutError:
            self.logger.warning("Timeout sending shutdown notification")
        except Exception as e:
            self.logger.error(f"Failed to send shutdown notification: {e}")
