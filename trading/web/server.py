"""FastAPI Dashboard Server - Real-time trading monitoring backend.

Provides REST API and WebSocket endpoints for the trading dashboard.
Serves static frontend files and manages real-time data broadcasting.
"""

import logging
import asyncio
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

from ..config import TradingConfig
from ..memory.trade_history import TradeJournal
from .websocket import WebSocketManager, MessageType
from ..safety.kill_switch import KillSwitch
from ..monitoring.metrics_tracker import MetricsTracker
from ..monitoring.system_health import SystemHealthMonitor
from ..monitoring.alert_manager import AlertManager
from ..api.ml_serving import router as ml_router
from ..logging.log_context import LogContext


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each API request for correlation tracking."""

    async def dispatch(self, request, call_next):
        """Add correlation ID to request context and response headers.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain

        Returns:
            Response with X-Request-ID header
        """
        # Generate correlation ID for this request
        correlation_id = LogContext.generate_correlation_id()
        LogContext.set_correlation_id(correlation_id)

        # Process request
        response = await call_next(request)

        # Add to response headers for debugging
        response.headers["X-Request-ID"] = correlation_id

        return response


class DashboardServer:
    """FastAPI-based dashboard server with WebSocket support.

    Provides real-time monitoring interface for trading bot with:
    - REST API for historical data
    - WebSocket for live updates
    - Static file serving for frontend

    Example:
        >>> server = DashboardServer(config, port=5173)
        >>> await server.start()  # Non-blocking
        >>> await server.broadcast_position_update({...})
        >>> await server.stop()
    """

    def __init__(
        self,
        config: TradingConfig,
        trade_journal: Optional[TradeJournal] = None,
        kill_switch: Optional[KillSwitch] = None,
        metrics_tracker: Optional[MetricsTracker] = None,
        health_monitor: Optional[SystemHealthMonitor] = None,
        alert_manager: Optional[AlertManager] = None,
        host: str = "0.0.0.0",
        port: int = 5173,
    ):
        """Initialize dashboard server.

        Args:
            config: Trading configuration
            trade_journal: Optional trade journal for historical data
            kill_switch: Optional kill switch instance for safety endpoints
            metrics_tracker: Optional real-time metrics tracker
            health_monitor: Optional system health monitor
            alert_manager: Optional alert manager for testing alerts
            host: Server host (default: 0.0.0.0)
            port: Server port (default: 5173)
        """
        self.config = config
        self.trade_journal = trade_journal
        self.kill_switch = kill_switch
        self.metrics_tracker = metrics_tracker
        self.health_monitor = health_monitor
        self.alert_manager = alert_manager
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)

        # WebSocket manager
        self.ws_manager = WebSocketManager()

        # FastAPI app
        self.app = FastAPI(
            title="LLM-TradeBot Dashboard",
            description="Real-time trading monitoring interface",
            version="1.0.0",
        )

        # CORS middleware for development
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Request ID middleware for correlation tracking
        self.app.add_middleware(RequestIDMiddleware)

        # Server task
        self._server_task: Optional[asyncio.Task] = None
        self._server = None

        # Register ML serving router
        self.app.include_router(ml_router)

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes and static file serving."""

        # WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket connection for real-time updates."""
            await self.ws_manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive, handle ping/pong
                    data = await websocket.receive_text()
                    if data == "ping":
                        await websocket.send_text("pong")
            except WebSocketDisconnect:
                self.ws_manager.disconnect(websocket)

        # Health check (legacy API endpoint)
        @self.app.get("/api/health")
        async def health_check():
            """Server health check."""
            return JSONResponse({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "provider": self.config.provider,
                "testnet": self.config.testnet,
            })

        # Docker health check endpoint (NEW - Phase 11 Task 2)
        @self.app.get("/health")
        async def docker_health_check():
            """Lightweight health check for Docker HEALTHCHECK.

            This is a simpler version of /api/v1/health/status designed specifically
            for Docker health checks. Returns 200 if healthy, 503 if unhealthy.

            Checks:
            - FastAPI server responsive (implicit - if this runs, server is up)
            - Kill switch not active
            - Circuit breaker not tripped
            - System health not critical

            Returns:
                200: {"status": "healthy", "timestamp": "..."}
                503: Service unavailable with error detail
            """
            try:
                # Check kill switch
                if self.kill_switch and self.kill_switch.is_active():
                    raise HTTPException(
                        status_code=503,
                        detail="Kill switch active - trading stopped"
                    )

                # Check system health
                if self.health_monitor:
                    health_status = self.health_monitor.get_health_status()
                    if health_status.health_level == "CRITICAL":
                        raise HTTPException(
                            status_code=503,
                            detail=f"System health critical: {', '.join(health_status.issues)}"
                        )

                return JSONResponse({
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat()
                })

            except HTTPException:
                # Re-raise HTTPException as-is
                raise
            except Exception as e:
                # Catch any unexpected errors
                self.logger.error(f"Health check failed: {e}", exc_info=True)
                raise HTTPException(
                    status_code=503,
                    detail=f"Health check failed: {str(e)}"
                )

        # Get current positions
        @self.app.get("/api/positions")
        async def get_positions():
            """Get current open positions."""
            if not self.trade_journal:
                return JSONResponse([])

            # Get open positions from trade journal
            open_trades = self.trade_journal.get_open_positions()
            return JSONResponse([self._format_position(trade) for trade in open_trades])

        # Get trade history
        @self.app.get("/api/trades")
        async def get_trades(limit: int = 50, offset: int = 0):
            """Get trade history with pagination."""
            if not self.trade_journal:
                return JSONResponse([])

            trades = self.trade_journal.get_recent_trades(limit=limit, offset=offset)
            return JSONResponse([self._format_trade(trade) for trade in trades])

        # Get performance metrics
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get performance metrics."""
            if not self.trade_journal:
                return JSONResponse({})

            return JSONResponse({
                "total_trades": self.trade_journal.get_total_trades(),
                "win_rate": self.trade_journal.calculate_win_rate(),
                "total_pnl": self.trade_journal.calculate_total_pnl(),
                "sharpe_ratio": self.trade_journal.calculate_sharpe_ratio(),
                "max_drawdown": self.trade_journal.calculate_max_drawdown(),
                "avg_win": self.trade_journal.calculate_average_win(),
                "avg_loss": self.trade_journal.calculate_average_loss(),
            })

        # Get real-time metrics (NEW - Phase 10 Task 1)
        @self.app.get("/api/v1/metrics/realtime")
        async def get_realtime_metrics():
            """Get real-time performance metrics snapshot.

            Returns current metrics from MetricsTracker including:
            - Sharpe ratio, Sortino ratio
            - Current drawdown, max drawdown
            - Win rate, consecutive losses
            - Total/daily/weekly P&L
            - Current equity, peak equity
            """
            if not self.metrics_tracker:
                return JSONResponse({
                    "error": "Metrics tracker not initialized",
                    "metrics": {}
                })

            metrics = self.metrics_tracker.get_current_metrics()
            return JSONResponse(metrics.to_dict())

        # Get system health status (NEW - Phase 10 Task 2)
        @self.app.get("/api/v1/health/status")
        async def get_health_status():
            """Get current system health status.

            Returns aggregated health status including:
            - Overall health level (HEALTHY, DEGRADED, CRITICAL)
            - Kill switch status
            - Circuit breaker status
            - Position utilization
            - API connection status
            """
            if not self.health_monitor:
                return JSONResponse({
                    "error": "Health monitor not initialized",
                    "health": {
                        "level": "unknown",
                        "timestamp": datetime.now().isoformat(),
                    }
                })

            health = self.health_monitor.get_health_status()
            return JSONResponse(health.to_dict())

        # Get safety controls detailed status (NEW - Phase 10 Task 2)
        @self.app.get("/api/v1/health/safety")
        async def get_safety_status():
            """Get detailed safety controls status.

            Returns detailed status of:
            - Kill switch (active, reason, triggered_by, timestamp)
            - Circuit breaker (open, reason, trip_time, thresholds)
            - Position limits (current, max, utilization)
            """
            if not self.health_monitor:
                return JSONResponse({
                    "error": "Health monitor not initialized",
                    "safety": {}
                })

            safety_status = self.health_monitor.get_safety_status()
            return JSONResponse(safety_status)

        # Get equity curve data
        @self.app.get("/api/equity-curve")
        async def get_equity_curve(days: int = 30):
            """Get equity curve data for charting."""
            if not self.trade_journal:
                return JSONResponse([])

            equity_data = self.trade_journal.get_equity_curve(days=days)
            return JSONResponse(equity_data)

        # Get agent decisions
        @self.app.get("/api/agent-decisions")
        async def get_agent_decisions(limit: int = 20):
            """Get recent agent decisions."""
            if not self.trade_journal:
                return JSONResponse([])

            decisions = self.trade_journal.get_agent_decisions(limit=limit)
            return JSONResponse(decisions)

        # Kill switch endpoints (safety controls)
        @self.app.post("/api/v1/safety/kill-switch/trigger")
        async def trigger_kill_switch(
            request: Request,
            x_hmac_signature: Optional[str] = Header(None, alias="X-HMAC-Signature")
        ):
            """Trigger kill switch - emergency shutdown.

            Requires HMAC-SHA256 signature in X-HMAC-Signature header.

            Body:
                {
                    "reason": "Emergency shutdown reason",
                    "triggered_by": "user@example.com"
                }
            """
            if not self.kill_switch:
                raise HTTPException(status_code=503, detail="Kill switch not initialized")

            # Verify HMAC signature
            body = await request.body()
            body_str = body.decode()

            if not x_hmac_signature:
                raise HTTPException(status_code=401, detail="Missing X-HMAC-Signature header")

            if not self.kill_switch.verify_hmac(body_str, x_hmac_signature):
                self.logger.warning("kill_switch_invalid_signature")
                raise HTTPException(status_code=401, detail="Invalid HMAC signature")

            # Parse body
            try:
                data = json.loads(body_str)
                reason = data.get("reason", "No reason provided")
                triggered_by = data.get("triggered_by", "Unknown")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON body")

            # Trigger kill switch
            success = self.kill_switch.trigger(reason, triggered_by)

            return JSONResponse({
                "success": success,
                "message": "Kill switch triggered - ALL TRADING STOPPED" if success else "Kill switch already active",
                "status": self.kill_switch.get_status()
            })

        @self.app.get("/api/v1/safety/kill-switch/status")
        async def get_kill_switch_status(
            x_hmac_signature: Optional[str] = Header(None, alias="X-HMAC-Signature")
        ):
            """Get kill switch status.

            Requires HMAC-SHA256 signature in X-HMAC-Signature header.
            Message to sign is empty string for GET requests.
            """
            if not self.kill_switch:
                raise HTTPException(status_code=503, detail="Kill switch not initialized")

            # Verify HMAC signature (empty message for GET)
            if not x_hmac_signature:
                raise HTTPException(status_code=401, detail="Missing X-HMAC-Signature header")

            if not self.kill_switch.verify_hmac("", x_hmac_signature):
                self.logger.warning("kill_switch_status_invalid_signature")
                raise HTTPException(status_code=401, detail="Invalid HMAC signature")

            return JSONResponse(self.kill_switch.get_status())

        @self.app.post("/api/v1/safety/kill-switch/reset")
        async def reset_kill_switch(
            request: Request,
            x_hmac_signature: Optional[str] = Header(None, alias="X-HMAC-Signature")
        ):
            """Reset kill switch - resume trading.

            Requires HMAC-SHA256 signature in X-HMAC-Signature header.

            Body:
                {
                    "reset_by": "user@example.com"
                }
            """
            if not self.kill_switch:
                raise HTTPException(status_code=503, detail="Kill switch not initialized")

            # Verify HMAC signature
            body = await request.body()
            body_str = body.decode()

            if not x_hmac_signature:
                raise HTTPException(status_code=401, detail="Missing X-HMAC-Signature header")

            if not self.kill_switch.verify_hmac(body_str, x_hmac_signature):
                self.logger.warning("kill_switch_reset_invalid_signature")
                raise HTTPException(status_code=401, detail="Invalid HMAC signature")

            # Parse body
            try:
                data = json.loads(body_str)
                reset_by = data.get("reset_by", "Unknown")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON body")

            # Reset kill switch
            success = self.kill_switch.reset(reset_by)

            return JSONResponse({
                "success": success,
                "message": "Kill switch reset - trading may resume" if success else "Kill switch already inactive",
                "status": self.kill_switch.get_status()
            })

        # Alert testing endpoint (Phase 10 Task 3)
        @self.app.post("/api/v1/alerts/test")
        async def test_alert(
            request: Request,
            x_hmac_signature: Optional[str] = Header(None, alias="X-HMAC-Signature")
        ):
            """Send test alert to verify channel configuration.

            Requires HMAC-SHA256 signature in X-HMAC-Signature header.

            Body:
                {
                    "channel": "slack" | "email" | "telegram" | "all",
                    "test_message": "Optional custom test message"
                }

            Returns success/failure status for each channel.
            """
            if not self.alert_manager:
                raise HTTPException(status_code=503, detail="Alert manager not initialized")

            # Verify HMAC signature
            body = await request.body()
            body_str = body.decode()

            if not x_hmac_signature:
                raise HTTPException(status_code=401, detail="Missing X-HMAC-Signature header")

            # Use kill switch HMAC verification (same security requirement)
            if not self.kill_switch:
                raise HTTPException(status_code=503, detail="Kill switch not initialized")

            if not self.kill_switch.verify_hmac(body_str, x_hmac_signature):
                self.logger.warning("test_alert_invalid_signature")
                raise HTTPException(status_code=401, detail="Invalid HMAC signature")

            # Parse body
            try:
                data = json.loads(body_str) if body_str else {}
                channel = data.get("channel", "all")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON body")

            # Send test alert
            result = await self.alert_manager.send_test_alert(channel=channel if channel != "all" else None)

            return JSONResponse(result)

        # Static files - serve frontend
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

            @self.app.get("/")
            async def serve_index():
                """Serve index.html."""
                index_path = static_dir / "index.html"
                if index_path.exists():
                    return FileResponse(str(index_path))
                else:
                    raise HTTPException(status_code=404, detail="Dashboard UI not found")

    def _format_position(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Format trade for position display."""
        return {
            "symbol": trade.get("symbol"),
            "side": trade.get("side"),
            "size": trade.get("amount"),
            "entry_price": trade.get("entry_price"),
            "current_price": trade.get("current_price", trade.get("entry_price")),
            "pnl": trade.get("pnl", 0.0),
            "pnl_pct": trade.get("pnl_pct", 0.0),
            "opened_at": trade.get("timestamp"),
        }

    def _format_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Format trade for history display."""
        return {
            "id": trade.get("id"),
            "symbol": trade.get("symbol"),
            "side": trade.get("side"),
            "amount": trade.get("amount"),
            "entry_price": trade.get("entry_price"),
            "exit_price": trade.get("exit_price"),
            "pnl": trade.get("pnl"),
            "pnl_pct": trade.get("pnl_pct"),
            "opened_at": trade.get("opened_at"),
            "closed_at": trade.get("closed_at"),
            "status": trade.get("status"),
        }

    async def start(self):
        """Start dashboard server (non-blocking)."""
        if self._server_task is not None:
            self.logger.warning("Dashboard server already running")
            return

        self.logger.info(f"Starting dashboard server on {self.host}:{self.port}")

        # Create server config
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )

        # Create server
        self._server = uvicorn.Server(config)

        # Start in background task
        self._server_task = asyncio.create_task(self._server.serve())

        self.logger.info(f"Dashboard available at http://{self.host}:{self.port}")

    async def stop(self):
        """Stop dashboard server."""
        if self._server_task is None:
            return

        self.logger.info("Stopping dashboard server")

        # Shutdown server
        if self._server:
            self._server.should_exit = True

        # Wait for task to complete
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass

        self._server_task = None
        self._server = None

        self.logger.info("Dashboard server stopped")

    async def broadcast_position_update(self, position: Dict[str, Any]):
        """Broadcast position update to all connected clients.

        Args:
            position: Position data to broadcast
        """
        await self.ws_manager.broadcast(MessageType.POSITION_UPDATE, position)

    async def broadcast_trade_executed(self, trade: Dict[str, Any]):
        """Broadcast trade execution to all connected clients.

        Args:
            trade: Trade data to broadcast
        """
        await self.ws_manager.broadcast(MessageType.TRADE_EXECUTED, trade)

    async def broadcast_metrics_update(self, metrics: Dict[str, Any]):
        """Broadcast metrics update to all connected clients.

        Args:
            metrics: Metrics data to broadcast
        """
        await self.ws_manager.broadcast(MessageType.METRICS_UPDATE, metrics)

    async def broadcast_agent_decision(self, decision: Dict[str, Any]):
        """Broadcast agent decision to all connected clients.

        Args:
            decision: Agent decision data to broadcast
        """
        await self.ws_manager.broadcast(MessageType.AGENT_DECISION, decision)

    async def broadcast_alert(self, alert: Dict[str, Any]):
        """Broadcast alert to all connected clients.

        Args:
            alert: Alert data to broadcast
        """
        await self.ws_manager.broadcast(MessageType.ALERT, alert)

    async def broadcast_health_update(self, health: Dict[str, Any]):
        """Broadcast system health update to all connected clients.

        Args:
            health: Health status data to broadcast
        """
        await self.ws_manager.broadcast(MessageType.HEALTH_UPDATE, health)

    async def broadcast_safety_update(self, safety: Dict[str, Any]):
        """Broadcast safety controls update to all connected clients.

        Args:
            safety: Safety status data to broadcast
        """
        await self.ws_manager.broadcast(MessageType.SAFETY_UPDATE, safety)

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._server_task is not None and not self._server_task.done()
