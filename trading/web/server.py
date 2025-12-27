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
import uvicorn

from ..config import TradingConfig
from ..memory.trade_history import TradeJournal
from .websocket import WebSocketManager, MessageType
from ..safety.kill_switch import KillSwitch


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
        host: str = "0.0.0.0",
        port: int = 5173,
    ):
        """Initialize dashboard server.

        Args:
            config: Trading configuration
            trade_journal: Optional trade journal for historical data
            kill_switch: Optional kill switch instance for safety endpoints
            host: Server host (default: 0.0.0.0)
            port: Server port (default: 5173)
        """
        self.config = config
        self.trade_journal = trade_journal
        self.kill_switch = kill_switch
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

        # Server task
        self._server_task: Optional[asyncio.Task] = None
        self._server = None

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

        # Health check
        @self.app.get("/api/health")
        async def health_check():
            """Server health check."""
            return JSONResponse({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "provider": self.config.provider,
                "testnet": self.config.testnet,
            })

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

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._server_task is not None and not self._server_task.done()
