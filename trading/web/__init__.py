"""Web Dashboard - Real-time trading monitoring interface.

FastAPI-based web dashboard with WebSocket support for live updates.
Provides real-time position tracking, P&L charts, agent decisions, and trade history.

Components:
- FastAPI server with REST API endpoints
- WebSocket connections for live updates
- Interactive frontend with charts and metrics

Features:
- Live position monitoring
- Real-time P&L charts
- Agent decision tracking
- Trade history browser
- Risk metrics dashboard
- Performance analytics

Example Usage:
    ```python
    from trading.web import DashboardServer
    from trading.config import TradingConfig

    # Initialize
    config = TradingConfig.from_env("binance_futures")
    server = DashboardServer(config, port=5173)

    # Start server (non-blocking)
    await server.start()

    # Update position in real-time
    await server.broadcast_position_update({
        "symbol": "BTC/USDT",
        "side": "long",
        "size": 0.5,
        "entry_price": 42000,
        "current_price": 42500,
        "pnl": 250.0
    })

    # Shutdown
    await server.stop()
    ```

Configuration:
    Set in TradingConfig or environment:
    ```bash
    DASHBOARD_ENABLED=true
    DASHBOARD_PORT=5173
    DASHBOARD_HOST=0.0.0.0  # or localhost for local-only access
    ```

Access:
    Open http://localhost:5173 in your browser after starting the server.

Security:
    - Dashboard bound to localhost by default for security
    - Set DASHBOARD_HOST=0.0.0.0 to allow external access (use with caution)
    - Consider adding authentication for production deployments
"""

from .server import DashboardServer
from .websocket import WebSocketManager, MessageType

__all__ = [
    "DashboardServer",
    "WebSocketManager",
    "MessageType",
]

__version__ = "1.0.0"
