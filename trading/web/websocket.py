"""WebSocket Manager - Real-time bidirectional communication.

Manages WebSocket connections and message broadcasting for the dashboard.
Handles connection lifecycle, message routing, and client state management.
"""

import logging
import json
from typing import Set, Dict, Any
from enum import Enum
from datetime import datetime

from fastapi import WebSocket


class MessageType(str, Enum):
    """WebSocket message types."""

    # Position updates
    POSITION_UPDATE = "position_update"

    # Trade events
    TRADE_EXECUTED = "trade_executed"
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"

    # Metrics updates
    METRICS_UPDATE = "metrics_update"
    EQUITY_UPDATE = "equity_update"

    # Agent events
    AGENT_DECISION = "agent_decision"
    AGENT_ANALYSIS = "agent_analysis"

    # System events
    ALERT = "alert"
    CIRCUIT_BREAKER = "circuit_breaker"
    CONNECTION_STATUS = "connection_status"

    # Client events
    PING = "ping"
    PONG = "pong"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"


class WebSocketManager:
    """Manages WebSocket connections and message broadcasting.

    Handles:
    - Connection lifecycle (connect/disconnect)
    - Message broadcasting to all/specific clients
    - Subscription management
    - Client state tracking

    Example:
        >>> manager = WebSocketManager()
        >>> await manager.connect(websocket)
        >>> await manager.broadcast(MessageType.TRADE_EXECUTED, {...})
        >>> manager.disconnect(websocket)
    """

    def __init__(self):
        """Initialize WebSocket manager."""
        self.logger = logging.getLogger(__name__)

        # Active connections
        self.active_connections: Set[WebSocket] = set()

        # Client subscriptions (websocket -> set of message types)
        self.subscriptions: Dict[WebSocket, Set[MessageType]] = {}

        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket):
        """Accept and register new WebSocket connection.

        Args:
            websocket: WebSocket connection to register
        """
        await websocket.accept()
        self.active_connections.add(websocket)

        # Initialize subscriptions (all types by default)
        self.subscriptions[websocket] = set(MessageType)

        # Store connection metadata
        self.connection_metadata[websocket] = {
            "connected_at": datetime.now().isoformat(),
            "client_info": websocket.client if websocket.client else None,
        }

        self.logger.info(
            f"WebSocket connected. Total connections: {len(self.active_connections)}"
        )

        # Send welcome message
        await self.send_personal_message(
            websocket,
            MessageType.CONNECTION_STATUS,
            {
                "status": "connected",
                "message": "Welcome to LLM-TradeBot Dashboard",
                "timestamp": datetime.now().isoformat(),
            },
        )

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection.

        Args:
            websocket: WebSocket connection to remove
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        if websocket in self.subscriptions:
            del self.subscriptions[websocket]

        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]

        self.logger.info(
            f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
        )

    async def send_personal_message(
        self,
        websocket: WebSocket,
        message_type: MessageType,
        data: Dict[str, Any],
    ):
        """Send message to specific client.

        Args:
            websocket: Target WebSocket connection
            message_type: Type of message
            data: Message payload
        """
        message = self._create_message(message_type, data)

        try:
            await websocket.send_json(message)
        except Exception as e:
            self.logger.error(f"Failed to send message to client: {e}")
            self.disconnect(websocket)

    async def broadcast(
        self,
        message_type: MessageType,
        data: Dict[str, Any],
        exclude: Set[WebSocket] = None,
    ):
        """Broadcast message to all connected clients.

        Args:
            message_type: Type of message
            data: Message payload
            exclude: Set of websockets to exclude from broadcast
        """
        if exclude is None:
            exclude = set()

        message = self._create_message(message_type, data)

        # Send to all subscribed clients
        disconnected = set()

        for connection in self.active_connections:
            # Skip excluded connections
            if connection in exclude:
                continue

            # Check subscription
            if connection in self.subscriptions:
                if message_type not in self.subscriptions[connection]:
                    continue

            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.error(f"Failed to broadcast to client: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_to_subscribers(
        self,
        message_type: MessageType,
        data: Dict[str, Any],
        subscribers_only: Set[MessageType],
    ):
        """Broadcast to clients subscribed to specific message types.

        Args:
            message_type: Type of message
            data: Message payload
            subscribers_only: Only send to clients subscribed to these types
        """
        message = self._create_message(message_type, data)
        disconnected = set()

        for connection in self.active_connections:
            # Check if client is subscribed to any of the required types
            if connection in self.subscriptions:
                client_subs = self.subscriptions[connection]
                if not any(sub in subscribers_only for sub in client_subs):
                    continue

            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.error(f"Failed to send to subscriber: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    def _create_message(
        self,
        message_type: MessageType,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create formatted WebSocket message.

        Args:
            message_type: Type of message
            data: Message payload

        Returns:
            Formatted message dict
        """
        return {
            "type": message_type.value,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }

    async def handle_client_message(self, websocket: WebSocket, message: str):
        """Handle incoming client message.

        Args:
            websocket: Client WebSocket connection
            message: Raw message string
        """
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == MessageType.PING.value:
                # Respond to ping
                await self.send_personal_message(
                    websocket,
                    MessageType.PONG,
                    {"timestamp": datetime.now().isoformat()},
                )

            elif message_type == MessageType.SUBSCRIBE.value:
                # Update subscriptions
                sub_types = data.get("subscriptions", [])
                if websocket in self.subscriptions:
                    self.subscriptions[websocket] = {
                        MessageType(t) for t in sub_types if t in MessageType.__members__.values()
                    }
                    self.logger.debug(f"Client subscriptions updated: {sub_types}")

            elif message_type == MessageType.UNSUBSCRIBE.value:
                # Remove subscriptions
                unsub_types = data.get("subscriptions", [])
                if websocket in self.subscriptions:
                    for t in unsub_types:
                        if t in MessageType.__members__.values():
                            self.subscriptions[websocket].discard(MessageType(t))
                    self.logger.debug(f"Client unsubscribed from: {unsub_types}")

        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse client message: {message}")
        except Exception as e:
            self.logger.error(f"Error handling client message: {e}")

    def get_active_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)

    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about all active connections."""
        return {
            "total_connections": len(self.active_connections),
            "connections": [
                {
                    "client": str(ws.client) if ws.client else "unknown",
                    "metadata": self.connection_metadata.get(ws, {}),
                    "subscriptions": [
                        t.value for t in self.subscriptions.get(ws, set())
                    ],
                }
                for ws in self.active_connections
            ],
        }
