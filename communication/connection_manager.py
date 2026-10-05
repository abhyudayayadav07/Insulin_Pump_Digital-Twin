"""
IITP Digital Twin - WebSocket connection manager.

This module owns the lifecycle and state of the external Python-side
WebSocket connection pool.

It sits above websocket_server.py:

    websocket_server.py
        |
        v
    ConnectionManager
        |
        +-- connected clients
        +-- client metadata
        +-- connection state
        +-- telemetry broadcast
        +-- command routing
        +-- response routing
        +-- heartbeat
        +-- reconnect/cleanup bookkeeping
        |
        v
    Omniverse Kit extension(s)

The manager intentionally does NOT:
    - run SimGlucose
    - calculate DT1
    - calculate DT2
    - make safety decisions
    - directly actuate insulin

Those responsibilities remain in the external digital-twin system.

The class is transport/orchestration infrastructure. It can be used by the
future digital_twin_server.py to keep the main simulation loop independent
of low-level WebSocket client management.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

try:
    from .message_schema import (
        TYPE_COMMAND,
        TYPE_HEARTBEAT,
        TYPE_RESPONSE,
        TYPE_TELEMETRY,
        build_error_message,
        build_heartbeat_message,
        build_response_message,
        get_message_id,
        get_message_type,
        get_payload,
        validate_message,
    )
    from .websocket_server import WebSocketServer
except ImportError:
    from message_schema import (  # type: ignore
        TYPE_COMMAND,
        TYPE_HEARTBEAT,
        TYPE_RESPONSE,
        TYPE_TELEMETRY,
        build_error_message,
        build_heartbeat_message,
        build_response_message,
        get_message_id,
        get_message_type,
        get_payload,
        validate_message,
    )
    from websocket_server import WebSocketServer  # type: ignore


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------

MessageCallback = Callable[
    [Dict[str, Any], "ClientConnection"],
    Optional[Awaitable[Optional[Dict[str, Any]]]],
]

ConnectionCallback = Callable[
    ["ClientConnection"],
    Optional[Awaitable[None]],
]


# ----------------------------------------------------------------------
# Client state
# ----------------------------------------------------------------------

@dataclass
class ClientConnection:
    """
    Runtime metadata for one WebSocket client.

    The actual WebSocket object is deliberately kept private to the manager
    through the `websocket` field. Application code normally only needs this
    metadata object.
    """

    connection_id: str
    websocket: Any
    remote_address: str
    connected_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    messages_received: int = 0
    messages_sent: int = 0
    telemetry_sent: int = 0
    commands_received: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def connected_duration(self) -> float:
        """Seconds since this client connected."""
        return max(
            0.0,
            time.time() - self.connected_at,
        )

    @property
    def idle_seconds(self) -> float:
        """Seconds since the last received message."""
        return max(
            0.0,
            time.time() - self.last_seen,
        )

    def touch(self) -> None:
        """Update the last-seen timestamp."""
        self.last_seen = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe client metadata."""
        return {
            "connection_id": self.connection_id,
            "remote_address": self.remote_address,
            "connected_at": self.connected_at,
            "last_seen": self.last_seen,
            "connected_duration": self.connected_duration,
            "idle_seconds": self.idle_seconds,
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "telemetry_sent": self.telemetry_sent,
            "commands_received": self.commands_received,
            "metadata": dict(self.metadata),
        }


# ----------------------------------------------------------------------
# Connection manager
# ----------------------------------------------------------------------

class ConnectionManager:
    """
    Manage WebSocket server lifecycle and connected Omniverse clients.

    The manager uses websocket_server.py for actual transport. This class
    provides the higher-level API used by the digital-twin runtime.
    """

    def __init__(
        self,
        server: Optional[WebSocketServer] = None,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        max_clients: int = 4,
        heartbeat_interval: float = 15.0,
        heartbeat_timeout: float = 60.0,
        logger: Optional[logging.Logger] = None,
        on_command: Optional[MessageCallback] = None,
        on_message: Optional[MessageCallback] = None,
        on_client_connected: Optional[ConnectionCallback] = None,
        on_client_disconnected: Optional[ConnectionCallback] = None,
    ):
        self.logger = logger or logging.getLogger(
            "iitp.digital_twin.connection_manager"
        )

        self._external_on_command = on_command
        self._external_on_message = on_message
        self._external_on_client_connected = (
            on_client_connected
        )
        self._external_on_client_disconnected = (
            on_client_disconnected
        )

        if server is None:
            self.server = WebSocketServer(
                host=host,
                port=port,
                max_clients=max_clients,
                on_message=self._handle_server_message,
                on_client_connected=self._handle_client_connected,
                on_client_disconnected=self._handle_client_disconnected,
                logger=self.logger,
            )
        else:
            self.server = server

            # The manager needs the server callbacks. For a supplied server,
            # replace the callbacks intentionally so all state changes pass
            # through this manager.
            self.server.on_message = self._handle_server_message
            self.server.on_client_connected = (
                self._handle_client_connected
            )
            self.server.on_client_disconnected = (
                self._handle_client_disconnected
            )

        self.heartbeat_interval = max(
            1.0,
            float(heartbeat_interval),
        )
        self.heartbeat_timeout = max(
            self.heartbeat_interval,
            float(heartbeat_timeout),
        )

        self._connections: Dict[str, ClientConnection] = {}
        self._websocket_to_id: Dict[int, str] = {}

        self._heartbeat_task: Optional[asyncio.Task] = None

        self._running = False
        self._stopping = False

        self._started_at: Optional[float] = None

        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the WebSocket server and connection manager."""
        if self._running:
            self.logger.warning(
                "Connection manager is already running"
            )
            return

        self._stopping = False

        await self.server.start()

        self._running = True
        self._started_at = time.time()

        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name="iitp-websocket-heartbeat",
        )

        self.logger.info(
            "IITP connection manager started at %s",
            self.endpoint,
        )

    async def stop(self) -> None:
        """Stop heartbeat, disconnect clients, and stop the server."""
        if not self._running and not self.server.is_running:
            return

        self._stopping = True

        self.logger.info(
            "Stopping IITP connection manager"
        )

        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()

            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self.logger.warning(
                    "Heartbeat task shutdown failed: %s",
                    exc,
                )

            self._heartbeat_task = None

        await self.server.stop()

        async with self._lock:
            self._connections.clear()
            self._websocket_to_id.clear()

        self._running = False
        self._stopping = False

        self.logger.info(
            "IITP connection manager stopped"
        )

    async def __aenter__(self) -> "ConnectionManager":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Properties / state
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def endpoint(self) -> str:
        return (
            f"ws://{self.server.host}:{self.server.port}"
        )

    @property
    def client_count(self) -> int:
        return len(self._connections)

    @property
    def uptime_seconds(self) -> float:
        if self._started_at is None:
            return 0.0

        return max(
            0.0,
            time.time() - self._started_at,
        )

    # ------------------------------------------------------------------
    # Client lookup
    # ------------------------------------------------------------------

    def get_client(
        self,
        connection_id: str,
    ) -> Optional[ClientConnection]:
        """Return client metadata by connection ID."""
        return self._connections.get(connection_id)

    def get_clients(self) -> List[ClientConnection]:
        """Return a snapshot of connected clients."""
        return list(self._connections.values())

    def get_client_ids(self) -> List[str]:
        """Return connected client IDs."""
        return list(self._connections.keys())

    def get_connection_id(
        self,
        websocket: Any,
    ) -> Optional[str]:
        """Resolve a WebSocket object to its manager connection ID."""
        return self._websocket_to_id.get(
            id(websocket)
        )

    def _get_or_create_connection(
        self,
        websocket: Any,
    ) -> Optional[ClientConnection]:
        connection_id = self.get_connection_id(websocket)

        if connection_id is not None:
            return self._connections.get(
                connection_id
            )

        # This normally should not happen because the server invokes the
        # connected callback before receiving messages. Keep a defensive
        # fallback for custom server implementations.
        if websocket not in self.server.get_clients():
            return None

        connection = ClientConnection(
            connection_id=str(uuid.uuid4()),
            websocket=websocket,
            remote_address=self._remote_address(websocket),
        )

        self._connections[connection.connection_id] = connection
        self._websocket_to_id[id(websocket)] = (
            connection.connection_id
        )

        return connection

    # ------------------------------------------------------------------
    # Server callbacks
    # ------------------------------------------------------------------

    async def _handle_client_connected(
        self,
        websocket: Any,
    ) -> None:
        connection = ClientConnection(
            connection_id=str(uuid.uuid4()),
            websocket=websocket,
            remote_address=self._remote_address(websocket),
        )

        async with self._lock:
            self._connections[
                connection.connection_id
            ] = connection

            self._websocket_to_id[
                id(websocket)
            ] = connection.connection_id

        self.logger.info(
            "Connection registered: %s (%s)",
            connection.connection_id,
            connection.remote_address,
        )

        if self._external_on_client_connected is not None:
            await self._invoke_callback(
                self._external_on_client_connected,
                connection,
                callback_name="on_client_connected",
            )

    async def _handle_client_disconnected(
        self,
        websocket: Any,
    ) -> None:
        connection_id = self.get_connection_id(websocket)

        if connection_id is None:
            return

        async with self._lock:
            connection = self._connections.pop(
                connection_id,
                None,
            )

            self._websocket_to_id.pop(
                id(websocket),
                None,
            )

        if connection is None:
            return

        self.logger.info(
            "Connection removed: %s",
            connection.connection_id,
        )

        if self._external_on_client_disconnected is not None:
            await self._invoke_callback(
                self._external_on_client_disconnected,
                connection,
                callback_name="on_client_disconnected",
            )

    async def _handle_server_message(
        self,
        message: Dict[str, Any],
        websocket: Any,
    ) -> Optional[Dict[str, Any]]:
        connection = self._get_or_create_connection(
            websocket
        )

        if connection is None:
            self.logger.warning(
                "Received message from unknown client"
            )

            return build_error_message(
                "Unknown WebSocket client",
                code="UNKNOWN_CLIENT",
            )

        connection.touch()
        connection.messages_received += 1

        message_type = get_message_type(message)

        if message_type == TYPE_COMMAND:
            connection.commands_received += 1

            return await self._handle_command(
                message,
                connection,
            )

        if message_type == TYPE_HEARTBEAT:
            return build_heartbeat_message(
                source="digital_twin",
            )

        if self._external_on_message is not None:
            return await self._invoke_message_callback(
                self._external_on_message,
                message,
                connection,
                callback_name="on_message",
            )

        return None

    # ------------------------------------------------------------------
    # Command routing
    # ------------------------------------------------------------------

    async def _handle_command(
        self,
        message: Dict[str, Any],
        connection: ClientConnection,
    ) -> Optional[Dict[str, Any]]:
        """
        Route an Omniverse command to the application callback.

        The callback is responsible for asking the DT/security/decision
        system to approve or reject the requested action.
        """
        if self._external_on_command is None:
            request_id = get_message_id(message)

            if request_id:
                return build_response_message(
                    request_message_id=request_id,
                    status="RECEIVED",
                    result={
                        "handled": False,
                        "reason": "No command handler registered",
                    },
                )

            return None

        try:
            result = self._external_on_command(
                message,
                connection,
            )

            if inspect.isawaitable(result):
                result = await result

            return result

        except Exception as exc:
            self.logger.exception(
                "Command handler failed: %s",
                exc,
            )

            request_id = get_message_id(message)

            if request_id:
                return build_response_message(
                    request_message_id=request_id,
                    status="ERROR",
                    error=str(exc),
                )

            return build_error_message(
                str(exc),
                code="COMMAND_HANDLER_ERROR",
            )

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send_to_client(
        self,
        connection_id: str,
        message: Dict[str, Any],
    ) -> bool:
        """Send one message to a specific client."""
        connection = self.get_client(
            connection_id
        )

        if connection is None:
            return False

        validate_message(message)

        success = await self.server.send(
            connection.websocket,
            message,
        )

        if success:
            connection.messages_sent += 1

            if message.get("type") == TYPE_TELEMETRY:
                connection.telemetry_sent += 1

        return success

    async def broadcast(
        self,
        message: Dict[str, Any],
    ) -> int:
        """Broadcast one message to all connected clients."""
        validate_message(message)

        count = await self.server.broadcast(
            message
        )

        if count:
            for connection in self._connections.values():
                connection.messages_sent += 1

                if message.get("type") == TYPE_TELEMETRY:
                    connection.telemetry_sent += 1

        return count

    async def broadcast_telemetry(
        self,
        telemetry: Dict[str, Any],
    ) -> int:
        """
        Broadcast one telemetry state to all Omniverse clients.

        This is the primary API the digital-twin simulation loop will call.
        """
        if telemetry.get("type") != TYPE_TELEMETRY:
            raise ValueError(
                "broadcast_telemetry() requires type='telemetry'"
            )

        return await self.broadcast(
            telemetry
        )

    async def send_response(
        self,
        connection_id: str,
        request_message_id: str,
        status: str,
        *,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Send a response to a specific command request."""
        message = build_response_message(
            request_message_id=request_message_id,
            status=status,
            result=result,
            error=error,
        )

        return await self.send_to_client(
            connection_id,
            message,
        )

    async def broadcast_response(
        self,
        request_message_id: str,
        status: str,
        *,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> int:
        """Broadcast a response to all clients."""
        message = build_response_message(
            request_message_id=request_message_id,
            status=status,
            result=result,
            error=error,
        )

        return await self.broadcast(
            message
        )

    # ------------------------------------------------------------------
    # Heartbeat / stale clients
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """
        Periodically send heartbeat messages and remove stale connections.

        WebSocket ping/pong is handled by websocket_server.py as a transport
        health mechanism. This application heartbeat is an additional
        protocol-level signal useful for the digital-twin monitor.
        """
        while self._running and not self._stopping:
            try:
                await asyncio.sleep(
                    self.heartbeat_interval
                )

                if not self._running:
                    break

                await self._send_heartbeats()
                await self._remove_stale_clients()

            except asyncio.CancelledError:
                break

            except Exception as exc:
                self.logger.exception(
                    "Heartbeat loop error: %s",
                    exc,
                )

    async def _send_heartbeats(self) -> None:
        if not self._connections:
            return

        message = build_heartbeat_message(
            source="digital_twin",
        )

        clients = list(
            self._connections.values()
        )

        await asyncio.gather(
            *[
                self.server.send(
                    connection.websocket,
                    message,
                )
                for connection in clients
            ],
            return_exceptions=True,
        )

    async def _remove_stale_clients(self) -> None:
        stale_ids = [
            connection.connection_id
            for connection in self._connections.values()
            if connection.idle_seconds
            > self.heartbeat_timeout
        ]

        if not stale_ids:
            return

        self.logger.warning(
            "Removing %d stale WebSocket client(s)",
            len(stale_ids),
        )

        for connection_id in stale_ids:
            await self.disconnect_client(
                connection_id,
                reason="heartbeat timeout",
            )

    async def disconnect_client(
        self,
        connection_id: str,
        *,
        reason: str = "server disconnect",
    ) -> bool:
        """Close a specific managed client."""
        connection = self.get_client(
            connection_id
        )

        if connection is None:
            return False

        try:
            await connection.websocket.close(
                code=1000,
                reason=reason,
            )
        except Exception as exc:
            self.logger.debug(
                "Client close failed: %s",
                exc,
            )

        # websocket_server's disconnect callback will normally remove it.
        async with self._lock:
            self._connections.pop(
                connection_id,
                None,
            )

            self._websocket_to_id.pop(
                id(connection.websocket),
                None,
            )

        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return a JSON-safe connection-manager status."""
        return {
            "running": self._running,
            "endpoint": self.endpoint,
            "uptime_seconds": self.uptime_seconds,
            "client_count": self.client_count,
            "heartbeat_interval": self.heartbeat_interval,
            "heartbeat_timeout": self.heartbeat_timeout,
            "clients": [
                connection.to_dict()
                for connection in self._connections.values()
            ],
        }

    def client_status(
        self,
        connection_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return one client's status."""
        connection = self.get_client(
            connection_id
        )

        if connection is None:
            return None

        return connection.to_dict()

    # ------------------------------------------------------------------
    # Callback utilities
    # ------------------------------------------------------------------

    async def _invoke_callback(
        self,
        callback: ConnectionCallback,
        connection: ClientConnection,
        *,
        callback_name: str,
    ) -> None:
        try:
            result = callback(connection)

            if inspect.isawaitable(result):
                await result

        except Exception as exc:
            self.logger.exception(
                "%s callback failed: %s",
                callback_name,
                exc,
            )

    async def _invoke_message_callback(
        self,
        callback: MessageCallback,
        message: Dict[str, Any],
        connection: ClientConnection,
        *,
        callback_name: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            result = callback(
                message,
                connection,
            )

            if inspect.isawaitable(result):
                result = await result

            return result

        except Exception as exc:
            self.logger.exception(
                "%s callback failed: %s",
                callback_name,
                exc,
            )

            return build_error_message(
                str(exc),
                code="MESSAGE_HANDLER_ERROR",
            )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _remote_address(
        websocket: Any,
    ) -> str:
        try:
            return str(
                websocket.remote_address
            )
        except Exception:
            return "unknown"


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def create_connection_manager(
    server: Optional[WebSocketServer] = None,
    **kwargs: Any,
) -> ConnectionManager:
    """Create an IITP ConnectionManager."""
    return ConnectionManager(
        server=server,
        **kwargs,
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

__all__ = [
    "ClientConnection",
    "ConnectionManager",
    "create_connection_manager",
]
