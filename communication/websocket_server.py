"""
IITP Digital Twin - External Python WebSocket server.

This module runs OUTSIDE Omniverse and acts as the communication bridge
between the external insulin-pump digital-twin system and the Omniverse
Kit extension.

Direction:

    Python Digital Twin
            |
            | telemetry JSON
            v
       WebSocketServer
            |
            v
       Omniverse Client

    Omniverse Client
            |
            | command JSON
            v
       WebSocketServer
            |
            v
       Python Digital Twin

The server is intentionally transport-focused. It does not:
    - run SimGlucose
    - execute DT1/DT2
    - decide whether insulin is safe
    - actuate a real insulin pump
    - modify USD

Those responsibilities remain outside this communication module.

Default endpoint:

    ws://127.0.0.1:8765

The server supports:
    - multiple Omniverse clients
    - telemetry broadcast
    - command reception
    - response/error messages
    - heartbeat messages
    - client connect/disconnect callbacks
    - bounded connection management
    - graceful async shutdown

Dependency:
    pip install websockets

The Omniverse side uses Kit's asyncio integration. NVIDIA documents
omni.kit.async_engine as the bridge between Python asyncio and Kit's
application update loop. This server itself runs in the external Python
process and therefore uses standard asyncio.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional, Set

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError as exc:  # pragma: no cover
    websockets = None
    ConnectionClosed = Exception
    _WEBSOCKETS_IMPORT_ERROR = exc
else:
    _WEBSOCKETS_IMPORT_ERROR = None

try:
    from .message_schema import (
        TYPE_COMMAND,
        TYPE_HEARTBEAT,
        TYPE_TELEMETRY,
        build_error_message,
        build_response_message,
        deserialize_message,
        serialize_message,
        validate_message,
    )
except ImportError:
    from message_schema import (  # type: ignore
        TYPE_COMMAND,
        TYPE_HEARTBEAT,
        TYPE_TELEMETRY,
        build_error_message,
        build_response_message,
        deserialize_message,
        serialize_message,
        validate_message,
    )


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------

MessageHandler = Callable[
    [Dict[str, Any], Any],
    Optional[Awaitable[Optional[Dict[str, Any]]]],
]

ClientHandler = Callable[
    [Any],
    Optional[Awaitable[None]],
]


# ----------------------------------------------------------------------
# WebSocket server
# ----------------------------------------------------------------------

class WebSocketServer:
    """
    Async WebSocket server for the IITP digital-twin communication layer.

    Example:

        server = WebSocketServer(
            host="127.0.0.1",
            port=8765,
        )

        await server.start()

        await server.broadcast_telemetry(telemetry)

        # Keep the application alive...
        await server.wait_closed()

        await server.stop()

    For a simulation loop, `broadcast_telemetry()` can simply be called
    after each simulation timestep.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        max_clients: int = 4,
        ping_interval: float = 20.0,
        ping_timeout: float = 20.0,
        close_timeout: float = 5.0,
        max_message_size: int = 2 * 1024 * 1024,
        logger: Optional[logging.Logger] = None,
        on_message: Optional[MessageHandler] = None,
        on_client_connected: Optional[ClientHandler] = None,
        on_client_disconnected: Optional[ClientHandler] = None,
    ):
        self.host = str(host)
        self.port = int(port)

        self.max_clients = max(1, int(max_clients))
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.close_timeout = close_timeout
        self.max_message_size = max_message_size

        self.logger = logger or logging.getLogger(
            "iitp.digital_twin.websocket_server"
        )

        self.on_message = on_message
        self.on_client_connected = on_client_connected
        self.on_client_disconnected = on_client_disconnected

        self._server = None
        self._clients: Set[Any] = set()
        self._client_tasks: Set[asyncio.Task] = set()

        self._started = False
        self._stopping = False

        self._closed_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start listening for Omniverse WebSocket connections."""
        self._ensure_dependency()

        if self._started:
            self.logger.warning(
                "WebSocket server is already running"
            )
            return

        self._stopping = False
        self._closed_event.clear()

        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
            close_timeout=self.close_timeout,
            max_size=self.max_message_size,
        )

        self._started = True

        self.logger.info(
            "IITP WebSocket server started at ws://%s:%d",
            self.host,
            self.port,
        )

    async def stop(self) -> None:
        """Gracefully stop the server and disconnect all clients."""
        if not self._started and self._server is None:
            return

        self._stopping = True

        self.logger.info("Stopping IITP WebSocket server")

        if self._server is not None:
            self._server.close()

            try:
                await self._server.wait_closed()
            except Exception as exc:
                self.logger.warning(
                    "Server wait_closed() failed: %s",
                    exc,
                )

            self._server = None

        clients = list(self._clients)

        if clients:
            await asyncio.gather(
                *[
                    self._close_client(
                        client,
                        code=1001,
                        reason="server shutdown",
                    )
                    for client in clients
                ],
                return_exceptions=True,
            )

        tasks = list(self._client_tasks)

        if tasks:
            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

        self._clients.clear()
        self._client_tasks.clear()

        self._started = False
        self._stopping = False
        self._closed_event.set()

        self.logger.info("IITP WebSocket server stopped")

    async def wait_closed(self) -> None:
        """Wait until stop() completes."""
        await self._closed_event.wait()

    async def __aenter__(self) -> "WebSocketServer":
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
    # Client management
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._started

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def get_clients(self) -> Set[Any]:
        """Return a snapshot of currently connected clients."""
        return set(self._clients)

    async def _handle_client(
        self,
        websocket,
    ) -> None:
        """
        Handle one connected Omniverse client.

        The callback signature is intentionally compatible with the
        current websockets API where the connection object is passed as
        the first handler argument.
        """
        if len(self._clients) >= self.max_clients:
            self.logger.warning(
                "Rejecting WebSocket client: maximum clients reached"
            )

            await self._close_client(
                websocket,
                code=1013,
                reason="maximum clients reached",
            )
            return

        self._clients.add(websocket)

        task = asyncio.current_task()

        if task is not None:
            self._client_tasks.add(task)

        remote = self._remote_address(websocket)

        self.logger.info(
            "Omniverse client connected: %s",
            remote,
        )

        try:
            await self._notify_client_connected(websocket)

            async for raw_message in websocket:
                await self._process_raw_message(
                    websocket,
                    raw_message,
                )

        except ConnectionClosed:
            self.logger.info(
                "Omniverse client disconnected: %s",
                remote,
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            self.logger.exception(
                "Client handler error for %s: %s",
                remote,
                exc,
            )

        finally:
            self._clients.discard(websocket)

            if task is not None:
                self._client_tasks.discard(task)

            await self._notify_client_disconnected(
                websocket
            )

            self.logger.info(
                "Omniverse client removed: %s",
                remote,
            )

    # ------------------------------------------------------------------
    # Incoming messages
    # ------------------------------------------------------------------

    async def _process_raw_message(
        self,
        websocket,
        raw_message: Any,
    ) -> None:
        try:
            message = deserialize_message(
                raw_message,
                validate=True,
            )

        except Exception as exc:
            self.logger.warning(
                "Invalid WebSocket message: %s",
                exc,
            )

            await self.send(
                websocket,
                build_error_message(
                    str(exc),
                    code="INVALID_MESSAGE",
                ),
            )
            return

        message_type = message.get("type")

        self.logger.debug(
            "Received %s from %s",
            message_type,
            self._remote_address(websocket),
        )

        # Heartbeats are handled internally.
        if message_type == TYPE_HEARTBEAT:
            await self.send(
                websocket,
                {
                    "type": TYPE_HEARTBEAT,
                    "status": "alive",
                },
            )
            return

        response = None

        if self.on_message is not None:
            try:
                response = self.on_message(
                    message,
                    websocket,
                )

                if inspect.isawaitable(response):
                    response = await response

            except Exception as exc:
                self.logger.exception(
                    "Application message handler failed: %s",
                    exc,
                )

                response = build_error_message(
                    str(exc),
                    code="HANDLER_ERROR",
                )

        # A handler can return a response dictionary.
        if response is not None:
            await self.send(
                websocket,
                response,
            )

    # ------------------------------------------------------------------
    # Outgoing messages
    # ------------------------------------------------------------------

    async def send(
        self,
        websocket,
        message: Dict[str, Any],
    ) -> bool:
        """Send one validated JSON message to a client."""
        if websocket not in self._clients:
            return False

        try:
            validate_message(message)

            payload = serialize_message(
                message,
                validate=False,
            )

            await websocket.send(payload)

            return True

        except Exception as exc:
            self.logger.warning(
                "Failed to send WebSocket message: %s",
                exc,
            )

            return False

    async def broadcast(
        self,
        message: Dict[str, Any],
    ) -> int:
        """
        Send a message to every connected Omniverse client.

        Returns the number of successful sends.
        """
        if not self._clients:
            return 0

        validate_message(message)

        clients = list(self._clients)

        results = await asyncio.gather(
            *[
                self.send(client, message)
                for client in clients
            ],
            return_exceptions=True,
        )

        successful = 0

        for result in results:
            if result is True:
                successful += 1

        return successful

    async def broadcast_telemetry(
        self,
        telemetry: Dict[str, Any],
    ) -> int:
        """
        Broadcast one telemetry message to all connected Omniverse clients.
        """
        if telemetry.get("type") != TYPE_TELEMETRY:
            raise ValueError(
                "broadcast_telemetry() requires a telemetry message"
            )

        return await self.broadcast(telemetry)

    async def send_response(
        self,
        websocket,
        request_message_id: str,
        status: str,
        *,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Send a response associated with a command request."""
        message = build_response_message(
            request_message_id=request_message_id,
            status=status,
            result=result,
            error=error,
        )

        return await self.send(
            websocket,
            message,
        )

    # ------------------------------------------------------------------
    # Connection notifications
    # ------------------------------------------------------------------

    async def _notify_client_connected(
        self,
        websocket,
    ) -> None:
        if self.on_client_connected is None:
            return

        try:
            result = self.on_client_connected(websocket)

            if inspect.isawaitable(result):
                await result

        except Exception as exc:
            self.logger.exception(
                "Client-connected callback failed: %s",
                exc,
            )

    async def _notify_client_disconnected(
        self,
        websocket,
    ) -> None:
        if self.on_client_disconnected is None:
            return

        try:
            result = self.on_client_disconnected(websocket)

            if inspect.isawaitable(result):
                await result

        except Exception as exc:
            self.logger.exception(
                "Client-disconnected callback failed: %s",
                exc,
            )

    # ------------------------------------------------------------------
    # Shutdown helpers
    # ------------------------------------------------------------------

    async def _close_client(
        self,
        websocket,
        *,
        code: int,
        reason: str,
    ) -> None:
        try:
            await websocket.close(
                code=code,
                reason=reason,
            )
        except Exception as exc:
            self.logger.debug(
                "Client close failed: %s",
                exc,
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return a serializable server status snapshot."""
        return {
            "running": self._started,
            "host": self.host,
            "port": self.port,
            "client_count": self.client_count,
            "max_clients": self.max_clients,
            "endpoint": f"ws://{self.host}:{self.port}",
        }

    def _remote_address(
        self,
        websocket,
    ) -> str:
        try:
            return str(websocket.remote_address)
        except Exception:
            return "unknown"

    @staticmethod
    def _ensure_dependency() -> None:
        if websockets is None:
            raise RuntimeError(
                "The 'websockets' package is required for "
                "WebSocketServer. Install it with: "
                "python -m pip install websockets"
            ) from _WEBSOCKETS_IMPORT_ERROR


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def create_websocket_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    **kwargs: Any,
) -> WebSocketServer:
    """Create an IITP WebSocketServer."""
    return WebSocketServer(
        host=host,
        port=port,
        **kwargs,
    )


# ----------------------------------------------------------------------
# Standalone development entry point
# ----------------------------------------------------------------------

async def _demo() -> None:
    """
    Minimal development server.

    This is intentionally not the production digital-twin loop.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    async def on_message(
        message: Dict[str, Any],
        websocket,
    ):
        print(
            "Received:",
            message.get("type"),
            message.get("command")
            or message.get("payload", {}).get("command"),
        )

        if message.get("type") == TYPE_COMMAND:
            request_id = message.get("message_id")

            if request_id:
                return build_response_message(
                    request_message_id=request_id,
                    status="RECEIVED",
                    result={
                        "server": "IITP Digital Twin",
                    },
                )

        return None

    server = WebSocketServer(
        host="127.0.0.1",
        port=8765,
        on_message=on_message,
    )

    await server.start()

    print(
        "IITP WebSocket server running at "
        "ws://127.0.0.1:8765"
    )

    try:
        await server.wait_closed()
    except KeyboardInterrupt:
        pass
    finally:
        await server.stop()


if __name__ == "__main__":
    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        pass
