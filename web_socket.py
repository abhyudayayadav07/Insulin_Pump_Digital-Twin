"""
IITP Digital Twin - Omniverse WebSocket client.

This module is the Omniverse-side client. The external Python digital-twin
process will act as the WebSocket server at:

    ws://127.0.0.1:8765

The client receives complete JSON telemetry messages and forwards them to
the callback supplied by the extension.

The implementation uses Kit's asyncio integration so the asynchronous
WebSocket task can run with Kit's application event loop.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Optional

import carb
import omni.kit.async_engine


class DigitalTwinWebSocketClient:
    """Asynchronous WebSocket client for the IITP digital twin."""

    def __init__(
        self,
        uri: str = "ws://127.0.0.1:8765",
        on_message: Optional[Callable[[dict[str, Any]], None]] = None,
        reconnect_delay: float = 2.0,
    ):
        self.uri = uri
        self.on_message = on_message
        self.reconnect_delay = reconnect_delay

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._websocket = None

    def start(self) -> None:
        """Start the background WebSocket task."""
        if self._running:
            carb.log_warn("[IITP DT] WebSocket client is already running")
            return

        self._running = True
        self._task = omni.kit.async_engine.run_coroutine(
            self._connection_loop()
        )

        carb.log_info(f"[IITP DT] WebSocket client starting: {self.uri}")

    def stop(self) -> None:
        """Stop the WebSocket client and cancel its background task."""
        self._running = False

        if self._task is not None:
            try:
                self._task.cancel()
            except Exception as exc:
                carb.log_warn(
                    f"[IITP DT] Error cancelling WebSocket task: {exc}"
                )
            finally:
                self._task = None

        self._websocket = None
        carb.log_info("[IITP DT] WebSocket client stopped")

    async def _connection_loop(self) -> None:
        """Connect, receive messages, and reconnect after disconnects."""
        try:
            import websockets
        except ImportError:
            carb.log_error(
                "[IITP DT] Python package 'websockets' is not available "
                "inside the Kit Python environment. "
                "Install/package it before enabling the live connection."
            )
            self._running = False
            return

        while self._running:
            try:
                carb.log_info(
                    f"[IITP DT] Connecting to {self.uri}"
                )

                async with websockets.connect(self.uri) as websocket:
                    self._websocket = websocket

                    carb.log_info(
                        f"[IITP DT] Connected to {self.uri}"
                    )
                    print(f"[IITP DT] Connected to {self.uri}")

                    while self._running:
                        raw_message = await websocket.recv()

                        if raw_message is None:
                            break

                        self._handle_message(raw_message)

            except asyncio.CancelledError:
                break

            except Exception as exc:
                if self._running:
                    carb.log_warn(
                        f"[IITP DT] WebSocket connection error: {exc}"
                    )
                    print(
                        f"[IITP DT] Connection unavailable: {exc}"
                    )

                    await asyncio.sleep(self.reconnect_delay)

            finally:
                self._websocket = None

        carb.log_info("[IITP DT] WebSocket connection loop ended")

    def _handle_message(self, raw_message: Any) -> None:
        """Parse a received JSON message and forward it to the callback."""
        try:
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")

            message = json.loads(raw_message)

        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            carb.log_warn(
                f"[IITP DT] Invalid WebSocket JSON message: {exc}"
            )
            return

        if not isinstance(message, dict):
            carb.log_warn(
                "[IITP DT] Ignoring message because the JSON root "
                "is not an object."
            )
            return

        message_type = message.get("type", "unknown")

        carb.log_info(
            f"[IITP DT] Received message type: {message_type}"
        )

        if self.on_message is not None:
            try:
                self.on_message(message)
            except Exception as exc:
                carb.log_error(
                    f"[IITP DT] Telemetry callback failed: {exc}"
                )

    def send_json(self, message: dict[str, Any]) -> None:
        """Schedule a JSON message to be sent to the Python server."""
        if not self._running or self._websocket is None:
            carb.log_warn(
                "[IITP DT] Cannot send: WebSocket is not connected"
            )
            return

        omni.kit.async_engine.run_coroutine(
            self._send_json_async(message)
        )

    async def _send_json_async(self, message: dict[str, Any]) -> None:
        """Send a JSON object over the active WebSocket."""
        if self._websocket is None:
            return

        try:
            payload = json.dumps(message)
            await self._websocket.send(payload)

        except Exception as exc:
            carb.log_warn(
                f"[IITP DT] Failed to send WebSocket message: {exc}"
            )

    @property
    def is_connected(self) -> bool:
        """Return True when a WebSocket connection is active."""
        return self._websocket is not None
