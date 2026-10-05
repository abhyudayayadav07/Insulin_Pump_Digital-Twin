import carb
import omni.ext


class IITPDigitalTwinWebSocketExtension(omni.ext.IExt):
    """Omniverse-side entry point for the IITP Digital Twin WebSocket bridge."""

    def __init__(self):
        super().__init__()
        self._ext_id = None
        self._websocket_client = None

    def on_startup(self, ext_id: str):
        """Called automatically when the extension is enabled."""
        self._ext_id = ext_id

        carb.log_info(
            "[IITP Digital Twin] WebSocket Bridge started "
            f"(extension_id={ext_id})"
        )
        print("[IITP DT] WebSocket Bridge started")

        # The WebSocket client will be initialized in the next step.
        #
        # from .websocket_client import WebSocketClient
        # self._websocket_client = WebSocketClient(
        #     uri="ws://127.0.0.1:8765"
        # )
        # self._websocket_client.start()

    def on_shutdown(self):
        """Called automatically when the extension is disabled."""
        carb.log_info("[IITP Digital Twin] WebSocket Bridge shutting down")

        if self._websocket_client is not None:
            try:
                self._websocket_client.stop()
            except Exception as exc:
                carb.log_error(
                    f"[IITP Digital Twin] Error stopping WebSocket client: {exc}"
                )
            finally:
                self._websocket_client = None

        self._ext_id = None
        print("[IITP DT] WebSocket Bridge stopped")
