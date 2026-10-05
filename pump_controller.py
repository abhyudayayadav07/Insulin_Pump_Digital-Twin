"""
IITP Digital Twin - Omniverse insulin-pump controller.

This module is the Omniverse-side controller for the visual insulin pump.

IMPORTANT:
    This controller does NOT directly decide whether insulin is medically
    safe to deliver. The external Python digital-twin/security/decision
    engine remains authoritative.

The controller has two responsibilities:

    1. Receive the latest pump telemetry and expose it to the 3D/UI layer.
    2. Send user-interface requests (for example REQUEST_BOLUS) back to
       the external digital-twin process through the WebSocket client.

Recommended architecture:

    Omniverse UI / 3D Pump
             |
             v
       PumpController
             |
       +-----+------+
       |            |
       v            v
   telemetry     commands
       |            |
       v            v
 TelemetryHandler  WebSocket
                      |
                      v
             Python Digital Twin
             Security + Decision
                      |
                      v
               authoritative
                  response

The pump model can therefore look interactive in Omniverse without making
the 3D scene itself the safety authority.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional

import carb


class PumpController:
    """
    Manage the Omniverse representation of the insulin pump.

    The class is intentionally independent of USD geometry. A later
    pump_visualizer.py module can subscribe to this controller and update:

        /World/.../InsulinPump/Pump_Screen
        /World/.../InsulinPump/Button_1
        /World/.../InsulinPump/Button_2
        /World/.../InsulinPump/Button_3

    The exact USD paths should be configured when the visualizer is created,
    rather than hard-coded here.
    """

    # Commands that the Omniverse UI is allowed to request.
    REQUEST_BOLUS = "REQUEST_BOLUS"
    REQUEST_BASAL_CHANGE = "REQUEST_BASAL_CHANGE"
    REQUEST_SUSPEND = "REQUEST_SUSPEND"
    REQUEST_RESUME = "REQUEST_RESUME"
    REQUEST_STATUS = "REQUEST_STATUS"

    # UI-level pump states.
    NORMAL = "NORMAL"
    MONITOR = "MONITOR"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"
    SAFE_MODE = "SAFE_MODE"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    UNKNOWN = "UNKNOWN"

    ALLOWED_COMMANDS = {
        REQUEST_BOLUS,
        REQUEST_BASAL_CHANGE,
        REQUEST_SUSPEND,
        REQUEST_RESUME,
        REQUEST_STATUS,
    }

    def __init__(
        self,
        websocket_client: Optional[Any] = None,
        telemetry_handler: Optional[Any] = None,
        on_state_change: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Args:
            websocket_client:
                Instance of DigitalTwinWebSocketClient. It must provide
                send_json(message).

            telemetry_handler:
                Instance of TelemetryHandler. If supplied, this controller
                can synchronize its state from the latest telemetry.

            on_state_change:
                Optional callback invoked whenever pump state changes.
        """
        self._websocket_client = websocket_client
        self._telemetry_handler = telemetry_handler
        self._on_state_change = on_state_change

        self._state: Dict[str, Any] = {
            "status": self.UNKNOWN,
            "insulin_rate": 0.0,
            "command": 0.0,
            "delivery": 0.0,
            "last_command": None,
            "last_command_value": None,
            "last_command_status": "NONE",
            "simulation_time": None,
            "timestamp": None,
        }

        carb.log_info("[IITP DT] Pump controller initialized")

    # ------------------------------------------------------------------
    # Connection / dependency management
    # ------------------------------------------------------------------

    def set_websocket_client(self, websocket_client: Any) -> None:
        """Attach or replace the WebSocket client."""
        self._websocket_client = websocket_client
        carb.log_info("[IITP DT] Pump WebSocket client attached")

    def set_telemetry_handler(self, telemetry_handler: Any) -> None:
        """Attach or replace the telemetry handler."""
        self._telemetry_handler = telemetry_handler
        carb.log_info("[IITP DT] Pump telemetry handler attached")

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def update_from_telemetry(
        self,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update pump state from a complete telemetry message.

        If telemetry is omitted, the latest message from TelemetryHandler
        is used.
        """
        if telemetry is None:
            if self._telemetry_handler is None:
                carb.log_warn(
                    "[IITP DT] Cannot update pump: no telemetry source"
                )
                return self.get_state()

            telemetry = self._telemetry_handler.get_latest()

        if not isinstance(telemetry, dict):
            carb.log_warn(
                "[IITP DT] Pump received invalid telemetry"
            )
            return self.get_state()

        pump = telemetry.get("pump", {})
        decision = telemetry.get("decision", {})

        if not isinstance(pump, dict):
            pump = {}

        if not isinstance(decision, dict):
            decision = {}

        self._state["timestamp"] = telemetry.get("timestamp")
        self._state["simulation_time"] = telemetry.get("simulation_time")

        self._state["insulin_rate"] = self._number(
            pump.get("insulin_rate"),
            default=0.0,
        )
        self._state["command"] = self._number(
            pump.get("command"),
            default=0.0,
        )
        self._state["delivery"] = self._number(
            pump.get("delivery"),
            default=0.0,
        )

        self._state["status"] = self._derive_status(
            pump_status=pump.get("status"),
            decision_action=decision.get("action"),
        )

        self._notify_state_change()

        return self.get_state()

    def handle_telemetry(
        self,
        telemetry: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Callback-compatible alias for update_from_telemetry().

        This can be registered directly with TelemetryHandler.
        """
        return self.update_from_telemetry(telemetry)

    # ------------------------------------------------------------------
    # Pump commands
    # ------------------------------------------------------------------

    def request_bolus(self, amount: float) -> bool:
        """
        Request a bolus from the authoritative digital-twin decision layer.

        This does NOT directly actuate insulin delivery.
        """
        amount = self._validate_non_negative_number(
            amount,
            field_name="bolus amount",
        )

        if amount is None:
            return False

        return self._send_command(
            command=self.REQUEST_BOLUS,
            value=amount,
        )

    def request_basal_change(self, rate: float) -> bool:
        """
        Request a basal-rate change.

        The external decision engine must approve/reject the request.
        """
        rate = self._validate_non_negative_number(
            rate,
            field_name="basal rate",
        )

        if rate is None:
            return False

        return self._send_command(
            command=self.REQUEST_BASAL_CHANGE,
            value=rate,
        )

    def request_suspend(self) -> bool:
        """Request pump suspension through the safety/decision layer."""
        return self._send_command(
            command=self.REQUEST_SUSPEND,
            value=None,
        )

    def request_resume(self) -> bool:
        """Request pump resume through the safety/decision layer."""
        return self._send_command(
            command=self.REQUEST_RESUME,
            value=None,
        )

    def request_status(self) -> bool:
        """Request an updated pump status from the external process."""
        return self._send_command(
            command=self.REQUEST_STATUS,
            value=None,
        )

    def _send_command(
        self,
        command: str,
        value: Optional[float],
    ) -> bool:
        """
        Send a UI command to the Python digital-twin process.

        Message schema:

        {
            "type": "command",
            "source": "insulin_pump",
            "command": "REQUEST_BOLUS",
            "value": 1.0
        }
        """
        if command not in self.ALLOWED_COMMANDS:
            carb.log_warn(
                f"[IITP DT] Refusing unknown pump command: {command}"
            )
            return False

        if self._websocket_client is None:
            carb.log_warn(
                "[IITP DT] Cannot send pump command: "
                "WebSocket client is not attached"
            )
            return False

        try:
            connected = bool(
                getattr(
                    self._websocket_client,
                    "is_connected",
                    False,
                )
            )
        except Exception:
            connected = False

        if not connected:
            carb.log_warn(
                "[IITP DT] Cannot send pump command: "
                "WebSocket is not connected"
            )
            return False

        message: Dict[str, Any] = {
            "type": "command",
            "source": "insulin_pump",
            "command": command,
            "value": value,
        }

        try:
            self._websocket_client.send_json(message)

            self._state["last_command"] = command
            self._state["last_command_value"] = value
            self._state["last_command_status"] = "SENT"

            self._notify_state_change()

            carb.log_info(
                "[IITP DT] Pump command sent: "
                f"{command}, value={value}"
            )

            return True

        except Exception as exc:
            self._state["last_command_status"] = "FAILED"

            carb.log_error(
                f"[IITP DT] Failed to send pump command: {exc}"
            )

            self._notify_state_change()
            return False

    # ------------------------------------------------------------------
    # UI interaction helpers
    # ------------------------------------------------------------------

    def handle_button_press(
        self,
        button_id: str,
        value: Optional[float] = None,
    ) -> bool:
        """
        Convert a 3D pump button press into a safe UI request.

        Suggested mapping:

            Button_1 -> status
            Button_2 -> bolus
            Button_3 -> suspend

        The exact mapping can be changed later by the UI layer.
        """
        button_id = str(button_id).upper()

        if button_id in {"BUTTON_1", "1"}:
            return self.request_status()

        if button_id in {"BUTTON_2", "2"}:
            if value is None:
                carb.log_warn(
                    "[IITP DT] Button 2 requires a bolus value"
                )
                return False

            return self.request_bolus(value)

        if button_id in {"BUTTON_3", "3"}:
            return self.request_suspend()

        carb.log_warn(
            f"[IITP DT] Unknown pump button: {button_id}"
        )
        return False

    def can_accept_user_request(self) -> bool:
        """
        Return whether the current pump state permits a UI request.

        This is only a UI gate. The external safety/decision engine remains
        authoritative and must independently validate every command.
        """
        return self._state["status"] not in {
            self.EMERGENCY_STOP,
            self.BLOCKED,
        }

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return a copy of the current pump state."""
        return deepcopy(self._state)

    @property
    def status(self) -> str:
        return str(self._state["status"])

    @property
    def insulin_rate(self) -> float:
        return float(self._state["insulin_rate"])

    @property
    def command(self) -> float:
        return float(self._state["command"])

    @property
    def delivery(self) -> float:
        return float(self._state["delivery"])

    @property
    def is_safe_mode(self) -> bool:
        return self.status == self.SAFE_MODE

    @property
    def is_emergency_stop(self) -> bool:
        return self.status == self.EMERGENCY_STOP

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _notify_state_change(self) -> None:
        if self._on_state_change is None:
            return

        try:
            self._on_state_change(self.get_state())
        except Exception as exc:
            carb.log_error(
                f"[IITP DT] Pump state callback failed: {exc}"
            )

    @staticmethod
    def _number(
        value: Any,
        default: float = 0.0,
    ) -> float:
        if value is None:
            return float(default)

        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @classmethod
    def _validate_non_negative_number(
        cls,
        value: Any,
        field_name: str,
    ) -> Optional[float]:
        try:
            value = float(value)
        except (TypeError, ValueError):
            carb.log_warn(
                f"[IITP DT] Invalid {field_name}: {value}"
            )
            return None

        if value < 0:
            carb.log_warn(
                f"[IITP DT] {field_name} cannot be negative: {value}"
            )
            return None

        return value

    @classmethod
    def _derive_status(
        cls,
        pump_status: Any,
        decision_action: Any,
    ) -> str:
        """
        Derive a display state from pump + decision telemetry.

        This is deliberately a presentation state, not a new safety policy.
        """
        pump_status = (
            str(pump_status).upper()
            if pump_status is not None
            else ""
        )

        decision_action = (
            str(decision_action).upper()
            if decision_action is not None
            else ""
        )

        # Preserve explicit high-priority states.
        if decision_action == "EMERGENCY_STOP":
            return cls.EMERGENCY_STOP

        if decision_action == "SAFE_MODE":
            return cls.SAFE_MODE

        if decision_action == "BLOCK_ML":
            return cls.BLOCKED

        if decision_action == "WARN":
            return cls.WARNING

        if decision_action == "MONITOR":
            return cls.MONITOR

        if pump_status in {
            cls.EMERGENCY_STOP,
            cls.SAFE_MODE,
            cls.BLOCKED,
            cls.WARNING,
            cls.MONITOR,
            cls.NORMAL,
        }:
            return pump_status

        if decision_action == "ALLOW":
            return cls.NORMAL

        return cls.UNKNOWN


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def create_pump_controller(
    websocket_client: Optional[Any] = None,
    telemetry_handler: Optional[Any] = None,
    on_state_change: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> PumpController:
    """Create a PumpController with the supplied dependencies."""
    return PumpController(
        websocket_client=websocket_client,
        telemetry_handler=telemetry_handler,
        on_state_change=on_state_change,
    )
