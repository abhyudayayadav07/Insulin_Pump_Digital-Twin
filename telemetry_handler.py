"""
IITP Digital Twin - Omniverse telemetry handler.

Receives parsed telemetry dictionaries from DigitalTwinWebSocketClient
and converts them into a stable, easy-to-consume runtime state.

The handler intentionally does NOT modify USD directly. Keeping telemetry
processing separate from USD visualization lets later modules such as:

    dt1_visualizer.py
    dt2_visualizer.py
    attack_visualizer.py
    safety_visualizer.py
    ui.py

consume the same state without coupling the WebSocket layer to the scene.

Expected message type:

{
    "type": "telemetry",
    "timestamp": "...",
    "simulation_time": 3600,
    "patient": {
        "glucose": 108.2,
        "cgm": 106.9,
        "meal": 0.0
    },
    "pump": {
        "insulin_rate": 1.20,
        "command": 1.20,
        "delivery": 1.20,
        "status": "NORMAL"
    },
    "dt1": {
        "ml_prediction": 101.4,
        "geco_prediction": 103.2,
        "fused_prediction": 102.3,
        "residual": -1.4,
        "cusum": 2.1,
        "status": "NORMAL"
    },
    "dt2": {
        "hazard": 0.12,
        "survival": 0.94,
        "event_probability": 0.06,
        "time_to_event": null,
        "risk_level": "NORMAL"
    },
    "security": {
        "attack_detected": false,
        "attack_type": null,
        "sensor_integrity": 0.99,
        "pump_integrity": 0.99
    },
    "decision": {
        "action": "ALLOW"
    }
}
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

import carb


class TelemetryHandler:
    """
    Process and store digital-twin telemetry received from WebSocket.

    Responsibilities:
        1. Validate the top-level telemetry message.
        2. Normalize missing sections with safe defaults.
        3. Store the latest complete telemetry state.
        4. Keep a bounded telemetry history.
        5. Notify registered callbacks.
        6. Provide convenient accessors for visualizer/UI modules.

    It does not:
        - run the digital twin
        - make safety decisions
        - actuate the insulin pump
        - directly edit USD
    """

    DEFAULT_HISTORY_SIZE = 300

    def __init__(
        self,
        history_size: int = DEFAULT_HISTORY_SIZE,
        on_telemetry: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.history_size = max(1, int(history_size))

        self._latest: Optional[Dict[str, Any]] = None
        self._history: List[Dict[str, Any]] = []
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []

        if on_telemetry is not None:
            self.add_callback(on_telemetry)

        carb.log_info("[IITP DT] Telemetry handler initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_message(self, message: Dict[str, Any]) -> bool:
        """
        Process one parsed WebSocket message.

        Returns:
            True  -> message was accepted as telemetry.
            False -> message was ignored or invalid.
        """
        if not isinstance(message, dict):
            carb.log_warn("[IITP DT] Telemetry message is not a dictionary")
            return False

        message_type = message.get("type", "unknown")

        if message_type != "telemetry":
            # Other message types may later include:
            # command, attack_event, simulation_control, heartbeat, etc.
            carb.log_info(
                f"[IITP DT] Ignoring non-telemetry message: {message_type}"
            )
            return False

        normalized = self._normalize_telemetry(message)

        if normalized is None:
            return False

        self._latest = normalized
        self._history.append(deepcopy(normalized))

        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size :]

        self._notify_callbacks(normalized)

        carb.log_info(
            "[IITP DT] Telemetry updated: "
            f"glucose={normalized['patient']['glucose']}, "
            f"action={normalized['decision']['action']}, "
            f"risk={normalized['dt2']['risk_level']}"
        )

        return True

    def add_callback(
        self,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Register a callback called after every accepted telemetry message."""
        if not callable(callback):
            raise TypeError("Telemetry callback must be callable")

        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(
        self,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Remove a previously registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Return a copy of the latest telemetry message."""
        if self._latest is None:
            return None

        return deepcopy(self._latest)

    def get_history(self) -> List[Dict[str, Any]]:
        """Return a copy of the bounded telemetry history."""
        return deepcopy(self._history)

    def clear(self) -> None:
        """Clear latest telemetry and history."""
        self._latest = None
        self._history.clear()
        carb.log_info("[IITP DT] Telemetry state cleared")

    @property
    def has_data(self) -> bool:
        """True when at least one valid telemetry message was received."""
        return self._latest is not None

    # ------------------------------------------------------------------
    # Convenient state accessors
    # ------------------------------------------------------------------

    @property
    def glucose(self) -> Optional[float]:
        return self._get_float("patient", "glucose")

    @property
    def cgm(self) -> Optional[float]:
        return self._get_float("patient", "cgm")

    @property
    def meal(self) -> Optional[float]:
        return self._get_float("patient", "meal")

    @property
    def insulin_rate(self) -> Optional[float]:
        return self._get_float("pump", "insulin_rate")

    @property
    def pump_status(self) -> str:
        return self._get_str("pump", "status", "UNKNOWN")

    @property
    def dt1_prediction(self) -> Optional[float]:
        return self._get_float("dt1", "fused_prediction")

    @property
    def dt1_ml_prediction(self) -> Optional[float]:
        return self._get_float("dt1", "ml_prediction")

    @property
    def dt1_geco_prediction(self) -> Optional[float]:
        return self._get_float("dt1", "geco_prediction")

    @property
    def dt1_residual(self) -> Optional[float]:
        return self._get_float("dt1", "residual")

    @property
    def cusum(self) -> Optional[float]:
        return self._get_float("dt1", "cusum")

    @property
    def dt1_status(self) -> str:
        return self._get_str("dt1", "status", "UNKNOWN")

    @property
    def hazard(self) -> Optional[float]:
        return self._get_float("dt2", "hazard")

    @property
    def survival(self) -> Optional[float]:
        return self._get_float("dt2", "survival")

    @property
    def event_probability(self) -> Optional[float]:
        return self._get_float("dt2", "event_probability")

    @property
    def time_to_event(self) -> Optional[float]:
        return self._get_float("dt2", "time_to_event")

    @property
    def risk_level(self) -> str:
        return self._get_str("dt2", "risk_level", "UNKNOWN")

    @property
    def attack_detected(self) -> bool:
        if self._latest is None:
            return False

        return bool(
            self._latest.get("security", {}).get("attack_detected", False)
        )

    @property
    def attack_type(self) -> Optional[str]:
        if self._latest is None:
            return None

        value = self._latest.get("security", {}).get("attack_type")
        return str(value) if value is not None else None

    @property
    def sensor_integrity(self) -> Optional[float]:
        return self._get_float("security", "sensor_integrity")

    @property
    def pump_integrity(self) -> Optional[float]:
        return self._get_float("security", "pump_integrity")

    @property
    def decision_action(self) -> str:
        return self._get_str("decision", "action", "UNKNOWN")

    # ------------------------------------------------------------------
    # Normalization / validation
    # ------------------------------------------------------------------

    def _normalize_telemetry(
        self,
        message: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Normalize telemetry into a stable schema.

        Missing optional fields receive safe neutral defaults. Numerical
        values are converted to float where possible.
        """
        try:
            normalized = deepcopy(message)

            normalized["type"] = "telemetry"

            normalized.setdefault("timestamp", None)
            normalized.setdefault("simulation_time", None)

            normalized["patient"] = self._section(
                normalized,
                "patient",
                {
                    "glucose": None,
                    "cgm": None,
                    "meal": 0.0,
                },
            )

            normalized["pump"] = self._section(
                normalized,
                "pump",
                {
                    "insulin_rate": 0.0,
                    "command": 0.0,
                    "delivery": 0.0,
                    "status": "UNKNOWN",
                },
            )

            normalized["dt1"] = self._section(
                normalized,
                "dt1",
                {
                    "ml_prediction": None,
                    "geco_prediction": None,
                    "fused_prediction": None,
                    "residual": None,
                    "cusum": None,
                    "status": "UNKNOWN",
                },
            )

            normalized["dt2"] = self._section(
                normalized,
                "dt2",
                {
                    "hazard": None,
                    "survival": None,
                    "event_probability": None,
                    "time_to_event": None,
                    "risk_level": "UNKNOWN",
                },
            )

            normalized["security"] = self._section(
                normalized,
                "security",
                {
                    "attack_detected": False,
                    "attack_type": None,
                    "sensor_integrity": None,
                    "pump_integrity": None,
                },
            )

            normalized["decision"] = self._section(
                normalized,
                "decision",
                {
                    "action": "UNKNOWN",
                },
            )

            # Normalize common numeric values.
            for section_name, fields in {
                "patient": ["glucose", "cgm", "meal"],
                "pump": ["insulin_rate", "command", "delivery"],
                "dt1": [
                    "ml_prediction",
                    "geco_prediction",
                    "fused_prediction",
                    "residual",
                    "cusum",
                ],
                "dt2": [
                    "hazard",
                    "survival",
                    "event_probability",
                    "time_to_event",
                ],
                "security": [
                    "sensor_integrity",
                    "pump_integrity",
                ],
            }.items():
                for field in fields:
                    normalized[section_name][field] = self._to_float_or_none(
                        normalized[section_name].get(field)
                    )

            normalized["security"]["attack_detected"] = bool(
                normalized["security"].get("attack_detected", False)
            )

            # Normalize strings.
            normalized["pump"]["status"] = str(
                normalized["pump"].get("status", "UNKNOWN")
            ).upper()

            normalized["dt1"]["status"] = str(
                normalized["dt1"].get("status", "UNKNOWN")
            ).upper()

            normalized["dt2"]["risk_level"] = str(
                normalized["dt2"].get("risk_level", "UNKNOWN")
            ).upper()

            normalized["decision"]["action"] = str(
                normalized["decision"].get("action", "UNKNOWN")
            ).upper()

            return normalized

        except Exception as exc:
            carb.log_error(
                f"[IITP DT] Failed to normalize telemetry: {exc}"
            )
            return None

    @staticmethod
    def _section(
        message: Dict[str, Any],
        name: str,
        defaults: Dict[str, Any],
    ) -> Dict[str, Any]:
        section = message.get(name)

        if not isinstance(section, dict):
            section = {}

        result = dict(defaults)
        result.update(section)

        return result

    # ------------------------------------------------------------------
    # Callback / helper functions
    # ------------------------------------------------------------------

    def _notify_callbacks(self, telemetry: Dict[str, Any]) -> None:
        for callback in list(self._callbacks):
            try:
                callback(deepcopy(telemetry))
            except Exception as exc:
                carb.log_error(
                    f"[IITP DT] Telemetry callback failed: {exc}"
                )

    def _get_float(
        self,
        section: str,
        field: str,
    ) -> Optional[float]:
        if self._latest is None:
            return None

        value = self._latest.get(section, {}).get(field)
        return self._to_float_or_none(value)

    def _get_str(
        self,
        section: str,
        field: str,
        default: str,
    ) -> str:
        if self._latest is None:
            return default

        value = self._latest.get(section, {}).get(field, default)

        if value is None:
            return default

        return str(value)

    @staticmethod
    def _to_float_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None


# ----------------------------------------------------------------------
# Optional functional helper
# ----------------------------------------------------------------------

def create_telemetry_handler(
    history_size: int = TelemetryHandler.DEFAULT_HISTORY_SIZE,
    on_telemetry: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> TelemetryHandler:
    """Factory function used by extension.py."""
    return TelemetryHandler(
        history_size=history_size,
        on_telemetry=on_telemetry,
    )
