"""
IITP Digital Twin - Omniverse CGM controller.

This module manages the Omniverse-side representation of the continuous
glucose monitor (CGM).

The controller receives glucose/CGM telemetry from TelemetryHandler and
provides a stable state for a later cgm_visualizer.py module.

It does NOT:
    - calculate clinical glucose values
    - replace the physiological simulator
    - make safety decisions
    - directly modify USD geometry
    - actuate any medical device

The external Python digital-twin system remains the source of truth.

Recommended flow:

    SimGlucose
        |
        v
    Python Digital Twin
        |
        v
      JSON
        |
        v
    WebSocket
        |
        v
    TelemetryHandler
        |
        v
    CGMController
        |
        +--> cgm_visualizer.py
        +--> UI / monitor
        +--> future USD sensor animation

Expected telemetry section:

    "patient": {
        "glucose": 108.2,
        "cgm": 106.9,
        "meal": 0.0
    }

Optional future CGM fields can be added without changing the public API.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import carb


class CGMController:
    """
    Manage live CGM state for the Omniverse visualization layer.

    The controller keeps a bounded history so the future visualizer can
    render a live glucose trend without querying the WebSocket directly.
    """

    NORMAL = "NORMAL"
    LOW = "LOW"
    HIGH = "HIGH"
    CRITICAL_LOW = "CRITICAL_LOW"
    CRITICAL_HIGH = "CRITICAL_HIGH"
    UNKNOWN = "UNKNOWN"

    # Display thresholds only.
    #
    # These are visualization categories, not clinical treatment rules.
    LOW_THRESHOLD = 70.0
    HIGH_THRESHOLD = 180.0
    CRITICAL_LOW_THRESHOLD = 54.0
    CRITICAL_HIGH_THRESHOLD = 250.0

    DEFAULT_HISTORY_SIZE = 288  # 24 h at 5-minute sampling

    def __init__(
        self,
        telemetry_handler: Optional[Any] = None,
        history_size: int = DEFAULT_HISTORY_SIZE,
        on_state_change: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Args:
            telemetry_handler:
                Existing TelemetryHandler instance.

            history_size:
                Maximum number of CGM samples retained.

            on_state_change:
                Optional callback called after every accepted CGM update.
        """
        self._telemetry_handler = telemetry_handler
        self._history_size = max(1, int(history_size))
        self._on_state_change = on_state_change

        self._history: Deque[Dict[str, Any]] = deque(
            maxlen=self._history_size
        )

        self._state: Dict[str, Any] = {
            "timestamp": None,
            "simulation_time": None,
            "glucose": None,
            "cgm": None,
            "delta": None,
            "rate_of_change": None,
            "status": self.UNKNOWN,
            "signal_quality": None,
            "sensor_integrity": None,
        }

        self._previous_cgm: Optional[float] = None
        self._previous_simulation_time: Optional[float] = None

        carb.log_info("[IITP DT] CGM controller initialized")

    # ------------------------------------------------------------------
    # Dependency management
    # ------------------------------------------------------------------

    def set_telemetry_handler(self, telemetry_handler: Any) -> None:
        """Attach or replace the TelemetryHandler."""
        self._telemetry_handler = telemetry_handler
        carb.log_info("[IITP DT] CGM telemetry handler attached")

    # ------------------------------------------------------------------
    # Telemetry processing
    # ------------------------------------------------------------------

    def update_from_telemetry(
        self,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update CGM state from a complete telemetry message.

        If telemetry is omitted, the latest message from TelemetryHandler
        is used.
        """
        if telemetry is None:
            if self._telemetry_handler is None:
                carb.log_warn(
                    "[IITP DT] Cannot update CGM: no telemetry source"
                )
                return self.get_state()

            telemetry = self._telemetry_handler.get_latest()

        if not isinstance(telemetry, dict):
            carb.log_warn(
                "[IITP DT] CGM received invalid telemetry"
            )
            return self.get_state()

        patient = telemetry.get("patient", {})
        security = telemetry.get("security", {})

        if not isinstance(patient, dict):
            patient = {}

        if not isinstance(security, dict):
            security = {}

        glucose = self._to_float_or_none(patient.get("glucose"))
        cgm = self._to_float_or_none(patient.get("cgm"))

        # If only one of glucose/CGM is present, preserve useful display
        # behavior without inventing a second physiological measurement.
        display_cgm = cgm if cgm is not None else glucose

        simulation_time = self._to_float_or_none(
            telemetry.get("simulation_time")
        )

        delta = self._calculate_delta(display_cgm)
        rate_of_change = self._calculate_rate(
            display_cgm,
            simulation_time,
        )

        self._state.update(
            {
                "timestamp": telemetry.get("timestamp"),
                "simulation_time": simulation_time,
                "glucose": glucose,
                "cgm": display_cgm,
                "delta": delta,
                "rate_of_change": rate_of_change,
                "status": self.classify_glucose(display_cgm),
                "signal_quality": self._extract_signal_quality(
                    telemetry
                ),
                "sensor_integrity": self._to_float_or_none(
                    security.get("sensor_integrity")
                ),
            }
        )

        if display_cgm is not None:
            sample = {
                "timestamp": telemetry.get("timestamp"),
                "simulation_time": simulation_time,
                "cgm": display_cgm,
                "glucose": glucose,
                "delta": delta,
                "rate_of_change": rate_of_change,
                "status": self._state["status"],
                "sensor_integrity": self._state["sensor_integrity"],
            }

            self._history.append(sample)

        self._previous_cgm = display_cgm

        if simulation_time is not None:
            self._previous_simulation_time = simulation_time

        self._notify_state_change()

        carb.log_info(
            "[IITP DT] CGM updated: "
            f"cgm={display_cgm}, "
            f"status={self._state['status']}, "
            f"rate={rate_of_change}"
        )

        return self.get_state()

    def handle_telemetry(
        self,
        telemetry: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Callback-compatible alias for update_from_telemetry().
        """
        return self.update_from_telemetry(telemetry)

    # ------------------------------------------------------------------
    # Glucose classification
    # ------------------------------------------------------------------

    @classmethod
    def classify_glucose(
        cls,
        cgm: Optional[float],
    ) -> str:
        """
        Classify CGM value for visualization.

        These categories are display states and should not be interpreted
        as a treatment recommendation.
        """
        if cgm is None:
            return cls.UNKNOWN

        if cgm < cls.CRITICAL_LOW_THRESHOLD:
            return cls.CRITICAL_LOW

        if cgm < cls.LOW_THRESHOLD:
            return cls.LOW

        if cgm > cls.CRITICAL_HIGH_THRESHOLD:
            return cls.CRITICAL_HIGH

        if cgm > cls.HIGH_THRESHOLD:
            return cls.HIGH

        return cls.NORMAL

    # ------------------------------------------------------------------
    # Trend calculations
    # ------------------------------------------------------------------

    def get_trend(
        self,
        samples: int = 6,
    ) -> List[Dict[str, Any]]:
        """Return the latest CGM samples in chronological order."""
        samples = max(1, int(samples))
        return list(self._history)[-samples:]

    def get_glucose_series(
        self,
        samples: int = 72,
    ) -> List[Tuple[Optional[float], Optional[float]]]:
        """
        Return (simulation_time, cgm) pairs for plotting.

        This is useful for a future live trend graph.
        """
        samples = max(1, int(samples))

        return [
            (
                item.get("simulation_time"),
                item.get("cgm"),
            )
            for item in list(self._history)[-samples:]
        ]

    def get_rate_description(self) -> str:
        """Return a human-readable CGM trend direction."""
        rate = self._state.get("rate_of_change")

        if rate is None:
            return "UNKNOWN"

        if rate > 1.0:
            return "RISING_FAST"

        if rate > 0.1:
            return "RISING"

        if rate < -1.0:
            return "FALLING_FAST"

        if rate < -0.1:
            return "FALLING"

        return "STABLE"

    def get_arrow(self) -> str:
        """
        Return a simple trend arrow for a future pump/monitor display.
        """
        trend = self.get_rate_description()

        return {
            "RISING_FAST": "↑↑",
            "RISING": "↑",
            "STABLE": "→",
            "FALLING": "↓",
            "FALLING_FAST": "↓↓",
            "UNKNOWN": "—",
        }.get(trend, "—")

    # ------------------------------------------------------------------
    # Sensor integrity
    # ------------------------------------------------------------------

    def is_sensor_integrity_degraded(
        self,
        threshold: float = 0.80,
    ) -> bool:
        """
        Return True when telemetry reports reduced sensor integrity.

        The threshold is a visualization/integration threshold. The
        security engine remains responsible for authoritative decisions.
        """
        integrity = self._state.get("sensor_integrity")

        if integrity is None:
            return False

        return integrity < float(threshold)

    def get_sensor_state(self) -> str:
        """Return a simple visualization state for sensor integrity."""
        integrity = self._state.get("sensor_integrity")

        if integrity is None:
            return "UNKNOWN"

        if integrity < 0.50:
            return "COMPROMISED"

        if integrity < 0.80:
            return "DEGRADED"

        return "HEALTHY"

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return a copy of the current CGM state."""
        state = deepcopy(self._state)

        state["trend"] = self.get_rate_description()
        state["trend_arrow"] = self.get_arrow()
        state["sensor_state"] = self.get_sensor_state()

        return state

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the complete bounded CGM history."""
        return deepcopy(list(self._history))

    def clear_history(self) -> None:
        """Clear historical samples and reset trend state."""
        self._history.clear()
        self._previous_cgm = None
        self._previous_simulation_time = None

        carb.log_info("[IITP DT] CGM history cleared")

    @property
    def cgm(self) -> Optional[float]:
        return self._state["cgm"]

    @property
    def glucose(self) -> Optional[float]:
        return self._state["glucose"]

    @property
    def delta(self) -> Optional[float]:
        return self._state["delta"]

    @property
    def rate_of_change(self) -> Optional[float]:
        return self._state["rate_of_change"]

    @property
    def status(self) -> str:
        return str(self._state["status"])

    @property
    def sensor_integrity(self) -> Optional[float]:
        return self._state["sensor_integrity"]

    @property
    def has_data(self) -> bool:
        return self._state["cgm"] is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_delta(
        self,
        current_cgm: Optional[float],
    ) -> Optional[float]:
        if current_cgm is None or self._previous_cgm is None:
            return None

        return current_cgm - self._previous_cgm

    def _calculate_rate(
        self,
        current_cgm: Optional[float],
        simulation_time: Optional[float],
    ) -> Optional[float]:
        """
        Calculate CGM change per minute using simulation time.

        simulation_time is expected to be seconds, matching the common
        digital-twin telemetry convention. If no simulation time is
        available, the method returns None rather than guessing a sampling
        interval.
        """
        if current_cgm is None:
            return None

        if self._previous_cgm is None:
            return None

        if simulation_time is None:
            return None

        if self._previous_simulation_time is None:
            return None

        dt_seconds = simulation_time - self._previous_simulation_time

        if dt_seconds <= 0:
            return None

        return (
            (current_cgm - self._previous_cgm)
            / (dt_seconds / 60.0)
        )

    @staticmethod
    def _extract_signal_quality(
        telemetry: Dict[str, Any],
    ) -> Optional[float]:
        """
        Read an optional signal-quality field.

        The current telemetry schema does not require signal_quality.
        Future simulator/sensor models may provide it under:

            patient.signal_quality
            cgm.signal_quality
            security.sensor_integrity

        Sensor integrity is handled separately.
        """
        patient = telemetry.get("patient", {})
        cgm_section = telemetry.get("cgm", {})

        candidates = []

        if isinstance(patient, dict):
            candidates.append(patient.get("signal_quality"))

        if isinstance(cgm_section, dict):
            candidates.append(cgm_section.get("signal_quality"))

        for value in candidates:
            parsed = CGMController._to_float_or_none(value)

            if parsed is not None:
                return parsed

        return None

    def _notify_state_change(self) -> None:
        if self._on_state_change is None:
            return

        try:
            self._on_state_change(self.get_state())
        except Exception as exc:
            carb.log_error(
                f"[IITP DT] CGM state callback failed: {exc}"
            )

    @staticmethod
    def _to_float_or_none(
        value: Any,
    ) -> Optional[float]:
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def create_cgm_controller(
    telemetry_handler: Optional[Any] = None,
    history_size: int = CGMController.DEFAULT_HISTORY_SIZE,
    on_state_change: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> CGMController:
    """Create a CGMController."""
    return CGMController(
        telemetry_handler=telemetry_handler,
        history_size=history_size,
        on_state_change=on_state_change,
    )
