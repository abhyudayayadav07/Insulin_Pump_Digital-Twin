"""
IITP Digital Twin - Omniverse UI.

This module provides the operator/debug UI for the insulin-pump digital
twin. It is deliberately separate from the 3D USD visualizers.

The UI displays:
    - CGM / glucose
    - CGM trend
    - insulin delivery
    - DT1 prediction and anomaly evidence
    - DT2 hazard / survival / time-to-event
    - security / sensor integrity
    - decision-engine action
    - WebSocket connection state

It also exposes request buttons for:
    - status
    - bolus
    - suspend
    - resume

IMPORTANT:
    These buttons send REQUEST_* messages through PumpController. They do
    not directly actuate insulin delivery. The external Python decision and
    safety layers remain authoritative.

Omni.UI is used for the window/layout because it provides Window, VStack,
HStack, Label, Button, and value widgets for Kit extensions.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import carb
import omni.ui as ui


class DigitalTwinUI:
    """
    Operator/debug window for the IITP insulin-pump digital twin.

    Dependencies are injected so the UI does not own the WebSocket or
    simulation lifecycle.
    """

    WINDOW_TITLE = "IITP Digital Twin Monitor"

    def __init__(
        self,
        telemetry_handler: Optional[Any] = None,
        pump_controller: Optional[Any] = None,
        cgm_controller: Optional[Any] = None,
        websocket_client: Optional[Any] = None,
        on_bolus_request: Optional[Callable[[float], bool]] = None,
    ):
        self._telemetry_handler = telemetry_handler
        self._pump_controller = pump_controller
        self._cgm_controller = cgm_controller
        self._websocket_client = websocket_client
        self._on_bolus_request = on_bolus_request

        self._window: Optional[ui.Window] = None

        # Labels updated by refresh().
        self._connection_label = None
        self._simulation_time_label = None

        self._glucose_label = None
        self._cgm_label = None
        self._trend_label = None
        self._sensor_integrity_label = None

        self._pump_status_label = None
        self._insulin_rate_label = None
        self._delivery_label = None

        self._dt1_prediction_label = None
        self._dt1_residual_label = None
        self._cusum_label = None
        self._dt1_status_label = None

        self._hazard_label = None
        self._survival_label = None
        self._event_probability_label = None
        self._time_to_event_label = None
        self._risk_label = None

        self._attack_label = None
        self._decision_label = None

        self._bolus_field = None
        self._command_status_label = None

        carb.log_info("[IITP DT] UI controller initialized")

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Create and show the operator window."""
        if self._window is not None:
            self._window.visible = True
            self.refresh()
            return

        self._window = ui.Window(
            self.WINDOW_TITLE,
            width=430,
            height=760,
        )

        # Lazy build follows the recommended Kit UI pattern.
        self._window.frame.set_build_fn(self._build_ui)

        carb.log_info("[IITP DT] UI window shown")

    def hide(self) -> None:
        """Hide the operator window without destroying it."""
        if self._window is not None:
            self._window.visible = False

    def destroy(self) -> None:
        """Destroy the UI window."""
        if self._window is not None:
            try:
                self._window.destroy()
            except Exception as exc:
                carb.log_warn(
                    f"[IITP DT] UI window destroy failed: {exc}"
                )

            self._window = None

    @property
    def visible(self) -> bool:
        return bool(
            self._window is not None
            and self._window.visible
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the complete monitoring window."""
        with self._window.frame:
            with ui.VStack(
                spacing=8,
                style={
                    "background_color": 0xFF151515,
                },
            ):
                self._build_header()

                with ui.ScrollingFrame(
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                ):
                    with ui.VStack(
                        spacing=8,
                        height=0,
                    ):
                        self._build_patient_section()
                        self._build_pump_section()
                        self._build_dt1_section()
                        self._build_dt2_section()
                        self._build_security_section()
                        self._build_command_section()

        self.refresh()

    def _build_header(self) -> None:
        with ui.VStack(spacing=4):
            ui.Label(
                "IITP INSULIN-PUMP DIGITAL TWIN",
                height=28,
                style={"font_size": 18},
            )

            with ui.HStack(height=24):
                ui.Label("Connection:", width=90)
                self._connection_label = ui.Label("DISCONNECTED")
                self._simulation_time_label = ui.Label(
                    "Sim: --",
                    width=110,
                )

    def _build_patient_section(self) -> None:
        self._section_title("PATIENT / CGM")

        with ui.VStack(spacing=3):
            self._glucose_label = self._value_row(
                "Physiological glucose",
                "-- mg/dL",
            )
            self._cgm_label = self._value_row(
                "CGM",
                "-- mg/dL",
            )
            self._trend_label = self._value_row(
                "Trend",
                "--",
            )
            self._sensor_integrity_label = self._value_row(
                "Sensor integrity",
                "--",
            )

    def _build_pump_section(self) -> None:
        self._section_title("INSULIN PUMP")

        with ui.VStack(spacing=3):
            self._pump_status_label = self._value_row(
                "Status",
                "UNKNOWN",
            )
            self._insulin_rate_label = self._value_row(
                "Insulin rate",
                "-- U/h",
            )
            self._delivery_label = self._value_row(
                "Delivery",
                "-- U/h",
            )

    def _build_dt1_section(self) -> None:
        self._section_title("DT1 — PREDICTIVE DIGITAL TWIN")

        with ui.VStack(spacing=3):
            self._dt1_prediction_label = self._value_row(
                "Fused prediction",
                "-- mg/dL",
            )
            self._dt1_residual_label = self._value_row(
                "Residual",
                "-- mg/dL",
            )
            self._cusum_label = self._value_row(
                "CUSUM",
                "--",
            )
            self._dt1_status_label = self._value_row(
                "Status",
                "UNKNOWN",
            )

    def _build_dt2_section(self) -> None:
        self._section_title("DT2 — HAZARD / SURVIVAL")

        with ui.VStack(spacing=3):
            self._hazard_label = self._value_row(
                "Hazard",
                "--",
            )
            self._survival_label = self._value_row(
                "Survival",
                "--",
            )
            self._event_probability_label = self._value_row(
                "Event probability",
                "--",
            )
            self._time_to_event_label = self._value_row(
                "Time to event",
                "-- min",
            )
            self._risk_label = self._value_row(
                "Risk level",
                "UNKNOWN",
            )

    def _build_security_section(self) -> None:
        self._section_title("SECURITY / DECISION")

        with ui.VStack(spacing=3):
            self._attack_label = self._value_row(
                "Attack",
                "NONE",
            )
            self._decision_label = self._value_row(
                "Decision",
                "UNKNOWN",
            )

    def _build_command_section(self) -> None:
        self._section_title("PUMP COMMAND REQUESTS")

        with ui.VStack(spacing=5):
            with ui.HStack(height=30):
                ui.Label("Bolus (U)", width=100)

                self._bolus_field = ui.FloatField(
                    width=120,
                    min=0.0,
                )

                ui.Button(
                    "REQUEST BOLUS",
                    clicked_fn=self._on_request_bolus,
                    width=150,
                )

            with ui.HStack(height=30):
                ui.Button(
                    "STATUS",
                    clicked_fn=self._on_request_status,
                    width=100,
                )

                ui.Button(
                    "SUSPEND",
                    clicked_fn=self._on_request_suspend,
                    width=100,
                )

                ui.Button(
                    "RESUME",
                    clicked_fn=self._on_request_resume,
                    width=100,
                )

            self._command_status_label = ui.Label(
                "Last command: NONE",
                height=24,
            )

    def _section_title(self, title: str) -> None:
        ui.Label(
            title,
            height=28,
            style={
                "font_size": 14,
            },
        )

    @staticmethod
    def _value_row(
        name: str,
        initial_value: str,
    ):
        with ui.HStack(height=22):
            ui.Label(name, width=175)
            return ui.Label(initial_value)

    # ------------------------------------------------------------------
    # Telemetry refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """
        Refresh all displayed values from the latest telemetry.

        This method is intentionally explicit rather than creating a UI
        polling thread. The extension can call it from its normal Kit
        update/callback path.
        """
        telemetry = self._get_latest_telemetry()

        if telemetry is None:
            self._set_text(self._connection_label, "NO DATA")
            return

        self._refresh_connection()
        self._refresh_patient(telemetry)
        self._refresh_pump(telemetry)
        self._refresh_dt1(telemetry)
        self._refresh_dt2(telemetry)
        self._refresh_security(telemetry)

    def update_from_telemetry(
        self,
        telemetry: Dict[str, Any],
    ) -> None:
        """
        Callback-compatible entry point.

        Register this method with TelemetryHandler if immediate UI updates
        are desired.
        """
        if self._telemetry_handler is not None:
            # The handler already stores the message. We simply refresh.
            pass

        self.refresh()

    def _get_latest_telemetry(self) -> Optional[Dict[str, Any]]:
        if self._telemetry_handler is None:
            return None

        try:
            return self._telemetry_handler.get_latest()
        except Exception as exc:
            carb.log_warn(
                f"[IITP DT] UI failed to read telemetry: {exc}"
            )
            return None

    def _refresh_connection(self) -> None:
        connected = False

        if self._websocket_client is not None:
            try:
                connected = bool(
                    self._websocket_client.is_connected
                )
            except Exception:
                connected = False

        self._set_text(
            self._connection_label,
            "CONNECTED" if connected else "DISCONNECTED",
        )

    def _refresh_patient(
        self,
        telemetry: Dict[str, Any],
    ) -> None:
        patient = telemetry.get("patient", {})
        if not isinstance(patient, dict):
            patient = {}

        glucose = self._number(patient.get("glucose"))
        cgm = self._number(patient.get("cgm"))

        if cgm is None:
            cgm = glucose

        self._set_text(
            self._glucose_label,
            self._format_glucose(glucose),
        )

        self._set_text(
            self._cgm_label,
            self._format_glucose(cgm),
        )

        if self._cgm_controller is not None:
            try:
                trend = self._cgm_controller.get_arrow()
                rate = self._cgm_controller.rate_of_change

                if rate is None:
                    trend_text = str(trend)
                else:
                    trend_text = (
                        f"{trend}  {rate:+.2f} mg/dL/min"
                    )

                self._set_text(
                    self._trend_label,
                    trend_text,
                )

                integrity = (
                    self._cgm_controller.sensor_integrity
                )

                self._set_text(
                    self._sensor_integrity_label,
                    self._format_ratio(integrity),
                )

            except Exception as exc:
                carb.log_warn(
                    f"[IITP DT] CGM UI refresh failed: {exc}"
                )
        else:
            security = telemetry.get("security", {})
            if not isinstance(security, dict):
                security = {}

            self._set_text(
                self._sensor_integrity_label,
                self._format_ratio(
                    self._number(
                        security.get("sensor_integrity")
                    )
                ),
            )

        self._set_text(
            self._simulation_time_label,
            self._format_simulation_time(
                telemetry.get("simulation_time")
            ),
        )

    def _refresh_pump(
        self,
        telemetry: Dict[str, Any],
    ) -> None:
        pump = telemetry.get("pump", {})
        if not isinstance(pump, dict):
            pump = {}

        self._set_text(
            self._pump_status_label,
            str(
                pump.get("status", "UNKNOWN")
            ).upper(),
        )

        self._set_text(
            self._insulin_rate_label,
            self._format_rate(
                self._number(
                    pump.get("insulin_rate")
                )
            ),
        )

        self._set_text(
            self._delivery_label,
            self._format_rate(
                self._number(
                    pump.get("delivery")
                )
            ),
        )

    def _refresh_dt1(
        self,
        telemetry: Dict[str, Any],
    ) -> None:
        dt1 = telemetry.get("dt1", {})
        if not isinstance(dt1, dict):
            dt1 = {}

        self._set_text(
            self._dt1_prediction_label,
            self._format_glucose(
                self._number(
                    dt1.get("fused_prediction")
                )
            ),
        )

        self._set_text(
            self._dt1_residual_label,
            self._format_number_with_unit(
                self._number(
                    dt1.get("residual")
                ),
                "mg/dL",
            ),
        )

        self._set_text(
            self._cusum_label,
            self._format_number(
                self._number(
                    dt1.get("cusum")
                )
            ),
        )

        self._set_text(
            self._dt1_status_label,
            str(
                dt1.get("status", "UNKNOWN")
            ).upper(),
        )

    def _refresh_dt2(
        self,
        telemetry: Dict[str, Any],
    ) -> None:
        dt2 = telemetry.get("dt2", {})
        if not isinstance(dt2, dict):
            dt2 = {}

        hazard = self._number(dt2.get("hazard"))
        survival = self._number(dt2.get("survival"))
        event_probability = self._number(
            dt2.get("event_probability")
        )
        time_to_event = self._number(
            dt2.get("time_to_event")
        )

        self._set_text(
            self._hazard_label,
            self._format_number(hazard),
        )

        self._set_text(
            self._survival_label,
            self._format_ratio(survival),
        )

        self._set_text(
            self._event_probability_label,
            self._format_ratio(event_probability),
        )

        if time_to_event is None:
            tte_text = "NONE"
        else:
            tte_text = f"{time_to_event:.1f} min"

        self._set_text(
            self._time_to_event_label,
            tte_text,
        )

        self._set_text(
            self._risk_label,
            str(
                dt2.get("risk_level", "UNKNOWN")
            ).upper(),
        )

    def _refresh_security(
        self,
        telemetry: Dict[str, Any],
    ) -> None:
        security = telemetry.get("security", {})
        decision = telemetry.get("decision", {})

        if not isinstance(security, dict):
            security = {}

        if not isinstance(decision, dict):
            decision = {}

        attack_detected = bool(
            security.get("attack_detected", False)
        )

        attack_type = security.get("attack_type")

        if attack_detected:
            attack_text = (
                f"DETECTED"
                f" ({attack_type})"
                if attack_type
                else "DETECTED"
            )
        else:
            attack_text = "NONE"

        self._set_text(
            self._attack_label,
            attack_text,
        )

        self._set_text(
            self._decision_label,
            str(
                decision.get("action", "UNKNOWN")
            ).upper(),
        )

    # ------------------------------------------------------------------
    # Pump command callbacks
    # ------------------------------------------------------------------

    def _on_request_bolus(self) -> None:
        if self._pump_controller is None:
            self._set_command_status(
                "FAILED — pump controller unavailable"
            )
            return

        amount = 0.0

        try:
            if self._bolus_field is not None:
                amount = float(
                    self._bolus_field.model.as_float
                )
        except Exception:
            self._set_command_status(
                "FAILED — invalid bolus value"
            )
            return

        if amount <= 0:
            self._set_command_status(
                "FAILED — bolus must be > 0"
            )
            return

        try:
            if self._on_bolus_request is not None:
                success = bool(
                    self._on_bolus_request(amount)
                )
            else:
                success = bool(
                    self._pump_controller.request_bolus(
                        amount
                    )
                )

            self._set_command_status(
                "REQUEST_BOLUS SENT"
                if success
                else "REQUEST_BOLUS FAILED"
            )

        except Exception as exc:
            carb.log_error(
                f"[IITP DT] Bolus request failed: {exc}"
            )
            self._set_command_status(
                "REQUEST_BOLUS FAILED"
            )

    def _on_request_status(self) -> None:
        self._run_pump_command(
            "REQUEST_STATUS",
            lambda: self._pump_controller.request_status(),
        )

    def _on_request_suspend(self) -> None:
        self._run_pump_command(
            "REQUEST_SUSPEND",
            lambda: self._pump_controller.request_suspend(),
        )

    def _on_request_resume(self) -> None:
        self._run_pump_command(
            "REQUEST_RESUME",
            lambda: self._pump_controller.request_resume(),
        )

    def _run_pump_command(
        self,
        name: str,
        callback: Callable[[], bool],
    ) -> None:
        if self._pump_controller is None:
            self._set_command_status(
                f"{name} FAILED — controller unavailable"
            )
            return

        try:
            success = bool(callback())

            self._set_command_status(
                f"{name} SENT"
                if success
                else f"{name} FAILED"
            )

        except Exception as exc:
            carb.log_error(
                f"[IITP DT] {name} failed: {exc}"
            )
            self._set_command_status(
                f"{name} FAILED"
            )

    def _set_command_status(
        self,
        text: str,
    ) -> None:
        self._set_text(
            self._command_status_label,
            f"Last command: {text}",
        )

        if self._pump_controller is not None:
            try:
                state = self._pump_controller.get_state()
                self._set_text(
                    self._command_status_label,
                    (
                        f"Last command: "
                        f"{state.get('last_command_status', text)}"
                    ),
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_text(
        label: Any,
        text: str,
    ) -> None:
        if label is None:
            return

        try:
            label.text = str(text)
        except Exception:
            pass

    @staticmethod
    def _number(
        value: Any,
    ) -> Optional[float]:
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_glucose(
        value: Optional[float],
    ) -> str:
        if value is None:
            return "-- mg/dL"

        return f"{value:.1f} mg/dL"

    @staticmethod
    def _format_rate(
        value: Optional[float],
    ) -> str:
        if value is None:
            return "-- U/h"

        return f"{value:.2f} U/h"

    @staticmethod
    def _format_ratio(
        value: Optional[float],
    ) -> str:
        if value is None:
            return "--"

        # Values in this project are normally represented in [0, 1].
        if 0.0 <= value <= 1.0:
            return f"{value:.1%}"

        return f"{value:.3f}"

    @staticmethod
    def _format_number(
        value: Optional[float],
    ) -> str:
        if value is None:
            return "--"

        return f"{value:.3f}"

    @staticmethod
    def _format_number_with_unit(
        value: Optional[float],
        unit: str,
    ) -> str:
        if value is None:
            return f"-- {unit}"

        return f"{value:+.2f} {unit}"

    @staticmethod
    def _format_simulation_time(
        value: Any,
    ) -> str:
        if value is None:
            return "Sim: --"

        try:
            seconds = max(0.0, float(value))
        except (TypeError, ValueError):
            return "Sim: --"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        return f"Sim: {hours:02d}:{minutes:02d}:{secs:02d}"


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def create_ui(
    telemetry_handler: Optional[Any] = None,
    pump_controller: Optional[Any] = None,
    cgm_controller: Optional[Any] = None,
    websocket_client: Optional[Any] = None,
    on_bolus_request: Optional[Callable[[float], bool]] = None,
) -> DigitalTwinUI:
    """Create the IITP Digital Twin operator UI."""
    return DigitalTwinUI(
        telemetry_handler=telemetry_handler,
        pump_controller=pump_controller,
        cgm_controller=cgm_controller,
        websocket_client=websocket_client,
        on_bolus_request=on_bolus_request,
    )
