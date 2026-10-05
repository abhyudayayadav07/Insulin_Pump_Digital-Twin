"""
IITP Digital Twin - Telemetry serializer.

This module converts digital-twin state into the canonical telemetry
representation used by the Omniverse communication layer.

Responsibilities:
    - Build canonical telemetry dictionaries.
    - Sanitize values so they are JSON-safe.
    - Serialize telemetry to compact JSON.
    - Validate telemetry using message_schema.py when available.
    - Deserialize incoming telemetry.
    - Provide helpers for incremental telemetry construction.

It deliberately contains no WebSocket code and no USD code.

Architecture:

    SimGlucose / DT1 / DT2 / Security / Decision
                         |
                         v
               TelemetrySerializer
                         |
                         v
                 message_schema
                         |
                         v
                       JSON
                         |
                         v
                    WebSocket
                         |
                         v
                    Omniverse

The serializer is also usable by the external Python digital-twin process,
so it does not import Omniverse-specific modules.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import date, datetime
from typing import Any, Dict, Mapping, Optional, Sequence

try:
    # Package layout:
    # communication/telemetry_serializer.py
    from .message_schema import (
        TYPE_TELEMETRY,
        SOURCE_DIGITAL_TWIN,
        build_telemetry_message,
        deserialize_message,
        serialize_message,
        validate_message,
    )
except ImportError:
    # Useful when this file is copied/run as a standalone module during
    # development or when the package is placed directly on sys.path.
    from message_schema import (  # type: ignore
        TYPE_TELEMETRY,
        SOURCE_DIGITAL_TWIN,
        build_telemetry_message,
        deserialize_message,
        serialize_message,
        validate_message,
    )


# ----------------------------------------------------------------------
# Canonical defaults
# ----------------------------------------------------------------------

DEFAULT_PATIENT = {
    "glucose": None,
    "cgm": None,
    "meal": 0.0,
}

DEFAULT_PUMP = {
    "insulin_rate": 0.0,
    "command": 0.0,
    "delivery": 0.0,
    "status": "UNKNOWN",
}

DEFAULT_DT1 = {
    "ml_prediction": None,
    "geco_prediction": None,
    "fused_prediction": None,
    "residual": None,
    "cusum": None,
    "status": "UNKNOWN",
}

DEFAULT_DT2 = {
    "hazard": None,
    "survival": None,
    "event_probability": None,
    "time_to_event": None,
    "risk_level": "UNKNOWN",
}

DEFAULT_SECURITY = {
    "attack_detected": False,
    "attack_type": None,
    "sensor_integrity": None,
    "pump_integrity": None,
}

DEFAULT_DECISION = {
    "action": "UNKNOWN",
}


# ----------------------------------------------------------------------
# Serializer
# ----------------------------------------------------------------------

class TelemetrySerializer:
    """
    Build and serialize canonical IITP digital-twin telemetry.

    The serializer is intentionally state-light. A single instance can be
    reused for every simulation timestep.
    """

    def __init__(
        self,
        source: str = SOURCE_DIGITAL_TWIN,
        validate: bool = True,
        include_flat_compatibility: bool = True,
        ensure_ascii: bool = False,
    ):
        self.source = source
        self.validate = bool(validate)
        self.include_flat_compatibility = bool(
            include_flat_compatibility
        )
        self.ensure_ascii = bool(ensure_ascii)

    # ------------------------------------------------------------------
    # Public serialization API
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        patient: Optional[Mapping[str, Any]] = None,
        pump: Optional[Mapping[str, Any]] = None,
        dt1: Optional[Mapping[str, Any]] = None,
        dt2: Optional[Mapping[str, Any]] = None,
        security: Optional[Mapping[str, Any]] = None,
        decision: Optional[Mapping[str, Any]] = None,
        simulation_time: Optional[float] = None,
        timestamp: Optional[Any] = None,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build one canonical telemetry dictionary.

        Missing sections are populated with stable defaults. NaN and
        infinity are converted to None so the result remains JSON-safe.
        """
        telemetry = build_telemetry_message(
            patient=self._merge_section(
                DEFAULT_PATIENT,
                patient,
            ),
            pump=self._merge_section(
                DEFAULT_PUMP,
                pump,
            ),
            dt1=self._merge_section(
                DEFAULT_DT1,
                dt1,
            ),
            dt2=self._merge_section(
                DEFAULT_DT2,
                dt2,
            ),
            security=self._merge_section(
                DEFAULT_SECURITY,
                security,
            ),
            decision=self._merge_section(
                DEFAULT_DECISION,
                decision,
            ),
            simulation_time=self._number_or_none(
                simulation_time
            ),
            timestamp=self._normalize_timestamp(
                timestamp
            ),
            source=self.source,
            message_id=message_id,
            flat_compatibility=self.include_flat_compatibility,
        )

        telemetry = self.sanitize(telemetry)

        if self.validate:
            validate_message(telemetry)

        return telemetry

    def serialize(
        self,
        telemetry: Mapping[str, Any],
    ) -> str:
        """
        Convert a telemetry dictionary into compact JSON.

        The returned string can be passed directly to WebSocket send().
        """
        sanitized = self.sanitize(dict(telemetry))

        if self.validate:
            validate_message(sanitized)

        return json.dumps(
            sanitized,
            separators=(",", ":"),
            ensure_ascii=self.ensure_ascii,
            allow_nan=False,
        )

    def deserialize(
        self,
        raw_message: Any,
    ) -> Dict[str, Any]:
        """
        Deserialize a telemetry JSON message and normalize its payload.

        Raises ValueError/MessageSchemaError for invalid messages when
        validation is enabled by the underlying schema.
        """
        message = deserialize_message(
            raw_message,
            validate=self.validate,
        )

        if message.get("type") != TYPE_TELEMETRY:
            raise ValueError(
                "Expected telemetry message, received "
                f"{message.get('type')}"
            )

        return message

    def round_trip(
        self,
        telemetry: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Serialize and deserialize telemetry as a protocol smoke test."""
        return self.deserialize(
            self.serialize(telemetry)
        )

    # ------------------------------------------------------------------
    # Convenience builders
    # ------------------------------------------------------------------

    def from_state(
        self,
        state: Mapping[str, Any],
        *,
        simulation_time: Optional[float] = None,
        timestamp: Optional[Any] = None,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build telemetry from a single state dictionary.

        This is useful when the external digital twin already has a
        dictionary containing patient/pump/dt1/dt2/security/decision.
        """
        state = dict(state)

        if simulation_time is None:
            simulation_time = state.get("simulation_time")

        if timestamp is None:
            timestamp = state.get("timestamp")

        return self.build(
            patient=state.get("patient"),
            pump=state.get("pump"),
            dt1=state.get("dt1"),
            dt2=state.get("dt2"),
            security=state.get("security"),
            decision=state.get("decision"),
            simulation_time=simulation_time,
            timestamp=timestamp,
            message_id=message_id,
        )

    def build_from_components(
        self,
        *,
        glucose: Optional[float] = None,
        cgm: Optional[float] = None,
        meal: float = 0.0,
        insulin_rate: float = 0.0,
        insulin_command: float = 0.0,
        insulin_delivery: float = 0.0,
        pump_status: str = "UNKNOWN",
        ml_prediction: Optional[float] = None,
        geco_prediction: Optional[float] = None,
        fused_prediction: Optional[float] = None,
        residual: Optional[float] = None,
        cusum: Optional[float] = None,
        dt1_status: str = "UNKNOWN",
        hazard: Optional[float] = None,
        survival: Optional[float] = None,
        event_probability: Optional[float] = None,
        time_to_event: Optional[float] = None,
        risk_level: str = "UNKNOWN",
        attack_detected: bool = False,
        attack_type: Optional[str] = None,
        sensor_integrity: Optional[float] = None,
        pump_integrity: Optional[float] = None,
        decision_action: str = "UNKNOWN",
        simulation_time: Optional[float] = None,
        timestamp: Optional[Any] = None,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convenience API for simulation loops.

        This lets the caller pass scalar outputs directly without manually
        constructing six nested dictionaries.
        """
        return self.build(
            patient={
                "glucose": glucose,
                "cgm": cgm,
                "meal": meal,
            },
            pump={
                "insulin_rate": insulin_rate,
                "command": insulin_command,
                "delivery": insulin_delivery,
                "status": pump_status,
            },
            dt1={
                "ml_prediction": ml_prediction,
                "geco_prediction": geco_prediction,
                "fused_prediction": fused_prediction,
                "residual": residual,
                "cusum": cusum,
                "status": dt1_status,
            },
            dt2={
                "hazard": hazard,
                "survival": survival,
                "event_probability": event_probability,
                "time_to_event": time_to_event,
                "risk_level": risk_level,
            },
            security={
                "attack_detected": attack_detected,
                "attack_type": attack_type,
                "sensor_integrity": sensor_integrity,
                "pump_integrity": pump_integrity,
            },
            decision={
                "action": decision_action,
            },
            simulation_time=simulation_time,
            timestamp=timestamp,
            message_id=message_id,
        )

    # ------------------------------------------------------------------
    # Serialization utilities
    # ------------------------------------------------------------------

    @classmethod
    def sanitize(
        cls,
        value: Any,
    ) -> Any:
        """
        Recursively convert a value into JSON-safe Python objects.

        Handles:
            - dict / Mapping
            - list / tuple / set
            - dataclasses exposing to_dict()
            - datetime/date
            - numpy-like scalar objects exposing item()
            - NaN / infinity
            - primitive values
        """
        if value is None:
            return None

        if isinstance(value, bool):
            return value

        if isinstance(value, (str, int)):
            return value

        if isinstance(value, float):
            return (
                value
                if math.isfinite(value)
                else None
            )

        if isinstance(value, (datetime, date)):
            return value.isoformat()

        if isinstance(value, Mapping):
            return {
                str(key): cls.sanitize(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set, frozenset)):
            return [
                cls.sanitize(item)
                for item in value
            ]

        # NumPy/Pandas-like scalar.
        item_method = getattr(value, "item", None)

        if callable(item_method):
            try:
                return cls.sanitize(item_method())
            except Exception:
                pass

        # Dataclass/custom object convention.
        to_dict = getattr(value, "to_dict", None)

        if callable(to_dict):
            try:
                return cls.sanitize(to_dict())
            except Exception:
                pass

        # Last-resort string representation keeps serialization alive
        # without allowing arbitrary Python objects into JSON.
        return str(value)

    @staticmethod
    def _merge_section(
        defaults: Mapping[str, Any],
        section: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        result = dict(defaults)

        if isinstance(section, Mapping):
            result.update(dict(section))

        return result

    @staticmethod
    def _number_or_none(
        value: Any,
    ) -> Optional[float]:
        if value is None:
            return None

        try:
            number = float(value)

            if not math.isfinite(number):
                return None

            return number

        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_timestamp(
        timestamp: Any,
    ) -> Optional[str]:
        if timestamp is None:
            return None

        if isinstance(timestamp, datetime):
            return timestamp.isoformat()

        if isinstance(timestamp, date):
            return timestamp.isoformat()

        return str(timestamp)


# ----------------------------------------------------------------------
# Functional API
# ----------------------------------------------------------------------

def serialize_telemetry(
    telemetry: Mapping[str, Any],
    *,
    validate: bool = True,
) -> str:
    """Serialize an existing telemetry dictionary."""
    serializer = TelemetrySerializer(
        validate=validate,
    )

    return serializer.serialize(telemetry)


def deserialize_telemetry(
    raw_message: Any,
    *,
    validate: bool = True,
) -> Dict[str, Any]:
    """Deserialize a telemetry JSON message."""
    serializer = TelemetrySerializer(
        validate=validate,
    )

    return serializer.deserialize(raw_message)


def create_telemetry(
    *,
    patient: Optional[Mapping[str, Any]] = None,
    pump: Optional[Mapping[str, Any]] = None,
    dt1: Optional[Mapping[str, Any]] = None,
    dt2: Optional[Mapping[str, Any]] = None,
    security: Optional[Mapping[str, Any]] = None,
    decision: Optional[Mapping[str, Any]] = None,
    simulation_time: Optional[float] = None,
    timestamp: Optional[Any] = None,
    source: str = SOURCE_DIGITAL_TWIN,
    message_id: Optional[str] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """Build canonical telemetry without creating a serializer manually."""
    serializer = TelemetrySerializer(
        source=source,
        validate=validate,
    )

    return serializer.build(
        patient=patient,
        pump=pump,
        dt1=dt1,
        dt2=dt2,
        security=security,
        decision=decision,
        simulation_time=simulation_time,
        timestamp=timestamp,
        message_id=message_id,
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

__all__ = [
    "TelemetrySerializer",
    "serialize_telemetry",
    "deserialize_telemetry",
    "create_telemetry",
    "DEFAULT_PATIENT",
    "DEFAULT_PUMP",
    "DEFAULT_DT1",
    "DEFAULT_DT2",
    "DEFAULT_SECURITY",
    "DEFAULT_DECISION",
]
