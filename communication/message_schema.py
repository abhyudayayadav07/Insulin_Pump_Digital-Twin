"""
IITP Digital Twin - WebSocket message schema.

This module defines the JSON message contract shared by:

    External Python Digital Twin
                <-->
             WebSocket
                <-->
          Omniverse Kit

The schema is intentionally implemented with standard-library Python only.
It does not depend on Omniverse, so the same validation/building logic can
also be imported by the external Python digital-twin process.

Message envelope:

    {
        "type": "telemetry" | "command" | "response" | "heartbeat",
        "timestamp": "...",
        "source": "...",
        "message_id": "...",
        "payload": {...}
    }

For backward compatibility with the first prototype, telemetry and command
builders also expose their important fields at the top level. The receiver
should therefore accept both:

    {"type": "telemetry", "patient": {...}, ...}

and the newer envelope form:

    {"type": "telemetry", "payload": {"patient": {...}, ...}}

The schema module does not make safety decisions. It only validates message
shape and provides serialization/deserialization helpers.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional


# ----------------------------------------------------------------------
# Message types
# ----------------------------------------------------------------------

TYPE_TELEMETRY = "telemetry"
TYPE_COMMAND = "command"
TYPE_RESPONSE = "response"
TYPE_HEARTBEAT = "heartbeat"
TYPE_ERROR = "error"

SOURCE_DIGITAL_TWIN = "digital_twin"
SOURCE_OMNIVERSE = "omniverse"
SOURCE_INSULIN_PUMP = "insulin_pump"
SOURCE_CGM = "cgm"

COMMAND_REQUEST_BOLUS = "REQUEST_BOLUS"
COMMAND_REQUEST_BASAL_CHANGE = "REQUEST_BASAL_CHANGE"
COMMAND_REQUEST_SUSPEND = "REQUEST_SUSPEND"
COMMAND_REQUEST_RESUME = "REQUEST_RESUME"
COMMAND_REQUEST_STATUS = "REQUEST_STATUS"

ALLOWED_MESSAGE_TYPES = {
    TYPE_TELEMETRY,
    TYPE_COMMAND,
    TYPE_RESPONSE,
    TYPE_HEARTBEAT,
    TYPE_ERROR,
}

ALLOWED_COMMANDS = {
    COMMAND_REQUEST_BOLUS,
    COMMAND_REQUEST_BASAL_CHANGE,
    COMMAND_REQUEST_SUSPEND,
    COMMAND_REQUEST_RESUME,
    COMMAND_REQUEST_STATUS,
}


# ----------------------------------------------------------------------
# Validation exception
# ----------------------------------------------------------------------

class MessageSchemaError(ValueError):
    """Raised when a message does not satisfy the IITP message schema."""


# ----------------------------------------------------------------------
# Dataclass envelope
# ----------------------------------------------------------------------

@dataclass
class MessageEnvelope:
    """
    Generic WebSocket message envelope.

    `payload` contains the operation-specific data.
    """

    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = SOURCE_DIGITAL_TWIN
    timestamp: str = field(default_factory=lambda: utc_timestamp())
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "type": self.type,
            "timestamp": self.timestamp,
            "source": self.source,
            "message_id": self.message_id,
            "payload": self.payload,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            separators=(",", ":"),
        )


# ----------------------------------------------------------------------
# Timestamp / ID helpers
# ----------------------------------------------------------------------

def utc_timestamp() -> str:
    """Return a timezone-aware UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def generate_message_id() -> str:
    """Generate a unique message ID."""
    return str(uuid.uuid4())


# ----------------------------------------------------------------------
# Generic builders
# ----------------------------------------------------------------------

def build_message(
    message_type: str,
    payload: Optional[Mapping[str, Any]] = None,
    source: str = SOURCE_DIGITAL_TWIN,
    message_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a standard envelope message.

    Raises:
        MessageSchemaError for unsupported message types.
    """
    _validate_message_type(message_type)

    return MessageEnvelope(
        type=message_type,
        payload=dict(payload or {}),
        source=str(source),
        message_id=message_id or generate_message_id(),
        timestamp=timestamp or utc_timestamp(),
    ).to_dict()


# ----------------------------------------------------------------------
# Telemetry builder
# ----------------------------------------------------------------------

def build_telemetry_message(
    *,
    patient: Optional[Mapping[str, Any]] = None,
    pump: Optional[Mapping[str, Any]] = None,
    dt1: Optional[Mapping[str, Any]] = None,
    dt2: Optional[Mapping[str, Any]] = None,
    security: Optional[Mapping[str, Any]] = None,
    decision: Optional[Mapping[str, Any]] = None,
    simulation_time: Optional[float] = None,
    timestamp: Optional[str] = None,
    source: str = SOURCE_DIGITAL_TWIN,
    message_id: Optional[str] = None,
    flat_compatibility: bool = True,
) -> Dict[str, Any]:
    """
    Build a complete telemetry message.

    `flat_compatibility=True` preserves the original prototype schema used
    by telemetry_handler.py:

        {
            "type": "telemetry",
            "timestamp": "...",
            "simulation_time": 3600,
            "patient": {...},
            "pump": {...},
            ...
        }

    A nested `payload` containing the same telemetry is also included, making
    the message compatible with the more formal envelope format.
    """
    payload: Dict[str, Any] = {
        "simulation_time": simulation_time,
        "patient": dict(patient or {}),
        "pump": dict(pump or {}),
        "dt1": dict(dt1 or {}),
        "dt2": dict(dt2 or {}),
        "security": dict(security or {}),
        "decision": dict(decision or {}),
    }

    message = build_message(
        TYPE_TELEMETRY,
        payload=payload,
        source=source,
        message_id=message_id,
        timestamp=timestamp,
    )

    if flat_compatibility:
        message.update(payload)

    return message


# ----------------------------------------------------------------------
# Command builder
# ----------------------------------------------------------------------

def build_command_message(
    command: str,
    value: Any = None,
    *,
    source: str = SOURCE_OMNIVERSE,
    timestamp: Optional[str] = None,
    message_id: Optional[str] = None,
    flat_compatibility: bool = True,
) -> Dict[str, Any]:
    """
    Build a command request from Omniverse to the external DT.

    Example:

        build_command_message(
            "REQUEST_BOLUS",
            1.0,
        )

    produces a message containing:

        {
            "type": "command",
            "source": "omniverse",
            "command": "REQUEST_BOLUS",
            "value": 1.0
        }

    The external safety/decision engine must validate the request before
    any real/simulated pump actuation occurs.
    """
    _validate_command(command)

    payload = {
        "source": SOURCE_INSULIN_PUMP,
        "command": command,
        "value": value,
    }

    message = build_message(
        TYPE_COMMAND,
        payload=payload,
        source=source,
        timestamp=timestamp,
        message_id=message_id,
    )

    if flat_compatibility:
        message.update(
            {
                "command": command,
                "value": value,
            }
        )

    return message


def build_bolus_request(
    amount: float,
    *,
    source: str = SOURCE_OMNIVERSE,
) -> Dict[str, Any]:
    """Build a REQUEST_BOLUS command."""
    amount = _validate_non_negative_number(
        amount,
        "bolus amount",
    )

    if amount <= 0:
        raise MessageSchemaError(
            "Bolus amount must be greater than zero"
        )

    return build_command_message(
        COMMAND_REQUEST_BOLUS,
        amount,
        source=source,
    )


def build_basal_change_request(
    rate: float,
    *,
    source: str = SOURCE_OMNIVERSE,
) -> Dict[str, Any]:
    """Build a REQUEST_BASAL_CHANGE command."""
    rate = _validate_non_negative_number(
        rate,
        "basal rate",
    )

    return build_command_message(
        COMMAND_REQUEST_BASAL_CHANGE,
        rate,
        source=source,
    )


def build_suspend_request(
    *,
    source: str = SOURCE_OMNIVERSE,
) -> Dict[str, Any]:
    """Build a REQUEST_SUSPEND command."""
    return build_command_message(
        COMMAND_REQUEST_SUSPEND,
        None,
        source=source,
    )


def build_resume_request(
    *,
    source: str = SOURCE_OMNIVERSE,
) -> Dict[str, Any]:
    """Build a REQUEST_RESUME command."""
    return build_command_message(
        COMMAND_REQUEST_RESUME,
        None,
        source=source,
    )


def build_status_request(
    *,
    source: str = SOURCE_OMNIVERSE,
) -> Dict[str, Any]:
    """Build a REQUEST_STATUS command."""
    return build_command_message(
        COMMAND_REQUEST_STATUS,
        None,
        source=source,
    )


# ----------------------------------------------------------------------
# Response / heartbeat builders
# ----------------------------------------------------------------------

def build_response_message(
    request_message_id: str,
    status: str,
    *,
    result: Optional[Mapping[str, Any]] = None,
    error: Optional[str] = None,
    source: str = SOURCE_DIGITAL_TWIN,
) -> Dict[str, Any]:
    """Build a response to a previously received command."""
    payload = {
        "request_message_id": request_message_id,
        "status": str(status).upper(),
        "result": dict(result or {}),
        "error": error,
    }

    return build_message(
        TYPE_RESPONSE,
        payload=payload,
        source=source,
    )


def build_heartbeat_message(
    *,
    source: str = SOURCE_DIGITAL_TWIN,
) -> Dict[str, Any]:
    """Build a lightweight connection heartbeat."""
    return build_message(
        TYPE_HEARTBEAT,
        payload={
            "status": "alive",
        },
        source=source,
    )


def build_error_message(
    error: str,
    *,
    code: str = "MESSAGE_ERROR",
    source: str = SOURCE_DIGITAL_TWIN,
) -> Dict[str, Any]:
    """Build a protocol-level error message."""
    return build_message(
        TYPE_ERROR,
        payload={
            "code": code,
            "error": str(error),
        },
        source=source,
    )


# ----------------------------------------------------------------------
# Parsing / normalization
# ----------------------------------------------------------------------

def parse_message(
    raw_message: Any,
    *,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Parse a JSON string/bytes or an already decoded dictionary.

    Returns a dictionary.

    Supports both:
        - formal envelope messages with payload
        - legacy flat telemetry/command messages
    """
    if isinstance(raw_message, bytes):
        try:
            raw_message = raw_message.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MessageSchemaError(
                f"Invalid UTF-8 message: {exc}"
            ) from exc

    if isinstance(raw_message, str):
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError as exc:
            raise MessageSchemaError(
                f"Invalid JSON: {exc}"
            ) from exc
    elif isinstance(raw_message, Mapping):
        message = dict(raw_message)
    else:
        raise MessageSchemaError(
            "Message must be JSON text, bytes, or a mapping"
        )

    if not isinstance(message, dict):
        raise MessageSchemaError(
            "JSON root must be an object"
        )

    if validate:
        validate_message(message)

    return normalize_message(message)


def parse_json(
    raw_message: str,
    *,
    validate: bool = True,
) -> Dict[str, Any]:
    """Convenience alias for parse_message()."""
    return parse_message(
        raw_message,
        validate=validate,
    )


def normalize_message(
    message: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Normalize a message so downstream code can use payload consistently.

    For a legacy flat message, the operation fields are copied into payload.
    Existing fields are preserved.
    """
    normalized = dict(message)

    message_type = normalized.get("type")

    if not isinstance(message_type, str):
        return normalized

    payload = normalized.get("payload")

    if isinstance(payload, Mapping):
        normalized["payload"] = dict(payload)
        return normalized

    if message_type == TYPE_TELEMETRY:
        keys = {
            "simulation_time",
            "patient",
            "pump",
            "dt1",
            "dt2",
            "security",
            "decision",
        }

        normalized["payload"] = {
            key: normalized[key]
            for key in keys
            if key in normalized
        }

    elif message_type == TYPE_COMMAND:
        keys = {
            "source",
            "command",
            "value",
        }

        normalized["payload"] = {
            key: normalized[key]
            for key in keys
            if key in normalized
        }

    else:
        normalized["payload"] = {}

    return normalized


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def validate_message(
    message: Mapping[str, Any],
) -> bool:
    """
    Validate a message against the protocol contract.

    Returns True when valid.

    Raises:
        MessageSchemaError when invalid.
    """
    if not isinstance(message, Mapping):
        raise MessageSchemaError(
            "Message must be a mapping"
        )

    message_type = message.get("type")

    if not isinstance(message_type, str):
        raise MessageSchemaError(
            "Message requires string field 'type'"
        )

    _validate_message_type(message_type)

    if "message_id" in message and message["message_id"] is not None:
        if not isinstance(message["message_id"], str):
            raise MessageSchemaError(
                "'message_id' must be a string"
            )

    if "timestamp" in message and message["timestamp"] is not None:
        if not isinstance(message["timestamp"], str):
            raise MessageSchemaError(
                "'timestamp' must be a string"
            )

    payload = message.get("payload")

    if payload is not None and not isinstance(payload, Mapping):
        raise MessageSchemaError(
            "'payload' must be an object"
        )

    if message_type == TYPE_TELEMETRY:
        _validate_telemetry(message)

    elif message_type == TYPE_COMMAND:
        _validate_command_message(message)

    elif message_type == TYPE_RESPONSE:
        _validate_response(message)

    return True


def _validate_message_type(
    message_type: str,
) -> None:
    if message_type not in ALLOWED_MESSAGE_TYPES:
        raise MessageSchemaError(
            f"Unsupported message type: {message_type}"
        )


def _validate_telemetry(
    message: Mapping[str, Any],
) -> None:
    payload = message.get("payload")

    if isinstance(payload, Mapping):
        data = payload
    else:
        data = message

    # These are required for the current digital-twin telemetry contract.
    # Individual subfields remain flexible so the protocol can evolve.
    required_sections = {
        "patient",
        "pump",
        "dt1",
        "dt2",
        "security",
        "decision",
    }

    for section in required_sections:
        if section in data and not isinstance(
            data[section],
            Mapping,
        ):
            raise MessageSchemaError(
                f"Telemetry section '{section}' must be an object"
            )

    if "simulation_time" in data:
        _validate_number_or_none(
            data["simulation_time"],
            "simulation_time",
        )


def _validate_command_message(
    message: Mapping[str, Any],
) -> None:
    payload = message.get("payload")

    if isinstance(payload, Mapping):
        command = payload.get("command")
        value = payload.get("value")
    else:
        command = message.get("command")
        value = message.get("value")

    if not isinstance(command, str):
        raise MessageSchemaError(
            "Command message requires string 'command'"
        )

    _validate_command(command)

    if command in {
        COMMAND_REQUEST_BOLUS,
        COMMAND_REQUEST_BASAL_CHANGE,
    }:
        if value is None:
            raise MessageSchemaError(
                f"{command} requires a numeric value"
            )

        numeric_value = _validate_non_negative_number(
            value,
            f"{command} value",
        )

        if command == COMMAND_REQUEST_BOLUS and numeric_value <= 0:
            raise MessageSchemaError(
                "REQUEST_BOLUS value must be greater than zero"
            )


def _validate_response(
    message: Mapping[str, Any],
) -> None:
    payload = message.get("payload")

    if not isinstance(payload, Mapping):
        raise MessageSchemaError(
            "Response message requires a payload object"
        )

    if "status" not in payload:
        raise MessageSchemaError(
            "Response message requires payload.status"
        )


def _validate_command(
    command: str,
) -> None:
    if command not in ALLOWED_COMMANDS:
        raise MessageSchemaError(
            f"Unsupported pump command: {command}"
        )


def _validate_number_or_none(
    value: Any,
    field_name: str,
) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MessageSchemaError(
            f"'{field_name}' must be numeric or null"
        ) from exc


def _validate_non_negative_number(
    value: Any,
    field_name: str,
) -> float:
    numeric = _validate_number_or_none(
        value,
        field_name,
    )

    if numeric is None:
        raise MessageSchemaError(
            f"'{field_name}' cannot be null"
        )

    if numeric < 0:
        raise MessageSchemaError(
            f"'{field_name}' cannot be negative"
        )

    return numeric


# ----------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------

def serialize_message(
    message: Mapping[str, Any],
    *,
    validate: bool = True,
) -> str:
    """
    Serialize a protocol message to compact JSON.

    The resulting string is ready for WebSocket send().
    """
    if validate:
        validate_message(message)

    return json.dumps(
        dict(message),
        separators=(",", ":"),
        ensure_ascii=False,
    )


def deserialize_message(
    raw_message: Any,
    *,
    validate: bool = True,
) -> Dict[str, Any]:
    """Deserialize and normalize a WebSocket message."""
    return parse_message(
        raw_message,
        validate=validate,
    )


# ----------------------------------------------------------------------
# Utility accessors
# ----------------------------------------------------------------------

def get_message_type(
    message: Mapping[str, Any],
) -> Optional[str]:
    """Return the message type."""
    value = message.get("type")

    return str(value) if value is not None else None


def get_payload(
    message: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return normalized payload data."""
    normalized = normalize_message(message)

    payload = normalized.get("payload")

    if not isinstance(payload, Mapping):
        return {}

    return dict(payload)


def get_message_id(
    message: Mapping[str, Any],
) -> Optional[str]:
    """Return the message ID."""
    value = message.get("message_id")

    return str(value) if value is not None else None


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

__all__ = [
    "MessageEnvelope",
    "MessageSchemaError",

    "TYPE_TELEMETRY",
    "TYPE_COMMAND",
    "TYPE_RESPONSE",
    "TYPE_HEARTBEAT",
    "TYPE_ERROR",

    "SOURCE_DIGITAL_TWIN",
    "SOURCE_OMNIVERSE",
    "SOURCE_INSULIN_PUMP",
    "SOURCE_CGM",

    "COMMAND_REQUEST_BOLUS",
    "COMMAND_REQUEST_BASAL_CHANGE",
    "COMMAND_REQUEST_SUSPEND",
    "COMMAND_REQUEST_RESUME",
    "COMMAND_REQUEST_STATUS",

    "build_message",
    "build_telemetry_message",
    "build_command_message",
    "build_bolus_request",
    "build_basal_change_request",
    "build_suspend_request",
    "build_resume_request",
    "build_status_request",
    "build_response_message",
    "build_heartbeat_message",
    "build_error_message",

    "parse_message",
    "parse_json",
    "normalize_message",
    "validate_message",

    "serialize_message",
    "deserialize_message",

    "get_message_type",
    "get_payload",
    "get_message_id",

    "utc_timestamp",
    "generate_message_id",
]
