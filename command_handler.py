"""
IITP Digital Twin - Command Handler.

Routes commands arriving from Omniverse to the external safety/decision
pipeline.

Important safety boundary:

    Omniverse UI
        |
        v
    CommandHandler
        |
        v
    validation
        |
        v
    application/safety callback
        |
        v
    Decision Engine
        |
        +--> ALLOW
        +--> BLOCK
        +--> SAFE_MODE
        +--> EMERGENCY_STOP

This module never directly actuates an insulin pump.

It supports the command protocol already used by PumpController:
    REQUEST_BOLUS
    REQUEST_BASAL_CHANGE
    REQUEST_SUSPEND
    REQUEST_RESUME
    REQUEST_STATUS

The real DT/security/decision modules can be connected through callbacks.
"""

from __future__ import annotations

import inspect
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

try:
    from .message_schema import (
        TYPE_COMMAND,
        build_error_message,
        build_response_message,
        get_message_id,
        get_payload,
        validate_message,
        COMMAND_REQUEST_BASAL_CHANGE,
        COMMAND_REQUEST_BOLUS,
        COMMAND_REQUEST_RESUME,
        COMMAND_REQUEST_STATUS,
        COMMAND_REQUEST_SUSPEND,
    )
except ImportError:
    from message_schema import (  # type: ignore
        TYPE_COMMAND,
        build_error_message,
        build_response_message,
        get_message_id,
        get_payload,
        validate_message,
        COMMAND_REQUEST_BASAL_CHANGE,
        COMMAND_REQUEST_BOLUS,
        COMMAND_REQUEST_RESUME,
        COMMAND_REQUEST_STATUS,
        COMMAND_REQUEST_SUSPEND,
    )


CommandCallback = Callable[
    [Dict[str, Any], "CommandRequest"],
    Optional[Awaitable[Optional[Dict[str, Any]]]],
]


def _get_command(message: Mapping[str, Any]) -> Optional[str]:
    """Return the command from either the flat or normalized payload form."""
    payload = message.get("payload")
    if isinstance(payload, Mapping) and payload.get("command") is not None:
        return str(payload.get("command"))
    value = message.get("command")
    return str(value) if value is not None else None


@dataclass
class CommandRequest:
    """Normalized command received from an Omniverse client."""

    request_id: str
    command: str
    source: str
    value: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    received_at: float = field(default_factory=time.time)
    connection_id: Optional[str] = None

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.received_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "command": self.command,
            "source": self.source,
            "value": self.value,
            "payload": dict(self.payload),
            "received_at": self.received_at,
            "connection_id": self.connection_id,
        }


@dataclass
class CommandResult:
    """Structured result returned by the command handler."""

    request_id: str
    command: str
    status: str
    accepted: bool = False
    action: str = "NONE"
    reason: Optional[str] = None
    result: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "request_id": self.request_id,
            "command": self.command,
            "status": self.status,
            "accepted": self.accepted,
            "action": self.action,
            "reason": self.reason,
            "result": dict(self.result),
        }
        return _json_safe(data)


class CommandHandler:
    """
    Validate and route Omniverse commands.

    The handler is deliberately conservative:
    - malformed requests are rejected;
    - unsupported commands are rejected;
    - dangerous requests are NOT executed locally;
    - the application callback decides whether a request is permitted.

    A future integration can connect the callback to:
        security -> hazard/risk -> decision engine -> pump controller
    """

    SUPPORTED_COMMANDS = frozenset(
        {
            COMMAND_REQUEST_BOLUS,
            COMMAND_REQUEST_BASAL_CHANGE,
            COMMAND_REQUEST_SUSPEND,
            COMMAND_REQUEST_RESUME,
            COMMAND_REQUEST_STATUS,
        }
    )

    # Conservative validation bounds for UI requests. These are protocol
    # sanity limits, not clinical dosing recommendations.
    DEFAULT_MAX_BOLUS = 25.0
    DEFAULT_MAX_BASAL_RATE = 20.0

    def __init__(
        self,
        *,
        application_callback: Optional[CommandCallback] = None,
        max_bolus: float = DEFAULT_MAX_BOLUS,
        max_basal_rate: float = DEFAULT_MAX_BASAL_RATE,
        logger: Optional[logging.Logger] = None,
    ):
        self.application_callback = application_callback
        self.max_bolus = float(max_bolus)
        self.max_basal_rate = float(max_basal_rate)
        self.logger = logger or logging.getLogger(
            "iitp.digital_twin.command_handler"
        )

        self._received_count = 0
        self._accepted_count = 0
        self._rejected_count = 0
        self._error_count = 0
        self._last_request: Optional[CommandRequest] = None
        self._last_result: Optional[CommandResult] = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def handle(
        self,
        message: Dict[str, Any],
        connection: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Handle one command message and return a protocol response.

        This method is compatible with ConnectionManager's on_command
        callback shape.
        """
        self._received_count += 1

        try:
            validate_message(message)
        except Exception as exc:
            self._error_count += 1
            return build_error_message(
                str(exc),
                code="INVALID_COMMAND_MESSAGE",
            )

        if message.get("type") != TYPE_COMMAND:
            self._error_count += 1
            return build_error_message(
                "Expected a command message",
                code="NOT_A_COMMAND",
            )

        request = self._parse_request(
            message,
            connection,
        )

        self._last_request = request

        self.logger.info(
            "Received command=%s request_id=%s source=%s",
            request.command,
            request.request_id,
            request.source,
        )

        validation_error = self._validate_request(
            request
        )

        if validation_error is not None:
            self._rejected_count += 1

            result = CommandResult(
                request_id=request.request_id,
                command=request.command,
                status="REJECTED",
                accepted=False,
                action="BLOCK",
                reason=validation_error,
            )

            return self._response(result)

        try:
            result = await self._dispatch(
                message,
                request,
            )
        except Exception as exc:
            self._error_count += 1
            self.logger.exception(
                "Command dispatch failed"
            )

            result = CommandResult(
                request_id=request.request_id,
                command=request.command,
                status="ERROR",
                accepted=False,
                action="BLOCK",
                reason=str(exc),
            )

        self._last_result = result

        if result.accepted:
            self._accepted_count += 1
        else:
            self._rejected_count += 1

        return self._response(result)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_request(
        self,
        message: Mapping[str, Any],
        connection: Optional[Any],
    ) -> CommandRequest:
        payload = get_payload(message)

        command = _get_command(message)

        request_id = get_message_id(message)
        if not request_id:
            request_id = str(uuid.uuid4())

        source = (
            message.get("source")
            or payload.get("source")
            or "omniverse"
        )

        value = (
            message.get("value")
            if "value" in message
            else payload.get("value")
        )

        value = _optional_float(value)

        connection_id = getattr(
            connection,
            "connection_id",
            None,
        )

        return CommandRequest(
            request_id=str(request_id),
            command=str(command),
            source=str(source),
            value=value,
            payload=dict(payload),
            connection_id=connection_id,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_request(
        self,
        request: CommandRequest,
    ) -> Optional[str]:
        if not request.command:
            return "Command is missing"

        if request.command not in self.SUPPORTED_COMMANDS:
            return (
                f"Unsupported command: {request.command}"
            )

        if not request.source:
            return "Command source is missing"

        if request.command == COMMAND_REQUEST_BOLUS:
            if request.value is None:
                return (
                    "REQUEST_BOLUS requires a numeric value"
                )

            if not math.isfinite(request.value):
                return "Bolus value must be finite"

            if request.value <= 0:
                return "Bolus value must be greater than zero"

            if request.value > self.max_bolus:
                return (
                    "Bolus request exceeds the configured "
                    "protocol sanity limit"
                )

        elif request.command == COMMAND_REQUEST_BASAL_CHANGE:
            if request.value is None:
                return (
                    "REQUEST_BASAL_CHANGE requires a numeric value"
                )

            if not math.isfinite(request.value):
                return "Basal rate must be finite"

            if request.value < 0:
                return "Basal rate cannot be negative"

            if request.value > self.max_basal_rate:
                return (
                    "Basal request exceeds the configured "
                    "protocol sanity limit"
                )

        return None

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _dispatch(
        self,
        message: Dict[str, Any],
        request: CommandRequest,
    ) -> CommandResult:
        """
        Pass the validated request to the real safety/decision pipeline.

        If no callback is connected, the command is deliberately NOT
        accepted for actuation.
        """
        if self.application_callback is None:
            return CommandResult(
                request_id=request.request_id,
                command=request.command,
                status="RECEIVED",
                accepted=False,
                action="BLOCK",
                reason=(
                    "Safety/decision callback is not connected; "
                    "no pump actuation performed."
                ),
            )

        result = self.application_callback(
            message,
            request,
        )

        if inspect.isawaitable(result):
            result = await result

        return self._normalize_callback_result(
            request,
            result,
        )

    def _normalize_callback_result(
        self,
        request: CommandRequest,
        result: Any,
    ) -> CommandResult:
        if isinstance(result, CommandResult):
            return result

        if result is None:
            return CommandResult(
                request_id=request.request_id,
                command=request.command,
                status="REJECTED",
                accepted=False,
                action="BLOCK",
                reason=(
                    "Safety/decision callback returned no decision"
                ),
            )

        if isinstance(result, Mapping):
            accepted = bool(
                result.get(
                    "accepted",
                    result.get("allow", False),
                )
            )

            status = str(
                result.get(
                    "status",
                    "ACCEPTED" if accepted else "REJECTED",
                )
            )

            action = str(
                result.get(
                    "action",
                    "ALLOW" if accepted else "BLOCK",
                )
            )

            reason = result.get("reason")

            nested_result = result.get(
                "result",
                result.get("data", {}),
            )

            if not isinstance(nested_result, Mapping):
                nested_result = {
                    "value": nested_result
                }

            return CommandResult(
                request_id=request.request_id,
                command=request.command,
                status=status,
                accepted=accepted,
                action=action,
                reason=(
                    str(reason)
                    if reason is not None
                    else None
                ),
                result=dict(nested_result),
            )

        raise TypeError(
            "Command callback must return a mapping, "
            "CommandResult, or None"
        )

    # ------------------------------------------------------------------
    # Protocol response
    # ------------------------------------------------------------------

    def _response(
        self,
        result: CommandResult,
    ) -> Dict[str, Any]:
        return build_response_message(
            request_message_id=result.request_id,
            status=result.status,
            result={
                "accepted": result.accepted,
                "action": result.action,
                "command": result.command,
                **result.result,
            },
            error=result.reason
            if result.status == "ERROR"
            else None,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def received_count(self) -> int:
        return self._received_count

    @property
    def accepted_count(self) -> int:
        return self._accepted_count

    @property
    def rejected_count(self) -> int:
        return self._rejected_count

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def last_request(self) -> Optional[CommandRequest]:
        return self._last_request

    @property
    def last_result(self) -> Optional[CommandResult]:
        return self._last_result

    def status(self) -> Dict[str, Any]:
        """Return diagnostic information."""
        return {
            "received_count": self._received_count,
            "accepted_count": self._accepted_count,
            "rejected_count": self._rejected_count,
            "error_count": self._error_count,
            "supported_commands": sorted(
                self.SUPPORTED_COMMANDS
            ),
            "max_bolus": self.max_bolus,
            "max_basal_rate": self.max_basal_rate,
            "last_request": (
                self._last_request.to_dict()
                if self._last_request
                else None
            ),
            "last_result": (
                self._last_result.to_dict()
                if self._last_result
                else None
            ),
        }


def create_command_handler(
    *,
    application_callback: Optional[CommandCallback] = None,
    max_bolus: float = CommandHandler.DEFAULT_MAX_BOLUS,
    max_basal_rate: float = CommandHandler.DEFAULT_MAX_BASAL_RATE,
    logger: Optional[logging.Logger] = None,
) -> CommandHandler:
    """Factory for the command handler."""
    return CommandHandler(
        application_callback=application_callback,
        max_bolus=max_bolus,
        max_basal_rate=max_basal_rate,
        logger=logger,
    )


async def demo_command_callback(
    message: Dict[str, Any],
    request: CommandRequest,
) -> Dict[str, Any]:
    """
    Demonstration callback.

    It demonstrates routing only. It does not perform pump actuation.
    """
    return {
        "accepted": False,
        "status": "REJECTED",
        "action": "BLOCK",
        "reason": (
            "Demo callback: no real pump actuation is implemented."
        ),
        "result": {
            "command": request.command,
            "value": request.value,
        },
    }


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(
        value,
        (str, bool, int),
    ):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Mapping):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(v)
            for v in value
        ]

    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    return str(value)


__all__ = [
    "CommandRequest",
    "CommandResult",
    "CommandHandler",
    "create_command_handler",
    "demo_command_callback",
]
