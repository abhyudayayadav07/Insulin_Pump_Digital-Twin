"""
IITP Digital Twin - Communication package.

Public communication-layer exports for the Omniverse insulin-pump
digital-twin extension.

The communication layer is responsible for moving telemetry and commands
between the external Python digital-twin process and Omniverse.

Current planned modules:

    web_socket.py
        WebSocket transport between Python DT and Omniverse.

    telemetry_handler.py
        Parsing, normalization, storage, and callback dispatch for
        telemetry messages.

The package intentionally does not contain safety or medical decision logic.
Those remain in the external digital-twin/security/decision system.

Example:

    from communication import (
        DigitalTwinWebSocketClient,
        TelemetryHandler,
    )

Note:
    The actual extension package namespace may later be changed to
    iitp.digital_twin.communication depending on the final extension
    directory layout.
"""

from __future__ import annotations

from .web_socket import DigitalTwinWebSocketClient
from .telemetry_handler import TelemetryHandler, create_telemetry_handler

__all__ = [
    "DigitalTwinWebSocketClient",
    "TelemetryHandler",
    "create_telemetry_handler",
]

__version__ = "0.1.0"
