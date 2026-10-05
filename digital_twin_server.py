"""
IITP Digital Twin - Digital Twin Server.

External Python runtime that orchestrates:
    simulator -> DT1 -> DT2 -> security -> decision -> telemetry -> WebSocket.

It supports a DEMO mode so the Omniverse/WebSocket pipeline can be tested
before SimGlucose and the real DT1/DT2 modules are connected.

Omniverse is a client of this server; it is not the safety authority.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import math
import signal
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

# digital_twin_server.py is the application entry point, while the
# communication modules live inside the communication package. Therefore
# these MUST be absolute imports when this file is run with:
#     python digital_twin_server.py
from communication.connection_manager import (
    ConnectionManager,
    ClientConnection,
)
from communication.message_schema import (
    TYPE_COMMAND,
    build_response_message,
    get_message_id,
    validate_message,
)


Callback = Callable[..., Any]


@dataclass
class DigitalTwinServerConfig:
    """Runtime configuration."""

    host: str = "127.0.0.1"
    port: int = 8765
    sample_time_seconds: float = 1.0
    simulation_step_minutes: float = 5.0

    demo_mode: bool = True
    demo_glucose: float = 108.0
    demo_cgm: float = 106.5
    demo_insulin_rate: float = 1.20
    demo_meal: float = 0.0

    max_clients: int = 4
    heartbeat_interval: float = 15.0
    heartbeat_timeout: float = 60.0


def _get_command(message: Mapping[str, Any]) -> Optional[str]:
    """Return the command from either the flat or normalized payload form."""
    payload = message.get("payload")
    if isinstance(payload, Mapping) and payload.get("command") is not None:
        return str(payload.get("command"))
    value = message.get("command")
    return str(value) if value is not None else None


class DigitalTwinServer:
    """
    Main external digital-twin runtime.

    Real processing modules can be supplied as callbacks:

        simulator(state_context) -> state
        dt1_processor(state, context) -> dt1_result
        dt2_processor(state, context) -> dt2_result
        security_processor(state, context) -> security_result
        decision_processor(state, context) -> decision_result
        command_handler(message, connection) -> response

    Until those callbacks are supplied, neutral demo results are used.
    """

    def __init__(
        self,
        config: Optional[DigitalTwinServerConfig] = None,
        *,
        connection_manager: Optional[ConnectionManager] = None,
        simulator: Optional[Callback] = None,
        dt1_processor: Optional[Callback] = None,
        dt2_processor: Optional[Callback] = None,
        security_processor: Optional[Callback] = None,
        decision_processor: Optional[Callback] = None,
        command_handler: Optional[Callback] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or DigitalTwinServerConfig()
        self.logger = logger or logging.getLogger(
            "iitp.digital_twin.server"
        )

        self.simulator = simulator
        self.dt1_processor = dt1_processor
        self.dt2_processor = dt2_processor
        self.security_processor = security_processor
        self.decision_processor = decision_processor
        self.application_command_handler = command_handler

        self.connection_manager = connection_manager or ConnectionManager(
            host=self.config.host,
            port=self.config.port,
            max_clients=self.config.max_clients,
            heartbeat_interval=self.config.heartbeat_interval,
            heartbeat_timeout=self.config.heartbeat_timeout,
            logger=self.logger,
            on_command=self._handle_command,
        )

        self._running = False
        self._stopping = False
        self._task: Optional[asyncio.Task] = None

        self._simulation_time_minutes = 0.0
        self._step_count = 0
        self._last_telemetry: Optional[Dict[str, Any]] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def endpoint(self) -> str:
        return self.connection_manager.endpoint

    @property
    def simulation_time_minutes(self) -> float:
        return self._simulation_time_minutes

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def last_telemetry(self) -> Optional[Dict[str, Any]]:
        return dict(self._last_telemetry) if self._last_telemetry else None

    async def start(self, *, run_simulation: bool = True) -> None:
        """Start WebSocket communication and optionally the simulation loop."""
        if self._running:
            return

        self._stopping = False
        await self.connection_manager.start()
        self._running = True

        self.logger.info(
            "Digital Twin Server started at %s",
            self.endpoint,
        )

        if run_simulation:
            self._task = asyncio.create_task(
                self.run_simulation_loop(),
                name="iitp-digital-twin-loop",
            )

    async def stop(self) -> None:
        """Stop the simulation and WebSocket server."""
        if not self._running:
            return

        self._stopping = True

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                self.logger.exception("Simulation task shutdown failed")
            self._task = None

        await self.connection_manager.stop()
        self._running = False
        self._stopping = False
        self.logger.info("Digital Twin Server stopped")

    async def __aenter__(self) -> "DigitalTwinServer":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.stop()

    async def run_simulation_loop(self) -> None:
        """Continuously generate one telemetry message per simulation step."""
        self.logger.info(
            "Simulation loop started; demo_mode=%s",
            self.config.demo_mode,
        )

        while self._running and not self._stopping:
            started = time.perf_counter()

            try:
                await self.step()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("Digital-twin step failed")

            elapsed = time.perf_counter() - started
            await asyncio.sleep(
                max(0.0, self.config.sample_time_seconds - elapsed)
            )

    async def step(self) -> Dict[str, Any]:
        """Execute one complete simulator -> DT -> telemetry cycle."""
        state = await self._get_state()

        dt1 = await self._processor(
            self.dt1_processor, state, {}, "DT1"
        )
        dt2 = await self._processor(
            self.dt2_processor, state, {"dt1": dt1}, "DT2"
        )
        security = await self._processor(
            self.security_processor,
            state,
            {"dt1": dt1, "dt2": dt2},
            "security",
        )
        decision = await self._processor(
            self.decision_processor,
            state,
            {
                "dt1": dt1,
                "dt2": dt2,
                "security": security,
            },
            "decision",
        )

        telemetry = self._build_telemetry(
            state, dt1, dt2, security, decision
        )

        validate_message(telemetry)
        self._last_telemetry = telemetry

        await self.connection_manager.broadcast_telemetry(telemetry)

        self._step_count += 1
        self._simulation_time_minutes += self.config.simulation_step_minutes

        return telemetry

    async def _get_state(self) -> Dict[str, Any]:
        if self.simulator is not None:
            result = self.simulator({
                "simulation_time_minutes": self._simulation_time_minutes,
                "step_count": self._step_count,
            })
            result = await _maybe_await(result)
            if result is not None:
                return self._normalize_state(result)

        return self._demo_state()

    def _demo_state(self) -> Dict[str, Any]:
        """Deterministic communication-test state, not a physiological model."""
        t = self._simulation_time_minutes
        glucose = self.config.demo_glucose + 8.0 * math.sin(t / 45.0)
        cgm = glucose - 1.5 + math.sin(t / 30.0)

        meal = 0.0
        if 20.0 <= (t % 120.0) < 35.0:
            meal = 30.0

        insulin = max(
            0.0,
            self.config.demo_insulin_rate
            + 0.10 * math.sin(t / 60.0),
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "simulation_time": t,
            "patient": {
                "glucose": round(glucose, 3),
                "cgm": round(cgm, 3),
                "meal": meal,
            },
            "pump": {
                "insulin_rate": round(insulin, 3),
                "command": round(insulin, 3),
                "delivery": round(insulin, 3),
                "status": "NORMAL",
            },
        }

    @staticmethod
    def _normalize_state(state: Mapping[str, Any]) -> Dict[str, Any]:
        """Accept either nested or flat simulator output."""
        data = dict(state)
        patient = dict(data.get("patient") or {})
        pump = dict(data.get("pump") or {})

        for key in ("glucose", "cgm", "meal"):
            if key in data:
                patient.setdefault(key, data[key])

        if "insulin" in data:
            pump.setdefault("insulin_rate", data["insulin"])
        if "insulin_rate" in data:
            pump.setdefault("insulin_rate", data["insulin_rate"])
        for key in ("command", "delivery", "status"):
            if key in data:
                pump.setdefault(key, data[key])

        pump.setdefault("status", "NORMAL")
        data["patient"] = patient
        data["pump"] = pump
        data.setdefault(
            "timestamp",
            datetime.now(timezone.utc).isoformat(),
        )
        data.setdefault(
            "simulation_time",
            data.get("simulation_time_minutes", 0.0),
        )
        return data

    async def _processor(
        self,
        processor: Optional[Callback],
        state: Dict[str, Any],
        context: Dict[str, Any],
        name: str,
    ) -> Dict[str, Any]:
        if processor is None:
            return self._default_result(name, state)

        result = processor(state, context)
        result = await _maybe_await(result)

        if result is None:
            return self._default_result(name, state)

        if not isinstance(result, Mapping):
            raise TypeError(
                f"{name} processor must return a mapping"
            )

        return dict(result)

    @staticmethod
    def _default_result(
        name: str,
        state: Mapping[str, Any],
    ) -> Dict[str, Any]:
        glucose = float(
            (state.get("patient") or {}).get("glucose", 0.0)
        )

        if name == "DT1":
            return {
                "ml_prediction": glucose,
                "geco_prediction": glucose,
                "fused_prediction": glucose,
                "residual": 0.0,
                "cusum": 0.0,
                "status": "DEMO",
            }

        if name == "DT2":
            return {
                "hazard": 0.0,
                "survival": 1.0,
                "event_probability": 0.0,
                "time_to_event": None,
                "risk_level": "NORMAL",
                "status": "DEMO",
            }

        if name == "security":
            return {
                "attack_detected": False,
                "attack_type": None,
                "sensor_integrity": 1.0,
                "pump_integrity": 1.0,
                "status": "NORMAL",
            }

        if name == "decision":
            return {
                "action": "ALLOW",
                "status": "DEMO",
            }

        return {"status": "DEMO"}

    def _build_telemetry(
        self,
        state: Mapping[str, Any],
        dt1: Mapping[str, Any],
        dt2: Mapping[str, Any],
        security: Mapping[str, Any],
        decision: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the canonical telemetry schema used by our Omniverse bridge.

        This implementation intentionally does not depend on a particular
        TelemetrySerializer API, so it can be connected to the serializer
        already generated in the communication package later.
        """
        patient = dict(state.get("patient") or {})
        pump = dict(state.get("pump") or {})

        telemetry = {
            "type": "telemetry",
            "timestamp": state.get(
                "timestamp",
                datetime.now(timezone.utc).isoformat(),
            ),
            "simulation_time": state.get(
                "simulation_time",
                self._simulation_time_minutes,
            ),
            "patient": {
                "glucose": _number(patient.get("glucose"), 0.0),
                "cgm": _number(patient.get("cgm"), 0.0),
                "meal": _number(patient.get("meal"), 0.0),
            },
            "pump": {
                "insulin_rate": _number(
                    pump.get("insulin_rate"), 0.0
                ),
                "command": _number(
                    pump.get("command"), 0.0
                ),
                "delivery": _number(
                    pump.get("delivery"), 0.0
                ),
                "status": pump.get("status", "NORMAL"),
            },
            "dt1": dict(dt1),
            "dt2": dict(dt2),
            "security": dict(security),
            "decision": dict(decision),
        }

        return _json_safe(telemetry)

    async def _handle_command(
        self,
        message: Dict[str, Any],
        connection: ClientConnection,
    ) -> Optional[Dict[str, Any]]:
        """Route Omniverse commands to the application/safety layer."""
        if message.get("type") != TYPE_COMMAND:
            return None

        command = _get_command(message)
        request_id = get_message_id(message)

        self.logger.info(
            "Command received: %s from %s",
            command,
            connection.connection_id,
        )

        if self.application_command_handler is not None:
            try:
                result = self.application_command_handler(
                    message, connection
                )
                result = await _maybe_await(result)
                if result is not None:
                    return result
            except Exception as exc:
                self.logger.exception("Command handler failed")
                if request_id:
                    return build_response_message(
                        request_message_id=request_id,
                        status="ERROR",
                        error=str(exc),
                    )
                return None

        # Safety rule: until the real command handler is connected, commands
        # are acknowledged as RECEIVED only. No insulin actuation occurs.
        if request_id:
            return build_response_message(
                request_message_id=request_id,
                status="RECEIVED",
                result={
                    "accepted_for_processing": False,
                    "command": command,
                    "reason": (
                        "Safety/decision command handler is not connected; "
                        "no pump actuation performed."
                    ),
                },
            )

        return None

    def status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "endpoint": self.endpoint,
            "demo_mode": self.config.demo_mode,
            "simulation_time_minutes": self._simulation_time_minutes,
            "step_count": self._step_count,
            "client_count": self.connection_manager.client_count,
            "connection_manager": self.connection_manager.status(),
        }


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_safe(value: Any) -> Any:
    """Recursively convert common Python/numpy-like values to JSON-safe data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    return str(value)


async def run_demo_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    interval: float = 1.0,
) -> None:
    """Run the communication-only demo server."""
    config = DigitalTwinServerConfig(
        host=host,
        port=port,
        sample_time_seconds=interval,
        demo_mode=True,
    )
    server = DigitalTwinServer(config)

    stop_event = asyncio.Event()

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except (NotImplementedError, RuntimeError):
                pass
    except RuntimeError:
        pass

    await server.start()

    print("IITP Digital Twin demo server running")
    print(f"WebSocket endpoint: {server.endpoint}")
    print("Waiting for an Omniverse client...")

    try:
        await stop_event.wait()
    finally:
        await server.stop()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IITP Insulin Digital Twin WebSocket server"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between demo telemetry messages",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        asyncio.run(
            run_demo_server(
                host=args.host,
                port=args.port,
                interval=args.interval,
            )
        )
    except KeyboardInterrupt:
        pass


__all__ = [
    "DigitalTwinServerConfig",
    "DigitalTwinServer",
    "run_demo_server",
    "build_argument_parser",
    "main",
]


if __name__ == "__main__":
    main()
