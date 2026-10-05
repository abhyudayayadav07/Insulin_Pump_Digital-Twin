"""Fault models used for DT2 robustness analysis."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class FaultType(str, Enum):
    SENSOR = "sensor"
    CONTROLLER = "controller"
    PUMP = "pump"
    COMMUNICATION = "communication"
    MODEL = "model"
    CYBER = "cyber"
    ENVIRONMENT = "environment"


@dataclass(frozen=True)
class FaultScenario:
    scenario_id: str
    fault_type: FaultType
    severity: float
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def normalized_severity(self) -> float:
        return max(0.0, min(1.0, float(self.severity)))


def default_fault_scenarios():
    return [
        FaultScenario("F01", FaultType.SENSOR, 0.25, "Moderate CGM bias"),
        FaultScenario("F02", FaultType.SENSOR, 0.50, "Large CGM dropout"),
        FaultScenario("F03", FaultType.CONTROLLER, 0.50, "Insulin command bias"),
        FaultScenario("F04", FaultType.PUMP, 0.50, "Delivery scaling fault"),
        FaultScenario("F05", FaultType.COMMUNICATION, 0.50, "Intermittent communication"),
        FaultScenario("F06", FaultType.MODEL, 0.25, "Model parameter mismatch"),
        FaultScenario("F07", FaultType.CYBER, 0.75, "Integrity attack indicator"),
    ]
