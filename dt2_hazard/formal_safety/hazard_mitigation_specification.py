"""Hazard Mitigation Specifications (HMS).

HMS encode context-dependent requirements for responding to predicted hazards.
They constrain the mitigation response rather than directly prescribing a
clinical insulin dose.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional


class MitigationLevel(str, Enum):
    MONITOR = "monitor"
    WARNING = "warning"
    ACTIVE = "active"
    CRITICAL = "critical"


class RequiredResponse(str, Enum):
    CONTINUE_MONITORING = "continue_monitoring"
    ISSUE_WARNING = "issue_warning"
    BLOCK_PREDICTIVE = "block_predictive"
    USE_SAFE_TWIN = "use_safe_twin"
    ENTER_SAFE_MODE = "enter_safe_mode"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True)
class HazardMitigationSpecification:
    hms_id: str
    name: str
    hazard_id: str
    minimum_level: MitigationLevel
    required_response: RequiredResponse
    deadline_minutes: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def applies(self, hazard_level: MitigationLevel) -> bool:
        order = {
            MitigationLevel.MONITOR: 0,
            MitigationLevel.WARNING: 1,
            MitigationLevel.ACTIVE: 2,
            MitigationLevel.CRITICAL: 3,
        }
        return order[hazard_level] >= order[self.minimum_level]


def default_hms():
    return [
        HazardMitigationSpecification(
            "HMS01", "Monitor emerging glucose hazard", "E01",
            MitigationLevel.MONITOR, RequiredResponse.CONTINUE_MONITORING,
            deadline_minutes=30.0,
        ),
        HazardMitigationSpecification(
            "HMS02", "Warn on active hazard", "E01",
            MitigationLevel.ACTIVE, RequiredResponse.USE_SAFE_TWIN,
            deadline_minutes=15.0,
        ),
        HazardMitigationSpecification(
            "HMS03", "Enter safe mode for critical hazard", "E01",
            MitigationLevel.CRITICAL, RequiredResponse.ENTER_SAFE_MODE,
            deadline_minutes=5.0,
        ),
    ]
