"""State machine for DT2 mitigation lifecycle."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MitigationState(str, Enum):
    INACTIVE = "inactive"
    MONITORING = "monitoring"
    WARNING = "warning"
    ACTIVE = "active"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    COMPLETE = "complete"
    ABORTED = "aborted"


@dataclass
class MitigationStateMachine:
    state: MitigationState = MitigationState.INACTIVE
    entered_at_minutes: Optional[float] = None

    def transition(self, hazard_score: float, attack_indicator: float = 0.0,
                   recovery: bool = False, timestamp_minutes: Optional[float] = None):
        score = float(hazard_score)
        attack = float(attack_indicator)

        if attack >= 0.5 or score >= 0.85:
            target = MitigationState.CRITICAL
        elif recovery and score < 0.30:
            target = MitigationState.COMPLETE
        elif recovery:
            target = MitigationState.RECOVERING
        elif score >= 0.60:
            target = MitigationState.ACTIVE
        elif score >= 0.30:
            target = MitigationState.WARNING
        elif score > 0.0:
            target = MitigationState.MONITORING
        else:
            target = MitigationState.INACTIVE

        if target != self.state:
            self.state = target
            self.entered_at_minutes = timestamp_minutes
        return self.state
