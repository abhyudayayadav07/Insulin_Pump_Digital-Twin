"""Policy mapping hazard evidence to mitigation actions."""
from dataclasses import dataclass
from typing import Optional

from .mitigation_action import (
    ActionPriority,
    MitigationAction,
    MitigationActionType,
)


@dataclass(frozen=True)
class MitigationPolicy:
    warning_threshold: float = 0.30
    active_threshold: float = 0.60
    critical_threshold: float = 0.85

    def select(self, hazard_score: float, attack_indicator: float = 0.0):
        score = max(0.0, min(1.0, float(hazard_score)))
        attack = max(0.0, min(1.0, float(attack_indicator)))

        if attack >= 0.5 or score >= self.critical_threshold:
            return MitigationAction(
                "MA_CRITICAL",
                MitigationActionType.EMERGENCY_STOP,
                ActionPriority.CRITICAL,
                "Critical hazard or active attack evidence.",
                deadline_minutes=5.0,
            )
        if score >= self.active_threshold:
            return MitigationAction(
                "MA_ACTIVE",
                MitigationActionType.USE_SAFE_TWIN,
                ActionPriority.HIGH,
                "Active hazard evidence requires predictive authority restriction.",
                deadline_minutes=15.0,
            )
        if score >= self.warning_threshold:
            return MitigationAction(
                "MA_WARNING",
                MitigationActionType.WARN,
                ActionPriority.MODERATE,
                "Warning-level hazard evidence.",
                deadline_minutes=30.0,
            )
        return MitigationAction(
            "MA_MONITOR",
            MitigationActionType.MONITOR,
            ActionPriority.LOW,
            "No active mitigation requirement.",
            deadline_minutes=None,
        )


def select_mitigation_action(
    hazard_score: float,
    attack_indicator: float = 0.0,
    policy: Optional[MitigationPolicy] = None,
):
    return (policy or MitigationPolicy()).select(hazard_score, attack_indicator)
