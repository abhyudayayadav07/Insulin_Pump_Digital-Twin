"""Hazard mitigation planning.

This module creates an abstract mitigation plan from DT2 evidence. It does
not directly calculate or command insulin doses.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .mitigation_action import MitigationAction, MitigationActionType
from .mitigation_deadline import estimate_mitigation_deadline
from .mitigation_policy import MitigationPolicy


@dataclass
class MitigationStep:
    step_id: str
    action: MitigationAction
    expected_duration_minutes: float = 0.0
    preconditions: Dict[str, Any] = field(default_factory=dict)
    exit_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MitigationPlan:
    plan_id: str
    steps: List[MitigationStep]
    deadline_minutes: Optional[float]
    hazard_score: float
    feasible: bool
    reason: str = ""


class MitigationPlanner:
    def __init__(self, policy: Optional[MitigationPolicy] = None):
        self.policy = policy or MitigationPolicy()

    def plan(
        self,
        hazard_score: float,
        attack_indicator: float = 0.0,
        future_times: Optional[Sequence[float]] = None,
        future_hazard_scores: Optional[Sequence[float]] = None,
    ) -> MitigationPlan:
        action = self.policy.select(hazard_score, attack_indicator)

        deadline = None
        if future_times is not None and future_hazard_scores is not None:
            result = estimate_mitigation_deadline(
                future_times, future_hazard_scores, self.policy.active_threshold
            )
            deadline = result.deadline_minutes

        if action.action_type == MitigationActionType.MONITOR:
            steps = [
                MitigationStep(
                    "STEP01", action, 0.0,
                    exit_conditions={"hazard_below": self.policy.warning_threshold},
                )
            ]
        elif action.action_type == MitigationActionType.WARN:
            steps = [
                MitigationStep(
                    "STEP01", action, 0.0,
                    exit_conditions={"hazard_below": self.policy.warning_threshold},
                ),
                MitigationStep(
                    "STEP02",
                    MitigationAction(
                        "MA_MONITOR",
                        MitigationActionType.MONITOR,
                        action.priority,
                        "Continue monitoring after warning.",
                    ),
                ),
            ]
        elif action.action_type == MitigationActionType.USE_SAFE_TWIN:
            steps = [
                MitigationStep("STEP01", action, 0.0),
                MitigationStep(
                    "STEP02",
                    MitigationAction(
                        "MA_MONITOR",
                        MitigationActionType.MONITOR,
                        action.priority,
                        "Monitor recovery and reassess hazard.",
                    ),
                ),
            ]
        else:
            steps = [MitigationStep("STEP01", action, 0.0)]

        feasible = deadline is None or action.deadline_minutes is None or deadline >= 0.0
        return MitigationPlan(
            plan_id="PLAN_DT2",
            steps=steps,
            deadline_minutes=deadline,
            hazard_score=float(hazard_score),
            feasible=feasible,
            reason="Plan generated from hazard and integrity evidence.",
        )


def create_mitigation_plan(
    hazard_score: float,
    attack_indicator: float = 0.0,
    future_times=None,
    future_hazard_scores=None,
):
    return MitigationPlanner().plan(
        hazard_score,
        attack_indicator,
        future_times,
        future_hazard_scores,
    )
