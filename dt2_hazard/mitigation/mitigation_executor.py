"""Stateful abstract mitigation-plan executor."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from .mitigation_action import MitigationActionType
from .mitigation_planner import MitigationPlan


class ExecutionState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class ExecutionEvent:
    step_id: str
    state: ExecutionState
    message: str
    timestamp_minutes: Optional[float] = None


@dataclass
class MitigationExecutor:
    state: ExecutionState = ExecutionState.IDLE
    current_step: int = 0
    events: list[ExecutionEvent] = field(default_factory=list)

    def start(self, plan: MitigationPlan, timestamp_minutes: Optional[float] = None):
        if not plan.feasible or not plan.steps:
            self.state = ExecutionState.ABORTED
            return
        self.state = ExecutionState.ACTIVE
        self.current_step = 0
        self.events.append(
            ExecutionEvent(
                plan.steps[0].step_id,
                self.state,
                "Mitigation plan started.",
                timestamp_minutes,
            )
        )

    def advance(
        self,
        plan: MitigationPlan,
        context: Optional[Dict[str, Any]] = None,
        timestamp_minutes: Optional[float] = None,
    ):
        if self.state != ExecutionState.ACTIVE:
            return None

        context = context or {}
        step = plan.steps[self.current_step]

        if context.get("abort", False):
            self.state = ExecutionState.ABORTED
            message = "Mitigation execution aborted."
        else:
            action_type = step.action.action_type
            if action_type == MitigationActionType.EMERGENCY_STOP:
                self.state = ExecutionState.COMPLETED
                message = "Emergency response step reached."
            elif context.get("target_reached", False):
                self.state = ExecutionState.COMPLETED
                message = "Target recovery condition reached."
            elif self.current_step + 1 < len(plan.steps):
                self.current_step += 1
                message = "Advanced to next mitigation step."
            else:
                self.state = ExecutionState.COMPLETED
                message = "Mitigation plan completed."

        self.events.append(
            ExecutionEvent(step.step_id, self.state, message, timestamp_minutes)
        )
        return self.state
