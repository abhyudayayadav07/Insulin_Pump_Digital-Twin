"""Structured output from the DT2 mitigation layer."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .mitigation_action import MitigationAction
from .mitigation_planner import MitigationPlan
from .mitigation_state import MitigationState


@dataclass
class MitigationOutput:
    state: MitigationState
    action: MitigationAction
    plan: MitigationPlan
    deadline_minutes: Optional[float]
    authority_restricted: bool
    safe_twin_recommended: bool
    emergency: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def predictive_authority_allowed(self) -> bool:
        return not self.authority_restricted
