"""Mitigation action definitions.

Actions describe safety responses without directly implementing a clinical
insulin-dosing policy.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MitigationActionType(str, Enum):
    CONTINUE = "continue"
    MONITOR = "monitor"
    WARN = "warn"
    BLOCK_PREDICTIVE = "block_predictive"
    USE_SAFE_TWIN = "use_safe_twin"
    ENTER_SAFE_MODE = "enter_safe_mode"
    EMERGENCY_STOP = "emergency_stop"


class ActionPriority(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class MitigationAction:
    action_id: str
    action_type: MitigationActionType
    priority: ActionPriority
    reason: str = ""
    deadline_minutes: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
