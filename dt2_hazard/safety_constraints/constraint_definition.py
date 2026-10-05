"""
Core safety-constraint definitions for DT2.

A safety constraint expresses a condition that should remain satisfied.
Constraints are deliberately separate from hazards: a hazard describes an
undesired event/state, while a constraint describes what must be prevented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np


class ConstraintType(str, Enum):
    STATE = "state"
    CONTROL = "control"
    TIMING = "timing"
    RANGE = "range"
    AUTHENTICATION = "authentication"
    COMMUNICATION = "communication"
    SENSOR_VALIDATION = "sensor_validation"
    ACTUATION = "actuation"
    SAFETY_RESPONSE = "safety_response"
    CONFIGURATION = "configuration"
    MODEL_CONSISTENCY = "model_consistency"
    TRAJECTORY = "trajectory"


class ConstraintSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    MAJOR = "major"
    CRITICAL = "critical"


@dataclass
class ConstraintEvaluation:
    constraint_id: str
    satisfied: bool
    severity: ConstraintSeverity
    score: float
    message: str = ""
    observed_value: Any = None
    expected_value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyConstraint:
    constraint_id: str
    name: str
    description: str
    constraint_type: ConstraintType
    severity: ConstraintSeverity = ConstraintSeverity.MAJOR
    checker: Optional[Callable[[Any], bool]] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, value: Any) -> ConstraintEvaluation:
        if self.checker is not None:
            satisfied = bool(self.checker(value))
        else:
            scalar = float(np.asarray(value).reshape(-1)[0])
            satisfied = True
            if self.lower is not None:
                satisfied &= scalar >= self.lower
            if self.upper is not None:
                satisfied &= scalar <= self.upper

        score = 0.0 if satisfied else 1.0
        return ConstraintEvaluation(
            constraint_id=self.constraint_id,
            satisfied=satisfied,
            severity=self.severity,
            score=score,
            message=(
                f"{self.name}: satisfied"
                if satisfied
                else f"{self.name}: violated"
            ),
            observed_value=value,
            expected_value={"lower": self.lower, "upper": self.upper},
            metadata=dict(self.metadata),
        )


class ConstraintRegistry:
    """Registry for named DT2 safety constraints."""

    def __init__(self, constraints: Optional[Sequence[SafetyConstraint]] = None):
        self._constraints: Dict[str, SafetyConstraint] = {}
        for constraint in constraints or []:
            self.register(constraint)

    def register(self, constraint: SafetyConstraint) -> None:
        if constraint.constraint_id in self._constraints:
            raise ValueError(
                f"Constraint already registered: {constraint.constraint_id}"
            )
        self._constraints[constraint.constraint_id] = constraint

    def get(self, constraint_id: str) -> SafetyConstraint:
        return self._constraints[constraint_id]

    def all(self):
        return list(self._constraints.values())

    def ids(self):
        return tuple(self._constraints.keys())


__all__ = [
    "ConstraintType",
    "ConstraintSeverity",
    "ConstraintEvaluation",
    "SafetyConstraint",
    "ConstraintRegistry",
]
