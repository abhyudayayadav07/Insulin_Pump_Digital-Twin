"""Formal safety specification primitives.

This module represents safety requirements independently from any specific
runtime decision engine. Specifications can be evaluated against current
state, predicted state, and control/action context.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional


class SpecificationType(str, Enum):
    STATE = "state"
    TRAJECTORY = "trajectory"
    CONTROL = "control"
    TIMING = "timing"
    TRANSITION = "transition"
    COMMUNICATION = "communication"
    INTEGRITY = "integrity"


class SpecificationSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class SafetySpecification:
    spec_id: str
    name: str
    description: str
    specification_type: SpecificationType
    severity: SpecificationSeverity = SpecificationSeverity.HIGH
    predicate: Optional[Callable[[Mapping[str, Any]], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, context: Mapping[str, Any]) -> bool:
        if self.predicate is None:
            return True
        return bool(self.predicate(context))


@dataclass
class SpecificationEvaluation:
    spec_id: str
    satisfied: bool
    severity: SpecificationSeverity
    message: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


class SafetySpecificationRegistry:
    def __init__(self, specifications=None):
        self._items: Dict[str, SafetySpecification] = {}
        for spec in specifications or []:
            self.register(spec)

    def register(self, specification: SafetySpecification):
        if specification.spec_id in self._items:
            raise ValueError(f"Duplicate specification: {specification.spec_id}")
        self._items[specification.spec_id] = specification

    def get(self, spec_id: str) -> SafetySpecification:
        return self._items[spec_id]

    def all(self):
        return list(self._items.values())

    def ids(self):
        return list(self._items.keys())


def evaluate_specification(specification, context):
    satisfied = specification.evaluate(context)
    return SpecificationEvaluation(
        spec_id=specification.spec_id,
        satisfied=satisfied,
        severity=specification.severity,
        message="satisfied" if satisfied else "violated",
    )
