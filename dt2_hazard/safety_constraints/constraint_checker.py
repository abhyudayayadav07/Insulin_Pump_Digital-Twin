"""
Unified safety-constraint checker for DT2.

This module combines individual state, trajectory, and control
constraints into a structured evidence object for downstream safety
reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from .constraint_definition import (
    ConstraintEvaluation,
    ConstraintRegistry,
    SafetyConstraint,
)
from .physiological_constraints import (
    PhysiologicalConstraintConfig,
    default_physiological_constraints,
)
from .trajectory_constraints import (
    TrajectoryConstraintConfig,
    check_trajectory,
)
from .control_constraints import (
    ControlConstraintConfig,
    check_command_sequence,
)


@dataclass
class SafetyConstraintReport:
    """Aggregated safety-constraint verification report."""

    all_satisfied: bool
    violation_count: int
    critical_violation: bool
    evaluations: list = field(default_factory=list)
    trajectory_result: Any = None
    control_result: Any = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SafetyConstraintChecker:
    """Evaluate DT2 safety constraints across state and trajectory data."""

    def __init__(
        self,
        registry: Optional[ConstraintRegistry] = None,
        *,
        physiological_config: Optional[PhysiologicalConstraintConfig] = None,
        trajectory_config: Optional[TrajectoryConstraintConfig] = None,
        control_config: Optional[ControlConstraintConfig] = None,
    ) -> None:
        self.registry = registry or ConstraintRegistry(
            default_physiological_constraints(physiological_config)
        )
        self.trajectory_config = (
            trajectory_config or TrajectoryConstraintConfig()
        )
        self.control_config = control_config or ControlConstraintConfig()

    def evaluate_state(
        self,
        *,
        glucose: Optional[float] = None,
        glucose_rate: Optional[float] = None,
        glucose_acceleration: Optional[float] = None,
        insulin: Optional[float] = None,
        meal: Optional[float] = None,
    ) -> list:
        """Evaluate available scalar state constraints."""
        values = {
            "PHYS01": glucose,
            "PHYS02": glucose_rate,
            "PHYS03": glucose_acceleration,
            "PHYS04": insulin,
            "PHYS05": meal,
        }

        results = []
        for constraint_id, value in values.items():
            if value is None:
                continue
            results.append(self.registry.get(constraint_id).evaluate(value))
        return results

    def evaluate(
        self,
        *,
        glucose: Optional[float] = None,
        glucose_rate: Optional[float] = None,
        glucose_acceleration: Optional[float] = None,
        insulin: Optional[float] = None,
        meal: Optional[float] = None,
        predicted_glucose: Optional[Sequence[float]] = None,
        insulin_commands: Optional[Sequence[float]] = None,
    ) -> SafetyConstraintReport:
        """Run state, trajectory, and control checks."""
        evaluations = self.evaluate_state(
            glucose=glucose,
            glucose_rate=glucose_rate,
            glucose_acceleration=glucose_acceleration,
            insulin=insulin,
            meal=meal,
        )

        trajectory_result = None
        if predicted_glucose is not None:
            trajectory_result = check_trajectory(
                predicted_glucose,
                config=self.trajectory_config,
            )

        control_result = None
        if insulin_commands is not None:
            control_result = check_command_sequence(
                insulin_commands,
                config=self.control_config,
            )

        violations = sum(
            not evaluation.satisfied for evaluation in evaluations
        )
        if trajectory_result is not None and not trajectory_result.satisfied:
            violations += 1
        if control_result is not None and not control_result.satisfied:
            violations += 1

        critical = any(
            not evaluation.satisfied
            and evaluation.severity.value == "critical"
            for evaluation in evaluations
        )

        all_satisfied = violations == 0
        total_checks = (
            len(evaluations)
            + (1 if trajectory_result is not None else 0)
            + (1 if control_result is not None else 0)
        )
        score = violations / total_checks if total_checks else 0.0

        return SafetyConstraintReport(
            all_satisfied=all_satisfied,
            violation_count=violations,
            critical_violation=critical,
            evaluations=evaluations,
            trajectory_result=trajectory_result,
            control_result=control_result,
            score=float(score),
        )


def check_safety_constraints(
    *,
    glucose: Optional[float] = None,
    glucose_rate: Optional[float] = None,
    glucose_acceleration: Optional[float] = None,
    insulin: Optional[float] = None,
    meal: Optional[float] = None,
    predicted_glucose: Optional[Sequence[float]] = None,
    insulin_commands: Optional[Sequence[float]] = None,
) -> SafetyConstraintReport:
    """Convenience function for one-shot constraint verification."""
    return SafetyConstraintChecker().evaluate(
        glucose=glucose,
        glucose_rate=glucose_rate,
        glucose_acceleration=glucose_acceleration,
        insulin=insulin,
        meal=meal,
        predicted_glucose=predicted_glucose,
        insulin_commands=insulin_commands,
    )


__all__ = [
    "SafetyConstraintReport",
    "SafetyConstraintChecker",
    "check_safety_constraints",
]
