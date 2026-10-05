"""
Physiological safety constraints for DT2.

Default ranges are research/configuration defaults, not universal clinical
limits. They are intentionally configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .constraint_definition import (
    ConstraintSeverity,
    ConstraintType,
    SafetyConstraint,
)


@dataclass(frozen=True)
class PhysiologicalConstraintConfig:
    glucose_min: float = 20.0
    glucose_max: float = 600.0
    warning_low: float = 70.0
    warning_high: float = 180.0
    max_abs_glucose_rate: float = 10.0
    max_abs_glucose_acceleration: float = 5.0
    insulin_min: float = 0.0
    meal_min: float = 0.0


def default_physiological_constraints(
    config: Optional[PhysiologicalConstraintConfig] = None,
):
    config = config or PhysiologicalConstraintConfig()

    return [
        SafetyConstraint(
            "PHYS01",
            "Glucose physical range",
            "Glucose must remain within the configured physiological simulation range.",
            ConstraintType.RANGE,
            ConstraintSeverity.CRITICAL,
            checker=lambda x: (
                np.isfinite(float(x))
                and config.glucose_min <= float(x) <= config.glucose_max
            ),
            lower=config.glucose_min,
            upper=config.glucose_max,
        ),
        SafetyConstraint(
            "PHYS02",
            "Glucose rate limit",
            "Absolute glucose rate should remain within the configured bound.",
            ConstraintType.STATE,
            ConstraintSeverity.MAJOR,
            checker=lambda x: abs(float(x)) <= config.max_abs_glucose_rate,
            upper=config.max_abs_glucose_rate,
        ),
        SafetyConstraint(
            "PHYS03",
            "Glucose acceleration limit",
            "Absolute glucose acceleration should remain within the configured bound.",
            ConstraintType.TRAJECTORY,
            ConstraintSeverity.MAJOR,
            checker=lambda x: abs(float(x)) <= config.max_abs_glucose_acceleration,
            upper=config.max_abs_glucose_acceleration,
        ),
        SafetyConstraint(
            "PHYS04",
            "Insulin non-negative",
            "Insulin quantity/rate cannot be negative.",
            ConstraintType.CONTROL,
            ConstraintSeverity.CRITICAL,
            checker=lambda x: float(x) >= config.insulin_min,
            lower=config.insulin_min,
        ),
        SafetyConstraint(
            "PHYS05",
            "Meal input non-negative",
            "Meal carbohydrate input cannot be negative.",
            ConstraintType.STATE,
            ConstraintSeverity.WARNING,
            checker=lambda x: float(x) >= config.meal_min,
            lower=config.meal_min,
        ),
    ]


def evaluate_glucose_state(
    glucose: float,
    *,
    config: Optional[PhysiologicalConstraintConfig] = None,
):
    """Evaluate the configured physical glucose range."""
    config = config or PhysiologicalConstraintConfig()
    constraint = default_physiological_constraints(config)[0]
    return constraint.evaluate(glucose)


__all__ = [
    "PhysiologicalConstraintConfig",
    "default_physiological_constraints",
    "evaluate_glucose_state",
]
