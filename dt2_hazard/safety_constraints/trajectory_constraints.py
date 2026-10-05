"""
Trajectory-level safety constraints for DT2.

These constraints inspect a future glucose trajectory rather than one
isolated state. They are useful for detecting sustained movement toward
unsafe regions and excessive rates of change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class TrajectoryConstraintConfig:
    glucose_min: float = 70.0
    glucose_max: float = 180.0
    hard_min: float = 20.0
    hard_max: float = 600.0
    max_abs_rate: float = 10.0
    max_abs_acceleration: float = 5.0
    max_consecutive_outside: int = 2
    dt_minutes: float = 5.0


@dataclass
class TrajectoryConstraintResult:
    satisfied: bool
    score: float
    violations: Dict[str, object]


def _rates(glucose: np.ndarray, dt: float):
    if len(glucose) < 2:
        return np.zeros_like(glucose)
    return np.gradient(glucose, dt)


def check_trajectory(
    glucose: Sequence[float],
    *,
    config: Optional[TrajectoryConstraintConfig] = None,
) -> TrajectoryConstraintResult:
    """Check a predicted glucose trajectory against configured constraints."""
    config = config or TrajectoryConstraintConfig()
    g = np.asarray(glucose, dtype=float).reshape(-1)

    if len(g) == 0:
        return TrajectoryConstraintResult(
            satisfied=False,
            score=1.0,
            violations={"empty_trajectory": True},
        )

    if not np.isfinite(g).all():
        return TrajectoryConstraintResult(
            satisfied=False,
            score=1.0,
            violations={"non_finite": True},
        )

    rate = _rates(g, config.dt_minutes)
    acceleration = _rates(rate, config.dt_minutes)

    violations = {}

    if np.any(g < config.hard_min) or np.any(g > config.hard_max):
        violations["hard_range"] = True

    outside = (g < config.glucose_min) | (g > config.glucose_max)
    if np.any(outside):
        longest = 0
        current = 0
        for flag in outside:
            current = current + 1 if flag else 0
            longest = max(longest, current)
        if longest > config.max_consecutive_outside:
            violations["persistent_outside_target"] = int(longest)

    if np.any(np.abs(rate) > config.max_abs_rate):
        violations["rate_limit"] = float(np.max(np.abs(rate)))

    if np.any(np.abs(acceleration) > config.max_abs_acceleration):
        violations["acceleration_limit"] = float(
            np.max(np.abs(acceleration))
        )

    score = min(
        1.0,
        len(violations) / 4.0,
    )

    return TrajectoryConstraintResult(
        satisfied=not violations,
        score=score,
        violations=violations,
    )


def check_glucose_bounds(
    glucose: Sequence[float],
    lower: float,
    upper: float,
) -> TrajectoryConstraintResult:
    """Check that every trajectory point remains in [lower, upper]."""
    g = np.asarray(glucose, dtype=float).reshape(-1)

    violations = {}
    if not np.isfinite(g).all():
        violations["non_finite"] = True
    else:
        below = np.where(g < lower)[0]
        above = np.where(g > upper)[0]
        if len(below):
            violations["below_lower"] = below.tolist()
        if len(above):
            violations["above_upper"] = above.tolist()

    return TrajectoryConstraintResult(
        satisfied=not violations,
        score=0.0 if not violations else 1.0,
        violations=violations,
    )


__all__ = [
    "TrajectoryConstraintConfig",
    "TrajectoryConstraintResult",
    "check_trajectory",
    "check_glucose_bounds",
]
