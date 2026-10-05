"""
Control-related safety constraints for DT2.

These constraints validate insulin/control commands without generating or
executing a dose. The decision and actuation layers remain responsible for
actual control authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ControlConstraintConfig:
    insulin_min: float = 0.0
    insulin_max: float = 10.0
    max_step_change: float = 2.0
    max_cumulative_change: float = 5.0


@dataclass
class ControlConstraintResult:
    satisfied: bool
    violations: dict


def check_insulin_command(
    command: float,
    *,
    config: Optional[ControlConstraintConfig] = None,
) -> ControlConstraintResult:
    """Validate one insulin command against range limits."""
    config = config or ControlConstraintConfig()
    value = float(command)

    violations = {}
    if not np.isfinite(value):
        violations["non_finite"] = True
    else:
        if value < config.insulin_min:
            violations["below_minimum"] = value
        if value > config.insulin_max:
            violations["above_maximum"] = value

    return ControlConstraintResult(
        satisfied=not violations,
        violations=violations,
    )


def check_command_sequence(
    commands: Sequence[float],
    *,
    config: Optional[ControlConstraintConfig] = None,
) -> ControlConstraintResult:
    """Validate a sequence of commands for range and abrupt changes."""
    config = config or ControlConstraintConfig()
    values = np.asarray(commands, dtype=float).reshape(-1)

    violations = {}

    if len(values) == 0:
        violations["empty_sequence"] = True
    elif not np.isfinite(values).all():
        violations["non_finite"] = True
    else:
        if np.any(values < config.insulin_min):
            violations["below_minimum"] = True
        if np.any(values > config.insulin_max):
            violations["above_maximum"] = True

        if len(values) > 1:
            steps = np.abs(np.diff(values))
            if np.any(steps > config.max_step_change):
                violations["abrupt_change"] = float(np.max(steps))

            cumulative = np.abs(values - values[0])
            if np.any(cumulative > config.max_cumulative_change):
                violations["cumulative_change"] = float(np.max(cumulative))

    return ControlConstraintResult(
        satisfied=not violations,
        violations=violations,
    )


__all__ = [
    "ControlConstraintConfig",
    "ControlConstraintResult",
    "check_insulin_command",
    "check_command_sequence",
]
