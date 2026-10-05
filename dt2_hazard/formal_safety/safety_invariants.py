"""Safety invariants used as hard constraints during DT2 monitoring."""

from dataclasses import dataclass
from typing import Mapping, Any


@dataclass(frozen=True)
class SafetyInvariant:
    name: str
    description: str
    key: str
    lower: float | None = None
    upper: float | None = None

    def check(self, state: Mapping[str, Any]) -> bool:
        if self.key not in state:
            return False
        value = float(state[self.key])
        if self.lower is not None and value < self.lower:
            return False
        if self.upper is not None and value > self.upper:
            return False
        return True


def default_invariants():
    return [
        SafetyInvariant(
            "glucose_physiological_bounds",
            "Glucose remains within simulator/model bounds.",
            "glucose",
            20.0,
            600.0,
        ),
        SafetyInvariant(
            "nonnegative_insulin",
            "Insulin command cannot be negative.",
            "insulin",
            0.0,
            None,
        ),
        SafetyInvariant(
            "sensor_integrity",
            "Sensor integrity indicator remains acceptable.",
            "sensor_integrity",
            0.0,
            1.0,
        ),
    ]


def check_invariants(state: Mapping[str, Any], invariants=None):
    invs = list(invariants) if invariants is not None else default_invariants()
    return {inv.name: inv.check(state) for inv in invs}
