"""Mitigation deadline estimation.

The deadline is the time from the current point to the first predicted
unsafe state. It is an engineering estimate and must be validated against
the selected hazard model and prediction horizon.
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class DeadlineResult:
    deadline_minutes: Optional[float]
    crossing_time_minutes: Optional[float]
    crossed: bool
    horizon_minutes: float


def estimate_mitigation_deadline(
    times_minutes: Sequence[float],
    hazard_scores: Sequence[float],
    threshold: float = 0.60,
) -> DeadlineResult:
    if len(times_minutes) != len(hazard_scores):
        raise ValueError("times_minutes and hazard_scores must have equal length")
    if not times_minutes:
        return DeadlineResult(None, None, False, 0.0)

    start = float(times_minutes[0])
    horizon = max(0.0, float(times_minutes[-1]) - start)

    for t, score in zip(times_minutes, hazard_scores):
        if float(score) >= threshold:
            crossing = float(t)
            return DeadlineResult(
                max(0.0, crossing - start),
                crossing,
                True,
                horizon,
            )

    return DeadlineResult(None, None, False, horizon)


def deadline_from_first_unsafe(
    current_time: float,
    future_times: Iterable[float],
    unsafe_flags: Iterable[bool],
) -> Optional[float]:
    for t, unsafe in zip(future_times, unsafe_flags):
        if unsafe:
            return max(0.0, float(t) - float(current_time))
    return None
