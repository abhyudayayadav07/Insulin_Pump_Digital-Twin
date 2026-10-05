"""Evaluation utilities for mitigation plans and outcomes."""
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class MitigationEvaluation:
    success: bool
    reaction_time_minutes: Optional[float]
    recovery_time_minutes: Optional[float]
    max_hazard_score: float
    deadline_met: bool
    new_hazard: bool
    reason: str = ""


def evaluate_mitigation(
    hazard_scores: Sequence[float],
    event_times: Optional[Sequence[float]] = None,
    recovery_threshold: float = 0.30,
    deadline_minutes: Optional[float] = None,
    new_hazard: bool = False,
) -> MitigationEvaluation:
    if not hazard_scores:
        return MitigationEvaluation(
            False, None, None, 0.0, False, new_hazard, "No hazard trajectory supplied."
        )

    peak_index = max(range(len(hazard_scores)), key=lambda i: float(hazard_scores[i]))
    peak = float(hazard_scores[peak_index])

    recovery_index = None
    for i in range(peak_index, len(hazard_scores)):
        if float(hazard_scores[i]) <= recovery_threshold:
            recovery_index = i
            break

    reaction_time = None
    recovery_time = None
    if event_times is not None and len(event_times) == len(hazard_scores):
        reaction_time = float(event_times[peak_index] - event_times[0])
        if recovery_index is not None:
            recovery_time = float(event_times[recovery_index] - event_times[peak_index])

    deadline_met = (
        True
        if deadline_minutes is None or reaction_time is None
        else reaction_time <= deadline_minutes
    )
    success = recovery_index is not None and deadline_met and not new_hazard

    return MitigationEvaluation(
        success=success,
        reaction_time_minutes=reaction_time,
        recovery_time_minutes=recovery_time,
        max_hazard_score=peak,
        deadline_met=deadline_met,
        new_hazard=new_hazard,
        reason="Mitigation outcome evaluated from hazard trajectory.",
    )
