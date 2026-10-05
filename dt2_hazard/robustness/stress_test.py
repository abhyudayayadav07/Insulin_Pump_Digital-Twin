"""Stress testing utilities for DT2 components."""
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Any
import numpy as np


@dataclass
class StressTestResult:
    scenario_id: str
    passed: bool
    score: float
    failures: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_stress_test(
    scenario_id: str,
    evaluator: Callable[[Any], Dict[str, Any]],
    inputs: Iterable[Any],
    success_threshold: float = 0.90,
) -> StressTestResult:
    outcomes = []
    failures = []

    for item in inputs:
        result = evaluator(item)
        outcomes.append(bool(result.get("passed", False)))
        if not outcomes[-1]:
            failures.append(str(result.get("reason", "unspecified failure")))

    score = float(np.mean(outcomes)) if outcomes else 0.0
    return StressTestResult(
        scenario_id=scenario_id,
        passed=score >= success_threshold,
        score=score,
        failures=failures,
    )
