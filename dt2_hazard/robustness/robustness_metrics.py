"""Metrics for quantifying robustness under perturbations and faults."""
from dataclasses import dataclass
from typing import Sequence, Optional
import numpy as np


@dataclass(frozen=True)
class RobustnessMetrics:
    baseline_performance: float
    perturbed_performance: float
    absolute_degradation: float
    relative_degradation: float
    robustness_ratio: float


def compute_robustness_metrics(
    baseline_performance: float,
    perturbed_performance: float,
) -> RobustnessMetrics:
    b = float(baseline_performance)
    p = float(perturbed_performance)
    degradation = b - p
    relative = degradation / max(abs(b), 1e-12)
    ratio = p / max(abs(b), 1e-12)
    return RobustnessMetrics(
        baseline_performance=b,
        perturbed_performance=p,
        absolute_degradation=degradation,
        relative_degradation=relative,
        robustness_ratio=ratio,
    )


def safety_violation_rate(values: Sequence[float], lower: float, upper: float) -> float:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.mean((x < lower) | (x > upper)))


def mean_absolute_change(
    baseline: Sequence[float],
    perturbed: Sequence[float],
) -> float:
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(perturbed, dtype=float)
    if a.shape != b.shape:
        raise ValueError("baseline and perturbed must have equal shape")
    return float(np.mean(np.abs(a - b))) if a.size else 0.0
