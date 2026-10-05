"""High-level robustness evaluation for DT2 outputs."""
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence
import numpy as np

from .robustness_config import RobustnessConfig
from .robustness_metrics import compute_robustness_metrics


@dataclass
class RobustnessEvaluation:
    robust: bool
    score: float
    degraded: bool
    safety_violations: int
    prediction_error_violations: int
    integrity_failures: int
    reasons: list[str] = field(default_factory=list)


class RobustnessEvaluator:
    def __init__(self, config: Optional[RobustnessConfig] = None):
        self.config = config or RobustnessConfig()

    def evaluate(
        self,
        glucose: Sequence[float],
        prediction_error: Optional[Sequence[float]] = None,
        hazard_score: Optional[Sequence[float]] = None,
        integrity: Optional[Mapping[str, float]] = None,
        baseline_score: float = 1.0,
        perturbed_score: float = 1.0,
    ) -> RobustnessEvaluation:
        g = np.asarray(glucose, dtype=float)
        safety_bad = int(np.sum(
            (g < self.config.glucose_min) | (g > self.config.glucose_max)
        ))

        error_bad = 0
        if prediction_error is not None:
            e = np.abs(np.asarray(prediction_error, dtype=float))
            error_bad = int(np.sum(e > self.config.max_prediction_error))

        hazard_bad = 0
        if hazard_score is not None:
            h = np.asarray(hazard_score, dtype=float)
            hazard_bad = int(np.sum(h > self.config.max_hazard_score))

        integrity_failures = 0
        reasons = []
        thresholds = {
            "sensor_integrity": self.config.min_sensor_integrity,
            "controller_integrity": self.config.min_controller_integrity,
            "pump_integrity": self.config.min_pump_integrity,
            "communication_integrity": self.config.min_communication_integrity,
        }
        if integrity:
            for key, threshold in thresholds.items():
                if key in integrity and float(integrity[key]) < threshold:
                    integrity_failures += 1
                    reasons.append(f"{key} below threshold")

        metrics = compute_robustness_metrics(baseline_score, perturbed_score)
        degraded = metrics.relative_degradation > 0.10

        if safety_bad:
            reasons.append("physiological safety bound violations")
        if error_bad:
            reasons.append("prediction error threshold violations")
        if hazard_bad:
            reasons.append("critical hazard-score threshold violations")
        if degraded:
            reasons.append("performance degraded by more than 10%")

        total_checks = max(1, len(g) + error_bad + hazard_bad + len(thresholds))
        penalty = min(
            1.0,
            (safety_bad + error_bad + hazard_bad + integrity_failures) / total_checks
        )
        score = max(0.0, 1.0 - penalty)

        return RobustnessEvaluation(
            robust=(score >= self.config.required_success_rate and not reasons),
            score=score,
            degraded=degraded,
            safety_violations=safety_bad,
            prediction_error_violations=error_bad,
            integrity_failures=integrity_failures,
            reasons=reasons,
        )


def evaluate_robustness(*args, config=None, **kwargs):
    return RobustnessEvaluator(config).evaluate(*args, **kwargs)
