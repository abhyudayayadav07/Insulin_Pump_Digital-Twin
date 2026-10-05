"""
DT2 Hazard Function Module
==========================

Computes continuous hazard-function values from physiological, device,
controller, and Digital Twin evidence.

Conceptually, the hazard function describes how close the system is to an
unsafe condition. It is a deterministic evidence-to-hazard mapping layer;
it is NOT a statistical probability estimator.

Expected DT2 flow:

    DT1 prediction / simulator / attack evidence
                    |
                    v
             hazard_function
                    |
                    v
              hazard score
                    |
          +---------+---------+
          |                   |
          v                   v
 hazard_probability      hazard_predictor
          |                   |
          +---------+---------+
                    v
                 risk/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import exp, isfinite
from typing import Any, Dict, Mapping, Optional

try:
    from .hazard_definition import HazardDefinition, HazardType
except ImportError:
    from hazard_definition import HazardDefinition, HazardType


class HazardFunctionMethod(str, Enum):
    """Methods available for converting evidence into hazard intensity."""

    THRESHOLD = "threshold"
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    EXPONENTIAL = "exponential"


@dataclass
class HazardFunctionConfig:
    """
    Configuration for one hazard-function calculation.

    Parameters
    ----------
    warning_threshold:
        Normalized evidence level at which a warning begins.
    active_threshold:
        Normalized evidence level at which the hazard becomes active.
    critical_threshold:
        Normalized evidence level considered critical.
    method:
        Mathematical mapping used to convert normalized evidence to hazard
        intensity.
    slope:
        Controls steepness for sigmoid/exponential mappings.
    """

    warning_threshold: float = 0.30
    active_threshold: float = 0.60
    critical_threshold: float = 0.85
    method: HazardFunctionMethod = HazardFunctionMethod.SIGMOID
    slope: float = 8.0

    def __post_init__(self) -> None:
        if not (
            0.0 <= self.warning_threshold
            <= self.active_threshold
            <= self.critical_threshold
            <= 1.0
        ):
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= warning <= active <= critical <= 1."
            )

        if self.slope <= 0:
            raise ValueError("slope must be greater than zero.")


@dataclass
class HazardFunctionResult:
    """
    Result produced by a hazard-function evaluation.

    ``intensity`` is a normalized [0, 1] hazard intensity. It is not itself
    a probability and should not be interpreted as a final safety decision.
    """

    hazard_id: str
    hazard_type: str
    method: str
    evidence_score: float
    intensity: float
    normalized_distance: float
    state: str
    triggered: bool
    critical: bool
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        return {
            "hazard_id": self.hazard_id,
            "hazard_type": self.hazard_type,
            "method": self.method,
            "evidence_score": self.evidence_score,
            "intensity": self.intensity,
            "normalized_distance": self.normalized_distance,
            "state": self.state,
            "triggered": self.triggered,
            "critical": self.critical,
            "components": dict(self.components),
            "metadata": dict(self.metadata),
        }


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a numeric value to a specified interval."""
    return max(lower, min(upper, float(value)))


def normalize_distance(
    value: float,
    *,
    safe_value: float,
    hazard_value: float,
) -> float:
    """
    Normalize distance from a safe value toward a hazard boundary.

    Returns 0 at the safe reference and 1 at the hazard boundary.

    Values beyond the hazard boundary are clipped to 1.
    """
    value = float(value)
    safe_value = float(safe_value)
    hazard_value = float(hazard_value)

    denominator = hazard_value - safe_value
    if denominator == 0:
        raise ValueError("safe_value and hazard_value must differ.")

    return clamp((value - safe_value) / denominator)


def _sigmoid(x: float, slope: float) -> float:
    """Numerically stable logistic sigmoid."""
    z = slope * (x - 0.5)

    if z >= 0:
        return 1.0 / (1.0 + exp(-z))

    ez = exp(z)
    return ez / (1.0 + ez)


def _map_evidence(
    evidence_score: float,
    config: HazardFunctionConfig,
) -> float:
    """Map normalized evidence into normalized hazard intensity."""
    score = clamp(evidence_score)

    if config.method == HazardFunctionMethod.THRESHOLD:
        if score < config.warning_threshold:
            return 0.0
        if score < config.active_threshold:
            return 0.5 * (
                score - config.warning_threshold
            ) / max(
                config.active_threshold - config.warning_threshold,
                1e-12,
            )
        if score < config.critical_threshold:
            return 0.5 + 0.5 * (
                score - config.active_threshold
            ) / max(
                config.critical_threshold - config.active_threshold,
                1e-12,
            )
        return 1.0

    if config.method == HazardFunctionMethod.LINEAR:
        return score

    if config.method == HazardFunctionMethod.EXPONENTIAL:
        denominator = exp(config.slope) - 1.0
        if denominator <= 0:
            return score
        return clamp((exp(config.slope * score) - 1.0) / denominator)

    # Default: sigmoid
    # The sigmoid is centered at 0.5 and normalized to preserve approximately
    # zero-to-one behavior at the boundaries.
    low = _sigmoid(0.0, config.slope)
    high = _sigmoid(1.0, config.slope)
    raw = _sigmoid(score, config.slope)
    return clamp((raw - low) / max(high - low, 1e-12))


def evidence_to_hazard_intensity(
    evidence_score: float,
    config: Optional[HazardFunctionConfig] = None,
) -> float:
    """
    Convert normalized evidence strength to hazard intensity.

    Parameters
    ----------
    evidence_score:
        Value in [0, 1], where larger values indicate stronger evidence of
        unsafe behavior.
    config:
        Hazard-function configuration.

    Returns
    -------
    float
        Hazard intensity in [0, 1].
    """
    if not isfinite(float(evidence_score)):
        raise ValueError("evidence_score must be finite.")

    config = config or HazardFunctionConfig()
    return _map_evidence(float(evidence_score), config)


def classify_hazard_intensity(
    intensity: float,
    config: Optional[HazardFunctionConfig] = None,
) -> str:
    """Classify intensity as inactive, warning, active, or critical."""
    config = config or HazardFunctionConfig()
    intensity = clamp(intensity)

    if intensity >= config.critical_threshold:
        return "critical"
    if intensity >= config.active_threshold:
        return "active"
    if intensity >= config.warning_threshold:
        return "warning"
    return "inactive"


def glucose_hazard_evidence(
    glucose: float,
    *,
    lower_safe: float = 70.0,
    upper_safe: float = 180.0,
    lower_critical: float = 54.0,
    upper_critical: float = 250.0,
) -> float:
    """
    Estimate glucose-related hazard evidence.

    This combines low-glucose and high-glucose deviation into a single
    normalized evidence value. The function is intentionally symmetric in
    structure but uses clinically configurable boundaries.

    It does not diagnose a patient and does not replace clinical criteria.
    """
    glucose = float(glucose)

    if not all(
        isfinite(float(x))
        for x in (
            glucose,
            lower_safe,
            upper_safe,
            lower_critical,
            upper_critical,
        )
    ):
        raise ValueError("All glucose boundaries must be finite.")

    if not (
        lower_critical < lower_safe < upper_safe < upper_critical
    ):
        raise ValueError(
            "Expected lower_critical < lower_safe < upper_safe < upper_critical."
        )

    low_evidence = 0.0
    high_evidence = 0.0

    if glucose < lower_safe:
        low_evidence = clamp(
            (lower_safe - glucose)
            / (lower_safe - lower_critical)
        )

    if glucose > upper_safe:
        high_evidence = clamp(
            (glucose - upper_safe)
            / (upper_critical - upper_safe)
        )

    return max(low_evidence, high_evidence)


def trajectory_hazard_evidence(
    current_value: float,
    predicted_value: float,
    *,
    lower_safe: float = 70.0,
    upper_safe: float = 180.0,
    lower_critical: float = 54.0,
    upper_critical: float = 250.0,
    prediction_weight: float = 0.6,
) -> float:
    """
    Combine current and predicted glucose hazard evidence.

    The prediction contribution allows DT1 to inform DT2 without allowing
    DT1's output to directly make the final safety decision.
    """
    if not 0.0 <= prediction_weight <= 1.0:
        raise ValueError("prediction_weight must be in [0, 1].")

    current_evidence = glucose_hazard_evidence(
        current_value,
        lower_safe=lower_safe,
        upper_safe=upper_safe,
        lower_critical=lower_critical,
        upper_critical=upper_critical,
    )

    predicted_evidence = glucose_hazard_evidence(
        predicted_value,
        lower_safe=lower_safe,
        upper_safe=upper_safe,
        lower_critical=lower_critical,
        upper_critical=upper_critical,
    )

    return clamp(
        (1.0 - prediction_weight) * current_evidence
        + prediction_weight * predicted_evidence
    )


def insulin_delivery_evidence(
    commanded_insulin: float,
    delivered_insulin: float,
    *,
    tolerance: float = 0.05,
    saturation_error: float = 1.0,
) -> float:
    """
    Estimate evidence of insulin delivery mismatch.

    ``tolerance`` is the relative mismatch considered acceptable.

    Example:
        commanded=1.0, delivered=0.95, tolerance=0.05
        produces approximately zero evidence.
    """
    commanded = float(commanded_insulin)
    delivered = float(delivered_insulin)

    if commanded < 0 or delivered < 0:
        raise ValueError("Insulin values must be non-negative.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    if saturation_error <= 0:
        raise ValueError("saturation_error must be > 0.")

    denominator = max(abs(commanded), 1e-8)
    relative_error = abs(delivered - commanded) / denominator

    excess = max(0.0, relative_error - tolerance)
    return clamp(excess / saturation_error)


def residual_hazard_evidence(
    residual: float,
    *,
    warning_residual: float,
    critical_residual: float,
) -> float:
    """
    Convert an absolute model/device residual into [0, 1] evidence.

    Residuals below ``warning_residual`` produce no warning evidence.
    ``critical_residual`` maps to 1.
    """
    residual = abs(float(residual))
    warning_residual = float(warning_residual)
    critical_residual = float(critical_residual)

    if warning_residual < 0:
        raise ValueError("warning_residual must be >= 0.")
    if critical_residual <= warning_residual:
        raise ValueError(
            "critical_residual must be greater than warning_residual."
        )

    if residual <= warning_residual:
        return 0.0

    return clamp(
        (residual - warning_residual)
        / (critical_residual - warning_residual)
    )


def combine_evidence(
    *components: float,
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """
    Combine multiple normalized evidence components.

    Without weights, the maximum component is returned. This is conservative
    for safety monitoring because a strong individual signal remains visible.

    With weights, a weighted average is returned.
    """
    values = [clamp(value) for value in components]

    if not values:
        return 0.0

    if weights is None:
        return max(values)

    weight_values = [float(value) for value in weights.values()]
    if len(weight_values) != len(values):
        raise ValueError(
            "Number of weights must match number of evidence components."
        )

    if any(weight < 0 for weight in weight_values):
        raise ValueError("Evidence weights must be non-negative.")

    total_weight = sum(weight_values)
    if total_weight <= 0:
        raise ValueError("At least one evidence weight must be positive.")

    return clamp(
        sum(value * weight for value, weight in zip(values, weight_values))
        / total_weight
    )


class HazardFunction:
    """
    High-level DT2 hazard-function evaluator.

    It accepts a HazardDefinition and produces a HazardFunctionResult from
    normalized or domain-specific evidence.
    """

    def __init__(
        self,
        hazard: HazardDefinition,
        config: Optional[HazardFunctionConfig] = None,
    ) -> None:
        self.hazard = hazard
        self.config = config or HazardFunctionConfig()

    def evaluate(
        self,
        evidence_score: float,
        *,
        components: Optional[Mapping[str, float]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> HazardFunctionResult:
        """Evaluate one hazard from a normalized evidence score."""
        score = clamp(evidence_score)
        intensity = evidence_to_hazard_intensity(score, self.config)
        state = classify_hazard_intensity(intensity, self.config)

        return HazardFunctionResult(
            hazard_id=self.hazard.hazard_id,
            hazard_type=self.hazard.hazard_type.value,
            method=self.config.method.value,
            evidence_score=score,
            intensity=intensity,
            normalized_distance=score,
            state=state,
            triggered=state in {"active", "critical"},
            critical=state == "critical",
            components={
                key: clamp(value)
                for key, value in (components or {}).items()
            },
            metadata=dict(metadata or {}),
        )

    def evaluate_observation(
        self,
        observation: Mapping[str, Any],
    ) -> HazardFunctionResult:
        """
        Evaluate evidence from a generic observation dictionary.

        If ``evidence_score`` is supplied it is used directly. Otherwise,
        common glucose/DT1 residual fields are inspected.
        """
        if "evidence_score" in observation:
            score = float(observation["evidence_score"])
            components = {}
        else:
            components = self._derive_components(observation)
            score = combine_evidence(*components.values())

        return self.evaluate(
            score,
            components=components,
            metadata={
                key: value
                for key, value in observation.items()
                if key not in components
            },
        )

    def _derive_components(
        self,
        observation: Mapping[str, Any],
    ) -> Dict[str, float]:
        """Derive hazard evidence from commonly available signals."""
        components: Dict[str, float] = {}

        hazard_type = self.hazard.hazard_type

        if hazard_type in {
            HazardType.HYPOGLYCEMIA,
            HazardType.HYPERGLYCEMIA,
        }:
            glucose_key = (
                "glucose"
                if "glucose" in observation
                else "cgm"
                if "cgm" in observation
                else None
            )

            if glucose_key is not None:
                current = float(observation[glucose_key])
                current_evidence = glucose_hazard_evidence(current)

                if "predicted_glucose" in observation:
                    score = trajectory_hazard_evidence(
                        current,
                        float(observation["predicted_glucose"]),
                    )
                else:
                    score = current_evidence

                # Keep hypoglycemia/hyperglycemia hazards direction-specific.
                if hazard_type == HazardType.HYPOGLYCEMIA:
                    score = (
                        clamp((70.0 - current) / (70.0 - 54.0))
                        if current < 70.0
                        else 0.0
                    )
                    if "predicted_glucose" in observation:
                        predicted = float(observation["predicted_glucose"])
                        predicted_score = (
                            clamp((70.0 - predicted) / (70.0 - 54.0))
                            if predicted < 70.0
                            else 0.0
                        )
                        score = max(score, predicted_score)

                if hazard_type == HazardType.HYPERGLYCEMIA:
                    score = (
                        clamp((current - 180.0) / (250.0 - 180.0))
                        if current > 180.0
                        else 0.0
                    )
                    if "predicted_glucose" in observation:
                        predicted = float(observation["predicted_glucose"])
                        predicted_score = (
                            clamp((predicted - 180.0) / (250.0 - 180.0))
                            if predicted > 180.0
                            else 0.0
                        )
                        score = max(score, predicted_score)

                components["glucose"] = score

        if hazard_type in {
            HazardType.INSULIN_OVERDELIVERY,
            HazardType.INSULIN_UNDERDELIVERY,
            HazardType.PUMP_FAILURE,
        }:
            if (
                "commanded_insulin" in observation
                and "delivered_insulin" in observation
            ):
                mismatch = insulin_delivery_evidence(
                    float(observation["commanded_insulin"]),
                    float(observation["delivered_insulin"]),
                )

                if hazard_type == HazardType.PUMP_FAILURE:
                    components["delivery_mismatch"] = mismatch
                elif hazard_type == HazardType.INSULIN_OVERDELIVERY:
                    commanded = float(observation["commanded_insulin"])
                    delivered = float(observation["delivered_insulin"])
                    relative = (
                        (delivered - commanded)
                        / max(abs(commanded), 1e-8)
                    )
                    components["overdelivery"] = clamp(
                        max(0.0, relative - 0.05)
                    )
                elif hazard_type == HazardType.INSULIN_UNDERDELIVERY:
                    commanded = float(observation["commanded_insulin"])
                    delivered = float(observation["delivered_insulin"])
                    relative = (
                        (commanded - delivered)
                        / max(abs(commanded), 1e-8)
                    )
                    components["underdelivery"] = clamp(
                        max(0.0, relative - 0.05)
                    )

        if "residual" in observation:
            components["residual"] = clamp(abs(float(observation["residual"])))

        if "attack_score" in observation:
            components["attack"] = clamp(float(observation["attack_score"]))

        if "sensor_failure_score" in observation:
            components["sensor_failure"] = clamp(
                float(observation["sensor_failure_score"])
            )

        if "controller_failure_score" in observation:
            components["controller_failure"] = clamp(
                float(observation["controller_failure_score"])
            )

        return components


__all__ = [
    "HazardFunctionMethod",
    "HazardFunctionConfig",
    "HazardFunctionResult",
    "HazardFunction",
    "clamp",
    "normalize_distance",
    "evidence_to_hazard_intensity",
    "classify_hazard_intensity",
    "glucose_hazard_evidence",
    "trajectory_hazard_evidence",
    "insulin_delivery_evidence",
    "residual_hazard_evidence",
    "combine_evidence",
]
