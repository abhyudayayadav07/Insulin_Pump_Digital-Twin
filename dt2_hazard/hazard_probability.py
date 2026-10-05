"""
DT2 Hazard Probability Module
=============================

Estimates the probability of a hazard event from hazard-function evidence,
historical observations, and optional prior probability.

Important:
    Hazard intensity/evidence is not automatically a probability.
    This module performs an explicit probabilistic mapping.

The output is an estimated probability for the configured observation
window. It should be used as one input to the later hazard predictor and
risk modules, not as a final safety decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, log
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass
class HazardProbabilityConfig:
    """
    Configuration for probability estimation.

    Parameters
    ----------
    prior_probability:
        Prior probability of the hazard for the configured observation window.
    evidence_weight:
        Strength with which current hazard evidence changes the prior.
    temporal_decay:
        Decay factor used when aggregating historical observations.
        Must be in (0, 1].
    minimum_probability:
        Lower numerical bound.
    maximum_probability:
        Upper numerical bound.
    """

    prior_probability: float = 0.01
    evidence_weight: float = 4.0
    temporal_decay: float = 0.90
    minimum_probability: float = 1e-6
    maximum_probability: float = 1.0 - 1e-6

    def __post_init__(self) -> None:
        self.prior_probability = _validate_probability(
            self.prior_probability,
            "prior_probability",
        )

        if self.evidence_weight < 0:
            raise ValueError("evidence_weight must be >= 0.")

        if not 0.0 < self.temporal_decay <= 1.0:
            raise ValueError("temporal_decay must be in (0, 1].")

        if not 0.0 < self.minimum_probability < 1.0:
            raise ValueError("minimum_probability must be in (0, 1).")

        if not 0.0 < self.maximum_probability < 1.0:
            raise ValueError("maximum_probability must be in (0, 1).")

        if self.minimum_probability >= self.maximum_probability:
            raise ValueError(
                "minimum_probability must be less than maximum_probability."
            )


@dataclass
class HazardProbabilityResult:
    """Structured result of a hazard-probability estimation."""

    hazard_id: str
    probability: float
    evidence_score: float
    prior_probability: float
    posterior_probability: float
    observation_window: Optional[float] = None
    method: str = "log_odds_evidence"
    historical_evidence: Optional[float] = None
    sample_count: int = 0
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def probability_percent(self) -> float:
        """Return probability as a percentage."""
        return 100.0 * self.probability

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary."""
        return {
            "hazard_id": self.hazard_id,
            "probability": self.probability,
            "probability_percent": self.probability_percent,
            "evidence_score": self.evidence_score,
            "prior_probability": self.prior_probability,
            "posterior_probability": self.posterior_probability,
            "observation_window": self.observation_window,
            "method": self.method,
            "historical_evidence": self.historical_evidence,
            "sample_count": self.sample_count,
            "components": dict(self.components),
            "metadata": dict(self.metadata),
        }


def _validate_probability(value: float, name: str) -> float:
    """Validate a probability value."""
    value = float(value)

    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")

    return value


def clamp_probability(
    probability: float,
    minimum: float = 1e-6,
    maximum: float = 1.0 - 1e-6,
) -> float:
    """Clamp probability to a numerically stable interval."""
    if minimum >= maximum:
        raise ValueError("minimum must be smaller than maximum.")

    return max(minimum, min(maximum, float(probability)))


def probability_to_log_odds(probability: float) -> float:
    """Convert probability to log-odds."""
    probability = clamp_probability(probability)

    return log(probability / (1.0 - probability))


def log_odds_to_probability(log_odds: float) -> float:
    """Convert log-odds to probability."""
    if log_odds >= 0:
        z = exp(-log_odds)
        return 1.0 / (1.0 + z)

    z = exp(log_odds)
    return z / (1.0 + z)


def evidence_to_probability(
    evidence_score: float,
    *,
    prior_probability: float = 0.01,
    evidence_weight: float = 4.0,
) -> float:
    """
    Convert normalized hazard evidence into a probability estimate.

    The mapping is performed in log-odds space:

        posterior_log_odds =
            prior_log_odds + evidence_weight * centered_evidence

    where evidence=0.5 represents neutral evidence.

    This is a configurable evidence model, not a learned/calibrated
    statistical model unless its parameters are calibrated from data.
    """
    evidence_score = float(evidence_score)
    prior_probability = _validate_probability(
        prior_probability,
        "prior_probability",
    )

    if not isfinite(evidence_score):
        raise ValueError("evidence_score must be finite.")

    if not 0.0 <= evidence_score <= 1.0:
        raise ValueError("evidence_score must be between 0 and 1.")

    if evidence_weight < 0:
        raise ValueError("evidence_weight must be >= 0.")

    prior = clamp_probability(prior_probability)

    centered_evidence = 2.0 * evidence_score - 1.0

    posterior_log_odds = (
        probability_to_log_odds(prior)
        + evidence_weight * centered_evidence
    )

    return clamp_probability(log_odds_to_probability(posterior_log_odds))


def aggregate_historical_evidence(
    evidence_history: Sequence[float],
    *,
    temporal_decay: float = 0.90,
) -> float:
    """
    Aggregate historical evidence using exponentially decaying weights.

    The newest observation is assumed to be the last element.
    """
    if not evidence_history:
        return 0.0

    if not 0.0 < temporal_decay <= 1.0:
        raise ValueError("temporal_decay must be in (0, 1].")

    values = [
        max(0.0, min(1.0, float(value)))
        for value in evidence_history
    ]

    weighted_sum = 0.0
    weight_sum = 0.0

    for age, value in enumerate(reversed(values)):
        weight = temporal_decay ** age
        weighted_sum += weight * value
        weight_sum += weight

    return weighted_sum / max(weight_sum, 1e-12)


def combine_probability_evidence(
    probabilities: Iterable[float],
    *,
    weights: Optional[Iterable[float]] = None,
) -> float:
    """
    Combine multiple probability estimates.

    A weighted arithmetic mean is used. This is an evidence aggregation
    mechanism and should be calibrated against the intended application.
    """
    values = [
        _validate_probability(value, "probability")
        for value in probabilities
    ]

    if not values:
        return 0.0

    if weights is None:
        return sum(values) / len(values)

    weight_values = [float(weight) for weight in weights]

    if len(weight_values) != len(values):
        raise ValueError(
            "Number of weights must match number of probabilities."
        )

    if any(weight < 0 for weight in weight_values):
        raise ValueError("Weights must be non-negative.")

    total_weight = sum(weight_values)
    if total_weight <= 0:
        raise ValueError("At least one weight must be positive.")

    return sum(
        probability * weight
        for probability, weight in zip(values, weight_values)
    ) / total_weight


class HazardProbabilityEstimator:
    """
    Stateful hazard-probability estimator.

    The estimator keeps a short evidence history for each hazard and can
    update probability at every simulation timestep.
    """

    def __init__(
        self,
        config: Optional[HazardProbabilityConfig] = None,
    ) -> None:
        self.config = config or HazardProbabilityConfig()
        self._history: Dict[str, List[float]] = {}

    def estimate(
        self,
        hazard_id: str,
        evidence_score: float,
        *,
        observation_window: Optional[float] = None,
        components: Optional[Mapping[str, float]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> HazardProbabilityResult:
        """
        Estimate probability for one hazard at the current timestep.
        """
        evidence_score = float(evidence_score)

        if not 0.0 <= evidence_score <= 1.0:
            raise ValueError("evidence_score must be between 0 and 1.")

        history = self._history.setdefault(hazard_id, [])
        history.append(evidence_score)

        historical_evidence = aggregate_historical_evidence(
            history,
            temporal_decay=self.config.temporal_decay,
        )

        # Blend current evidence and historical evidence. The current value
        # gets equal conceptual importance with the decayed history while
        # remaining explicit and inspectable.
        effective_evidence = (
            0.5 * evidence_score
            + 0.5 * historical_evidence
        )

        probability = evidence_to_probability(
            effective_evidence,
            prior_probability=self.config.prior_probability,
            evidence_weight=self.config.evidence_weight,
        )

        return HazardProbabilityResult(
            hazard_id=hazard_id,
            probability=probability,
            evidence_score=evidence_score,
            prior_probability=self.config.prior_probability,
            posterior_probability=probability,
            observation_window=observation_window,
            historical_evidence=historical_evidence,
            sample_count=len(history),
            components={
                key: max(0.0, min(1.0, float(value)))
                for key, value in (components or {}).items()
            },
            metadata=dict(metadata or {}),
        )

    def estimate_from_hazard_function(
        self,
        hazard_function_result: Any,
        *,
        observation_window: Optional[float] = None,
    ) -> HazardProbabilityResult:
        """
        Estimate probability directly from a HazardFunctionResult.

        The object is duck-typed to avoid a hard dependency on
        hazard_function.py.
        """
        hazard_id = str(hazard_function_result.hazard_id)
        evidence_score = float(hazard_function_result.intensity)

        components = getattr(
            hazard_function_result,
            "components",
            {},
        )

        metadata = {
            "hazard_function_method": getattr(
                hazard_function_result,
                "method",
                "unknown",
            ),
            "hazard_function_state": getattr(
                hazard_function_result,
                "state",
                "unknown",
            ),
        }

        return self.estimate(
            hazard_id,
            evidence_score,
            observation_window=observation_window,
            components=components,
            metadata=metadata,
        )

    def history(self, hazard_id: str) -> List[float]:
        """Return a copy of the evidence history for one hazard."""
        return list(self._history.get(hazard_id, []))

    def reset(self, hazard_id: Optional[str] = None) -> None:
        """Reset one hazard history or all histories."""
        if hazard_id is None:
            self._history.clear()
        else:
            self._history.pop(hazard_id, None)


def estimate_hazard_probability(
    hazard_id: str,
    evidence_score: float,
    *,
    prior_probability: float = 0.01,
    evidence_weight: float = 4.0,
    observation_window: Optional[float] = None,
) -> HazardProbabilityResult:
    """
    Stateless convenience function for one probability estimate.
    """
    config = HazardProbabilityConfig(
        prior_probability=prior_probability,
        evidence_weight=evidence_weight,
    )

    estimator = HazardProbabilityEstimator(config)

    return estimator.estimate(
        hazard_id,
        evidence_score,
        observation_window=observation_window,
    )


__all__ = [
    "HazardProbabilityConfig",
    "HazardProbabilityResult",
    "HazardProbabilityEstimator",
    "clamp_probability",
    "probability_to_log_odds",
    "log_odds_to_probability",
    "evidence_to_probability",
    "aggregate_historical_evidence",
    "combine_probability_evidence",
    "estimate_hazard_probability",
]
