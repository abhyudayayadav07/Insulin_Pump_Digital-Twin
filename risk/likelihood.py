"""
Likelihood estimation for the insulin-pump Digital Twin risk layer.

This module estimates how likely a hazard is to occur or become active based
on observable evidence, historical occurrence information, temporal trends,
and configurable weighting.

The module is intentionally separate from severity and final risk scoring:

    Hazard -> Severity
            -> Likelihood  <- this module
            -> Risk Score

IMPORTANT:
The default likelihood mapping is an evidence-based research model. It is
not a clinically calibrated probability unless validated against appropriate
empirical data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import exp, log
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


class LikelihoodLevel(str, Enum):
    """Ordinal likelihood categories."""

    RARE = "rare"
    UNLIKELY = "unlikely"
    POSSIBLE = "possible"
    LIKELY = "likely"
    ALMOST_CERTAIN = "almost_certain"


LIKELIHOOD_SCORES: Dict[LikelihoodLevel, int] = {
    LikelihoodLevel.RARE: 1,
    LikelihoodLevel.UNLIKELY: 2,
    LikelihoodLevel.POSSIBLE: 3,
    LikelihoodLevel.LIKELY: 4,
    LikelihoodLevel.ALMOST_CERTAIN: 5,
}


@dataclass(frozen=True)
class LikelihoodScale:
    """Definition of one ordinal likelihood level."""

    level: LikelihoodLevel
    score: int
    description: str
    probability_lower: float
    probability_upper: float


DEFAULT_LIKELIHOOD_SCALE: List[LikelihoodScale] = [
    LikelihoodScale(
        LikelihoodLevel.RARE,
        1,
        "Very low likelihood under the evaluated operating conditions.",
        0.00,
        0.05,
    ),
    LikelihoodScale(
        LikelihoodLevel.UNLIKELY,
        2,
        "Low likelihood, but the event remains plausible.",
        0.05,
        0.20,
    ),
    LikelihoodScale(
        LikelihoodLevel.POSSIBLE,
        3,
        "Meaningful possibility under the evaluated conditions.",
        0.20,
        0.50,
    ),
    LikelihoodScale(
        LikelihoodLevel.LIKELY,
        4,
        "High likelihood under the evaluated conditions.",
        0.50,
        0.80,
    ),
    LikelihoodScale(
        LikelihoodLevel.ALMOST_CERTAIN,
        5,
        "Very high likelihood under the evaluated conditions.",
        0.80,
        1.00,
    ),
]


@dataclass
class LikelihoodEvidence:
    """
    Evidence used to estimate hazard likelihood.

    Values are normalized to [0, 1].

    - current_evidence: present hazard indicators.
    - historical_frequency: empirical occurrence information, if available.
    - trend_strength: evidence that the hazard is becoming more likely.
    - persistence: repeated/consecutive observations.
    - exposure: opportunity for the hazard to occur.
    - uncertainty: uncertainty in the evidence/model.
    """

    current_evidence: float = 0.0
    historical_frequency: float = 0.0
    trend_strength: float = 0.0
    persistence: float = 0.0
    exposure: float = 0.0
    uncertainty: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "current_evidence",
            "historical_frequency",
            "trend_strength",
            "persistence",
            "exposure",
            "uncertainty",
        ):
            value = float(getattr(self, name))
            setattr(self, name, max(0.0, min(1.0, value)))

    def to_dict(self) -> Dict[str, float]:
        return {
            "current_evidence": self.current_evidence,
            "historical_frequency": self.historical_frequency,
            "trend_strength": self.trend_strength,
            "persistence": self.persistence,
            "exposure": self.exposure,
            "uncertainty": self.uncertainty,
        }


@dataclass
class LikelihoodConfig:
    """Configuration for the likelihood evidence model."""

    current_weight: float = 0.35
    historical_weight: float = 0.20
    trend_weight: float = 0.15
    persistence_weight: float = 0.15
    exposure_weight: float = 0.15

    uncertainty_penalty: float = 0.10

    rare_threshold: float = 0.05
    unlikely_threshold: float = 0.20
    possible_threshold: float = 0.50
    likely_threshold: float = 0.80

    min_probability: float = 0.001
    max_probability: float = 0.999

    def normalized_weights(self) -> Dict[str, float]:
        weights = {
            "current_evidence": max(0.0, self.current_weight),
            "historical_frequency": max(0.0, self.historical_weight),
            "trend_strength": max(0.0, self.trend_weight),
            "persistence": max(0.0, self.persistence_weight),
            "exposure": max(0.0, self.exposure_weight),
        }

        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError("At least one likelihood weight must be positive.")

        return {key: value / total for key, value in weights.items()}


@dataclass
class LikelihoodAssessment:
    """Result of a likelihood estimation."""

    hazard_id: str
    probability: float
    level: LikelihoodLevel
    score: int

    evidence: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    confidence: float = 1.0
    source: str = "evidence_model"
    calibrated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.probability = max(0.0, min(1.0, float(self.probability)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

        expected_score = LIKELIHOOD_SCORES[self.level]
        if int(self.score) != expected_score:
            raise ValueError(
                f"Score {self.score} does not match likelihood "
                f"{self.level.value} ({expected_score})."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hazard_id": self.hazard_id,
            "probability": self.probability,
            "level": self.level.value,
            "score": self.score,
            "evidence": dict(self.evidence),
            "rationale": self.rationale,
            "confidence": self.confidence,
            "source": self.source,
            "calibrated": self.calibrated,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LikelihoodAssessment":
        return cls(
            hazard_id=str(data["hazard_id"]),
            probability=float(data["probability"]),
            level=LikelihoodLevel(data["level"]),
            score=int(data["score"]),
            evidence=dict(data.get("evidence", {})),
            rationale=str(data.get("rationale", "")),
            confidence=float(data.get("confidence", 1.0)),
            source=str(data.get("source", "evidence_model")),
            calibrated=bool(data.get("calibrated", False)),
            metadata=dict(data.get("metadata", {})),
        )


def clamp_probability(
    probability: float,
    minimum: float = 0.001,
    maximum: float = 0.999,
) -> float:
    """Clamp a probability to a valid configurable interval."""
    if minimum < 0.0 or maximum > 1.0 or minimum >= maximum:
        raise ValueError("Invalid probability bounds.")

    return max(minimum, min(maximum, float(probability)))


def probability_to_level(
    probability: float,
    config: Optional[LikelihoodConfig] = None,
) -> LikelihoodLevel:
    """Convert probability to an ordinal likelihood category."""
    config = config or LikelihoodConfig()
    p = float(probability)

    if p < config.rare_threshold:
        return LikelihoodLevel.RARE
    if p < config.unlikely_threshold:
        return LikelihoodLevel.UNLIKELY
    if p < config.possible_threshold:
        return LikelihoodLevel.POSSIBLE
    if p < config.likely_threshold:
        return LikelihoodLevel.LIKELY

    return LikelihoodLevel.ALMOST_CERTAIN


def likelihood_score(
    level: LikelihoodLevel | str,
) -> int:
    """Return the ordinal 1-5 score for a likelihood level."""
    if not isinstance(level, LikelihoodLevel):
        level = LikelihoodLevel(level)

    return LIKELIHOOD_SCORES[level]


def likelihood_description(
    level: LikelihoodLevel | str,
) -> str:
    """Return the standard description for a likelihood level."""
    if not isinstance(level, LikelihoodLevel):
        level = LikelihoodLevel(level)

    for item in DEFAULT_LIKELIHOOD_SCALE:
        if item.level == level:
            return item.description

    return ""


def weighted_evidence_score(
    evidence: LikelihoodEvidence,
    config: Optional[LikelihoodConfig] = None,
) -> float:
    """
    Compute the weighted likelihood evidence score.

    Uncertainty reduces the resulting score but does not automatically mean
    that an event is unlikely.
    """
    config = config or LikelihoodConfig()
    weights = config.normalized_weights()

    score = (
        weights["current_evidence"] * evidence.current_evidence
        + weights["historical_frequency"] * evidence.historical_frequency
        + weights["trend_strength"] * evidence.trend_strength
        + weights["persistence"] * evidence.persistence
        + weights["exposure"] * evidence.exposure
    )

    penalty = config.uncertainty_penalty * evidence.uncertainty
    return max(0.0, min(1.0, score - penalty))


def evidence_to_probability(
    evidence: LikelihoodEvidence,
    config: Optional[LikelihoodConfig] = None,
) -> float:
    """
    Convert normalized likelihood evidence to an evidence-model probability.

    This is not a calibrated clinical probability unless the model is
    validated/calibrated against empirical event data.
    """
    config = config or LikelihoodConfig()

    score = weighted_evidence_score(evidence, config)

    # Smooth the evidence score while retaining its [0, 1] ordering.
    # The transformation avoids an overly sharp threshold around 0.5.
    centered = (score - 0.5) * 6.0
    probability = 1.0 / (1.0 + exp(-centered))

    # Preserve a small amount of evidence at the extremes.
    probability = 0.02 * score + 0.96 * probability

    return clamp_probability(
        probability,
        config.min_probability,
        config.max_probability,
    )


def assess_likelihood(
    hazard_id: str,
    evidence: LikelihoodEvidence,
    *,
    config: Optional[LikelihoodConfig] = None,
    confidence: float = 1.0,
    source: str = "evidence_model",
    calibrated: bool = False,
    rationale: str = "",
) -> LikelihoodAssessment:
    """Estimate likelihood for a hazard from structured evidence."""
    config = config or LikelihoodConfig()

    probability = evidence_to_probability(evidence, config)
    level = probability_to_level(probability, config)

    return LikelihoodAssessment(
        hazard_id=hazard_id,
        probability=probability,
        level=level,
        score=LIKELIHOOD_SCORES[level],
        evidence=evidence.to_dict(),
        rationale=rationale,
        confidence=confidence,
        source=source,
        calibrated=calibrated,
    )


# ---------------------------------------------------------------------------
# Historical frequency utilities
# ---------------------------------------------------------------------------

def historical_frequency(
    occurrences: int,
    opportunities: int,
) -> float:
    """
    Estimate empirical event frequency.

    This is a frequency estimate, not necessarily a calibrated probability.
    """
    if occurrences < 0 or opportunities < 0:
        raise ValueError("Occurrences and opportunities must be non-negative.")

    if opportunities == 0:
        return 0.0

    return max(0.0, min(1.0, occurrences / float(opportunities)))


def bayesian_frequency(
    occurrences: int,
    opportunities: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 9.0,
) -> float:
    """
    Simple Beta-binomial posterior mean for event frequency.

    The default prior has mean 0.10 and is intended only as a transparent
    research baseline.
    """
    if occurrences < 0 or opportunities < 0 or occurrences > opportunities:
        raise ValueError("Invalid occurrence/opportunity counts.")

    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Beta prior parameters must be positive.")

    return (
        prior_alpha + occurrences
    ) / (
        prior_alpha + prior_beta + opportunities
    )


# ---------------------------------------------------------------------------
# Stateful likelihood estimator
# ---------------------------------------------------------------------------

@dataclass
class LikelihoodObservation:
    """One historical likelihood observation."""

    timestamp: Any
    evidence: float
    occurred: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LikelihoodEstimator:
    """
    Stateful estimator for hazard likelihood.

    The estimator maintains recent evidence and optionally observed event
    outcomes. It can be used by DT2 while processing a simulation trajectory.
    """

    def __init__(
        self,
        hazard_id: str,
        config: Optional[LikelihoodConfig] = None,
        history_limit: int = 100,
    ):
        self.hazard_id = hazard_id
        self.config = config or LikelihoodConfig()
        self.history_limit = max(1, int(history_limit))
        self._observations: List[LikelihoodObservation] = []

    def add_observation(
        self,
        timestamp: Any,
        evidence: float,
        occurred: Optional[bool] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Add one observation to the estimator history."""
        observation = LikelihoodObservation(
            timestamp=timestamp,
            evidence=max(0.0, min(1.0, float(evidence))),
            occurred=occurred,
            metadata=dict(metadata or {}),
        )

        self._observations.append(observation)

        if len(self._observations) > self.history_limit:
            self._observations = self._observations[-self.history_limit :]

    def history(self) -> List[LikelihoodObservation]:
        """Return a copy of the current observation history."""
        return list(self._observations)

    def empirical_frequency(self) -> float:
        """Return observed event frequency when outcomes are available."""
        outcomes = [
            observation.occurred
            for observation in self._observations
            if observation.occurred is not None
        ]

        if not outcomes:
            return 0.0

        return sum(bool(value) for value in outcomes) / len(outcomes)

    def recent_evidence(self, window: Optional[int] = None) -> float:
        """Return the mean recent evidence."""
        observations = self._observations[-window:] if window else self._observations

        if not observations:
            return 0.0

        return sum(item.evidence for item in observations) / len(observations)

    def persistence(self) -> float:
        """Measure the fraction of recent observations carrying evidence."""
        if not self._observations:
            return 0.0

        return sum(
            item.evidence >= 0.5
            for item in self._observations
        ) / len(self._observations)

    def estimate(
        self,
        *,
        current_evidence: Optional[float] = None,
        trend_strength: float = 0.0,
        exposure: float = 0.0,
        uncertainty: float = 0.0,
    ) -> LikelihoodAssessment:
        """Generate a likelihood estimate from current and historical state."""
        if current_evidence is None:
            current_evidence = self.recent_evidence(window=10)

        evidence = LikelihoodEvidence(
            current_evidence=current_evidence,
            historical_frequency=self.empirical_frequency(),
            trend_strength=trend_strength,
            persistence=self.persistence(),
            exposure=exposure,
            uncertainty=uncertainty,
        )

        return assess_likelihood(
            self.hazard_id,
            evidence,
            config=self.config,
            source="stateful_evidence_model",
            calibrated=False,
        )

    def reset(self) -> None:
        """Clear all observations."""
        self._observations.clear()


def estimate_hazard_likelihood(
    hazard_id: str,
    *,
    current_evidence: float = 0.0,
    historical_frequency_value: float = 0.0,
    trend_strength: float = 0.0,
    persistence: float = 0.0,
    exposure: float = 0.0,
    uncertainty: float = 0.0,
    config: Optional[LikelihoodConfig] = None,
) -> LikelihoodAssessment:
    """Convenience function for one-shot hazard likelihood estimation."""
    evidence = LikelihoodEvidence(
        current_evidence=current_evidence,
        historical_frequency=historical_frequency_value,
        trend_strength=trend_strength,
        persistence=persistence,
        exposure=exposure,
        uncertainty=uncertainty,
    )

    return assess_likelihood(
        hazard_id,
        evidence,
        config=config,
        calibrated=False,
    )


__all__ = [
    "LikelihoodLevel",
    "LIKELIHOOD_SCORES",
    "LikelihoodScale",
    "DEFAULT_LIKELIHOOD_SCALE",
    "LikelihoodEvidence",
    "LikelihoodConfig",
    "LikelihoodAssessment",
    "LikelihoodObservation",
    "LikelihoodEstimator",
    "clamp_probability",
    "probability_to_level",
    "likelihood_score",
    "likelihood_description",
    "weighted_evidence_score",
    "evidence_to_probability",
    "assess_likelihood",
    "historical_frequency",
    "bayesian_frequency",
    "estimate_hazard_likelihood",
]
