"""
Severity assessment for the insulin-pump Digital Twin risk layer.

This module represents the consequence/severity dimension of DT2 risk
assessment. Severity is deliberately separated from likelihood/probability:

    Hazard -> Severity
            -> Likelihood
            -> Risk score

The default severity model is a structured consequence assessment based on
physiological impact, safety impact, duration, detectability, and affected
system scope. It is a research framework, not a clinical risk scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional


class SeverityLevel(str, Enum):
    """Ordinal severity levels for hazardous outcomes."""

    NEGLIGIBLE = "negligible"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


SEVERITY_SCORES: Dict[SeverityLevel, int] = {
    SeverityLevel.NEGLIGIBLE: 1,
    SeverityLevel.MINOR: 2,
    SeverityLevel.MODERATE: 3,
    SeverityLevel.MAJOR: 4,
    SeverityLevel.CRITICAL: 5,
}


@dataclass(frozen=True)
class SeverityScale:
    """Definition of the ordinal severity scale."""

    level: SeverityLevel
    score: int
    description: str


DEFAULT_SEVERITY_SCALE: List[SeverityScale] = [
    SeverityScale(
        SeverityLevel.NEGLIGIBLE,
        1,
        "Little or no direct safety consequence; transient deviation.",
    ),
    SeverityScale(
        SeverityLevel.MINOR,
        2,
        "Limited safety impact that is unlikely to produce serious harm.",
    ),
    SeverityScale(
        SeverityLevel.MODERATE,
        3,
        "Meaningful safety impact requiring intervention or increased monitoring.",
    ),
    SeverityScale(
        SeverityLevel.MAJOR,
        4,
        "Serious hazardous consequence with substantial potential for harm.",
    ),
    SeverityScale(
        SeverityLevel.CRITICAL,
        5,
        "Potentially life-threatening or catastrophic safety consequence.",
    ),
]


@dataclass
class SeverityAssessment:
    """Result of a severity assessment."""

    hazard_id: str
    level: SeverityLevel
    score: int
    rationale: str = ""
    consequence_factors: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "rule_based"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        expected_score = SEVERITY_SCORES[self.level]
        if int(self.score) != expected_score:
            raise ValueError(
                f"Score {self.score} does not match severity "
                f"{self.level.value} ({expected_score})."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hazard_id": self.hazard_id,
            "level": self.level.value,
            "score": self.score,
            "rationale": self.rationale,
            "consequence_factors": dict(self.consequence_factors),
            "confidence": self.confidence,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SeverityAssessment":
        return cls(
            hazard_id=str(data["hazard_id"]),
            level=SeverityLevel(data["level"]),
            score=int(data["score"]),
            rationale=str(data.get("rationale", "")),
            consequence_factors=dict(data.get("consequence_factors", {})),
            confidence=float(data.get("confidence", 1.0)),
            source=str(data.get("source", "rule_based")),
            metadata=dict(data.get("metadata", {})),
        )


def severity_from_score(score: float) -> SeverityLevel:
    """Map a numeric 1-5 severity score to an ordinal severity level."""
    value = float(score)

    if value <= 1.0:
        return SeverityLevel.NEGLIGIBLE
    if value <= 2.0:
        return SeverityLevel.MINOR
    if value <= 3.0:
        return SeverityLevel.MODERATE
    if value <= 4.0:
        return SeverityLevel.MAJOR
    return SeverityLevel.CRITICAL


def severity_score(level: SeverityLevel | str) -> int:
    """Return the numeric score associated with a severity level."""
    if not isinstance(level, SeverityLevel):
        level = SeverityLevel(level)
    return SEVERITY_SCORES[level]


def severity_description(level: SeverityLevel | str) -> str:
    """Return the standard description for a severity level."""
    if not isinstance(level, SeverityLevel):
        level = SeverityLevel(level)

    for item in DEFAULT_SEVERITY_SCALE:
        if item.level == level:
            return item.description

    return ""


# ---------------------------------------------------------------------------
# Consequence factors
# ---------------------------------------------------------------------------

@dataclass
class ConsequenceFactors:
    """
    Structured factors used to support a severity assessment.

    All factors use a 0-1 scale unless otherwise specified.
    """

    physiological_impact: float = 0.0
    safety_impact: float = 0.0
    duration: float = 0.0
    affected_scope: float = 0.0
    reversibility: float = 1.0
    detectability: float = 1.0
    intervention_required: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "physiological_impact",
            "safety_impact",
            "duration",
            "affected_scope",
            "reversibility",
            "detectability",
            "intervention_required",
        ):
            value = float(getattr(self, name))
            setattr(self, name, max(0.0, min(1.0, value)))

    def to_dict(self) -> Dict[str, float]:
        return {
            "physiological_impact": self.physiological_impact,
            "safety_impact": self.safety_impact,
            "duration": self.duration,
            "affected_scope": self.affected_scope,
            "reversibility": self.reversibility,
            "detectability": self.detectability,
            "intervention_required": self.intervention_required,
        }


def consequence_score(factors: ConsequenceFactors) -> float:
    """
    Compute a normalized consequence score.

    Higher physiological/safety impact, duration, affected scope, and
    intervention requirement increase severity.

    Low reversibility and low detectability also increase concern.
    """
    score = (
        0.28 * factors.physiological_impact
        + 0.24 * factors.safety_impact
        + 0.12 * factors.duration
        + 0.10 * factors.affected_scope
        + 0.10 * (1.0 - factors.reversibility)
        + 0.08 * (1.0 - factors.detectability)
        + 0.08 * factors.intervention_required
    )

    return max(0.0, min(1.0, score))


def assess_consequence_factors(
    hazard_id: str,
    factors: ConsequenceFactors,
    rationale: str = "",
    confidence: float = 1.0,
    source: str = "rule_based",
) -> SeverityAssessment:
    """Convert consequence factors into an ordinal severity assessment."""
    normalized = consequence_score(factors)

    # Convert [0, 1] to a 1-5 ordinal score.
    ordinal_score = min(5, max(1, int(normalized * 5.0) + 1))
    level = severity_from_score(ordinal_score)

    return SeverityAssessment(
        hazard_id=hazard_id,
        level=level,
        score=SEVERITY_SCORES[level],
        rationale=rationale,
        consequence_factors=factors.to_dict(),
        confidence=confidence,
        source=source,
    )


# ---------------------------------------------------------------------------
# Default hazard severity mapping
# ---------------------------------------------------------------------------

DEFAULT_HAZARD_SEVERITY: Dict[str, SeverityLevel] = {
    "H01": SeverityLevel.CRITICAL,   # Severe hypoglycemia
    "H02": SeverityLevel.CRITICAL,   # Severe hyperglycemia
    "H03": SeverityLevel.CRITICAL,   # Insulin overdelivery
    "H04": SeverityLevel.MAJOR,      # Insulin underdelivery
    "H05": SeverityLevel.MAJOR,      # Glucose sensor failure
    "H06": SeverityLevel.MAJOR,      # Controller malfunction
    "H07": SeverityLevel.MAJOR,      # Pump delivery failure
    "H08": SeverityLevel.CRITICAL,   # Cyber-induced unsafe control
}


class SeverityRegistry:
    """Registry of default or custom hazard severity assessments."""

    def __init__(
        self,
        assessments: Optional[Iterable[SeverityAssessment]] = None,
    ):
        self._assessments: Dict[str, SeverityAssessment] = {}

        for assessment in assessments or []:
            self.register(assessment)

    def register(self, assessment: SeverityAssessment) -> None:
        self._assessments[assessment.hazard_id] = assessment

    def get(self, hazard_id: str) -> Optional[SeverityAssessment]:
        return self._assessments.get(hazard_id)

    def require(self, hazard_id: str) -> SeverityAssessment:
        result = self.get(hazard_id)
        if result is None:
            raise KeyError(f"No severity assessment for hazard: {hazard_id}")
        return result

    def all(self) -> List[SeverityAssessment]:
        return list(self._assessments.values())

    def highest(self) -> Optional[SeverityAssessment]:
        if not self._assessments:
            return None

        return max(
            self._assessments.values(),
            key=lambda assessment: assessment.score,
        )

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            hazard_id: assessment.to_dict()
            for hazard_id, assessment in self._assessments.items()
        }

    def __len__(self) -> int:
        return len(self._assessments)


def get_default_severity_assessments() -> List[SeverityAssessment]:
    """
    Build the default severity assessments for H01-H08.

    These are baseline research assumptions and should be reviewed against
    the project's formal hazard analysis and domain-expert criteria.
    """
    rationales = {
        "H01": "Severe hypoglycemia can produce a critical physiological safety consequence.",
        "H02": "Severe hyperglycemia can produce a serious physiological safety consequence.",
        "H03": "Insulin overdelivery can directly cause dangerous glucose reduction.",
        "H04": "Insulin underdelivery can permit sustained hyperglycemia and associated harm.",
        "H05": "Sensor failure can corrupt the feedback signal used by the controller.",
        "H06": "Controller malfunction can generate unsafe or inappropriate control actions.",
        "H07": "Pump delivery failure can cause unintended overdelivery, underdelivery, or loss of therapy.",
        "H08": "Cyber-induced unsafe control can deliberately or unintentionally cause hazardous actuation.",
    }

    assessments = []

    for hazard_id, level in DEFAULT_HAZARD_SEVERITY.items():
        assessments.append(
            SeverityAssessment(
                hazard_id=hazard_id,
                level=level,
                score=SEVERITY_SCORES[level],
                rationale=rationales.get(hazard_id, ""),
                consequence_factors={
                    "basis": "default_hazard_consequence_classification",
                    "requires_domain_validation": True,
                },
                confidence=0.75,
                source="default_stpa_hazard_mapping",
            )
        )

    return assessments


def get_severity_registry() -> SeverityRegistry:
    """Return a registry populated with default hazard severity mappings."""
    return SeverityRegistry(get_default_severity_assessments())


def assess_hazard_severity(
    hazard_id: str,
    *,
    factors: Optional[ConsequenceFactors] = None,
    registry: Optional[SeverityRegistry] = None,
) -> SeverityAssessment:
    """
    Obtain severity for a hazard.

    If consequence factors are supplied, they are assessed dynamically.
    Otherwise the default hazard mapping is used.
    """
    if factors is not None:
        return assess_consequence_factors(
            hazard_id=hazard_id,
            factors=factors,
        )

    registry = registry or get_severity_registry()
    return registry.require(hazard_id)


def severity_table() -> List[Dict[str, Any]]:
    """Return the default severity scale as serializable records."""
    return [
        {
            "level": item.level.value,
            "score": item.score,
            "description": item.description,
        }
        for item in DEFAULT_SEVERITY_SCALE
    ]


__all__ = [
    "SeverityLevel",
    "SEVERITY_SCORES",
    "SeverityScale",
    "DEFAULT_SEVERITY_SCALE",
    "SeverityAssessment",
    "ConsequenceFactors",
    "SeverityRegistry",
    "DEFAULT_HAZARD_SEVERITY",
    "severity_from_score",
    "severity_score",
    "severity_description",
    "consequence_score",
    "assess_consequence_factors",
    "get_default_severity_assessments",
    "get_severity_registry",
    "assess_hazard_severity",
    "severity_table",
]
