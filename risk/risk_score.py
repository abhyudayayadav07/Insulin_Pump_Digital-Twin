"""
Risk scoring for the insulin-pump Digital Twin security/risk layer.

Combines severity and likelihood into an interpretable engineering risk score.
This is separate from the reactive-safe digital twin and is not a clinical
risk score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    from .severity import SeverityAssessment, SeverityLevel, SEVERITY_SCORES
    from .likelihood import LikelihoodAssessment, LikelihoodLevel, LIKELIHOOD_SCORES
except ImportError:
    from severity import SeverityAssessment, SeverityLevel, SEVERITY_SCORES
    from likelihood import LikelihoodAssessment, LikelihoodLevel, LIKELIHOOD_SCORES


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


RISK_LEVEL_SCORES = {
    RiskLevel.LOW: 1,
    RiskLevel.MODERATE: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.CRITICAL: 4,
}


def risk_level_from_score(score: float) -> RiskLevel:
    """Map a 1-25 matrix score to a qualitative risk level."""
    score = float(score)
    if score <= 4:
        return RiskLevel.LOW
    if score <= 9:
        return RiskLevel.MODERATE
    if score <= 16:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


@dataclass(frozen=True)
class RiskMatrixCell:
    severity: int
    likelihood: int
    risk_score: int
    level: RiskLevel


def build_risk_matrix() -> List[RiskMatrixCell]:
    """Build the complete 5x5 severity/likelihood matrix."""
    return [
        RiskMatrixCell(
            severity=s,
            likelihood=l,
            risk_score=s * l,
            level=risk_level_from_score(s * l),
        )
        for s in range(1, 6)
        for l in range(1, 6)
    ]


@dataclass
class RiskAssessment:
    """Combined severity, likelihood, and risk assessment."""

    hazard_id: str
    severity_level: SeverityLevel
    severity_score: int
    likelihood_level: LikelihoodLevel
    likelihood_score: int
    risk_score: float
    risk_level: RiskLevel
    likelihood_probability: Optional[float] = None
    confidence: float = 1.0
    rationale: str = ""
    components: Dict[str, Any] = field(default_factory=dict)
    source: str = "severity_likelihood_matrix"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.severity_score not in range(1, 6):
            raise ValueError("severity_score must be between 1 and 5")
        if self.likelihood_score not in range(1, 6):
            raise ValueError("likelihood_score must be between 1 and 5")
        self.risk_score = max(0.0, float(self.risk_score))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        if self.likelihood_probability is not None:
            self.likelihood_probability = max(
                0.0, min(1.0, float(self.likelihood_probability))
            )

    @property
    def requires_attention(self) -> bool:
        return self.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hazard_id": self.hazard_id,
            "severity_level": self.severity_level.value,
            "severity_score": self.severity_score,
            "likelihood_level": self.likelihood_level.value,
            "likelihood_score": self.likelihood_score,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "likelihood_probability": self.likelihood_probability,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "components": dict(self.components),
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RiskAssessment":
        return cls(
            hazard_id=str(data["hazard_id"]),
            severity_level=SeverityLevel(data["severity_level"]),
            severity_score=int(data["severity_score"]),
            likelihood_level=LikelihoodLevel(data["likelihood_level"]),
            likelihood_score=int(data["likelihood_score"]),
            risk_score=float(data["risk_score"]),
            risk_level=RiskLevel(data["risk_level"]),
            likelihood_probability=data.get("likelihood_probability"),
            confidence=float(data.get("confidence", 1.0)),
            rationale=str(data.get("rationale", "")),
            components=dict(data.get("components", {})),
            source=str(data.get("source", "severity_likelihood_matrix")),
            metadata=dict(data.get("metadata", {})),
        )


def calculate_risk_score(severity: int, likelihood: int) -> float:
    """Calculate the conventional severity x likelihood score."""
    if int(severity) not in range(1, 6):
        raise ValueError("severity must be between 1 and 5")
    if int(likelihood) not in range(1, 6):
        raise ValueError("likelihood must be between 1 and 5")
    return float(int(severity) * int(likelihood))


def assess_risk(
    hazard_id: str,
    severity: SeverityAssessment,
    likelihood: LikelihoodAssessment,
    *,
    confidence: Optional[float] = None,
    rationale: str = "",
) -> RiskAssessment:
    """Combine SeverityAssessment and LikelihoodAssessment."""
    score = calculate_risk_score(severity.score, likelihood.score)
    level = risk_level_from_score(score)
    combined_confidence = (
        min(severity.confidence, likelihood.confidence)
        if confidence is None else float(confidence)
    )

    return RiskAssessment(
        hazard_id=hazard_id,
        severity_level=severity.level,
        severity_score=severity.score,
        likelihood_level=likelihood.level,
        likelihood_score=likelihood.score,
        risk_score=score,
        risk_level=level,
        likelihood_probability=likelihood.probability,
        confidence=combined_confidence,
        rationale=rationale,
        components={
            "severity": severity.to_dict(),
            "likelihood": likelihood.to_dict(),
        },
    )


def assess_risk_from_scores(
    hazard_id: str,
    severity: int,
    likelihood: int,
    *,
    likelihood_probability: Optional[float] = None,
    confidence: float = 1.0,
    rationale: str = "",
) -> RiskAssessment:
    """Convenience function for raw ordinal scores."""
    severity_level = next(k for k, v in SEVERITY_SCORES.items() if v == int(severity))
    likelihood_level = next(k for k, v in LIKELIHOOD_SCORES.items() if v == int(likelihood))
    score = calculate_risk_score(severity, likelihood)

    return RiskAssessment(
        hazard_id=hazard_id,
        severity_level=severity_level,
        severity_score=int(severity),
        likelihood_level=likelihood_level,
        likelihood_score=int(likelihood),
        risk_score=score,
        risk_level=risk_level_from_score(score),
        likelihood_probability=likelihood_probability,
        confidence=confidence,
        rationale=rationale,
    )


def probability_adjusted_risk(
    severity: int,
    likelihood_probability: float,
) -> float:
    """
    Supplementary continuous indicator: severity x likelihood probability.
    Range is approximately 0-5.
    """
    if int(severity) not in range(1, 6):
        raise ValueError("severity must be between 1 and 5")
    probability = max(0.0, min(1.0, float(likelihood_probability)))
    return float(int(severity) * probability)


def risk_level_description(level: RiskLevel | str) -> str:
    if not isinstance(level, RiskLevel):
        level = RiskLevel(level)

    return {
        RiskLevel.LOW: "Lower-priority risk under the configured matrix; continue monitoring.",
        RiskLevel.MODERATE: "Meaningful risk requiring monitoring and appropriate mitigation.",
        RiskLevel.HIGH: "High risk requiring active mitigation and safety attention.",
        RiskLevel.CRITICAL: "Critical risk requiring immediate safety-oriented handling.",
    }[level]


class RiskRegistry:
    """Registry of hazard risk assessments."""

    def __init__(self, assessments: Optional[Iterable[RiskAssessment]] = None):
        self._assessments: Dict[str, RiskAssessment] = {}
        for assessment in assessments or []:
            self.register(assessment)

    def register(self, assessment: RiskAssessment) -> None:
        self._assessments[assessment.hazard_id] = assessment

    def get(self, hazard_id: str) -> Optional[RiskAssessment]:
        return self._assessments.get(hazard_id)

    def require(self, hazard_id: str) -> RiskAssessment:
        result = self.get(hazard_id)
        if result is None:
            raise KeyError(f"No risk assessment for hazard: {hazard_id}")
        return result

    def all(self) -> List[RiskAssessment]:
        return list(self._assessments.values())

    def high_risk(self) -> List[RiskAssessment]:
        return [a for a in self._assessments.values() if a.requires_attention]

    def highest(self) -> Optional[RiskAssessment]:
        return max(self._assessments.values(), key=lambda a: a.risk_score, default=None)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {k: v.to_dict() for k, v in self._assessments.items()}

    def __len__(self) -> int:
        return len(self._assessments)


def risk_matrix_table() -> List[Dict[str, Any]]:
    """Return the 25 matrix cells as serializable dictionaries."""
    return [
        {
            "severity": cell.severity,
            "likelihood": cell.likelihood,
            "risk_score": cell.risk_score,
            "risk_level": cell.level.value,
        }
        for cell in build_risk_matrix()
    ]


__all__ = [
    "RiskLevel",
    "RISK_LEVEL_SCORES",
    "RiskMatrixCell",
    "RiskAssessment",
    "RiskRegistry",
    "calculate_risk_score",
    "risk_level_from_score",
    "build_risk_matrix",
    "assess_risk",
    "assess_risk_from_scores",
    "probability_adjusted_risk",
    "risk_level_description",
    "risk_matrix_table",
]
