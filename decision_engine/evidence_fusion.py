from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, x))


def _unique(values: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


class EvidenceType(str, Enum):
    DT1_RESIDUAL = "dt1_residual"
    DT2_DISAGREEMENT = "dt2_disagreement"
    HAZARD_FUNCTION = "hazard_function"
    HAZARD_PROBABILITY = "hazard_probability"
    TRAJECTORY = "trajectory"
    SAFETY_CONSTRAINT = "safety_constraint"
    ATTACK = "attack"
    SENSOR_INTEGRITY = "sensor_integrity"
    CONTROLLER_INTEGRITY = "controller_integrity"
    PUMP_INTEGRITY = "pump_integrity"
    COMMUNICATION = "communication"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class EvidencePolarity(str, Enum):
    SUPPORTS_HAZARD = "supports_hazard"
    SUPPORTS_SAFETY = "supports_safety"
    NEUTRAL = "neutral"


class FusionState(str, Enum):
    NORMAL = "normal"
    MONITOR = "monitor"
    WARNING = "warning"
    BLOCK = "block"
    SAFE_MODE = "safe_mode"
    EMERGENCY_STOP = "emergency_stop"
    UNKNOWN = "unknown"


@dataclass
class EvidenceItem:
    source: str
    evidence_type: EvidenceType | str
    score: float
    polarity: EvidencePolarity | str = EvidencePolarity.SUPPORTS_HAZARD
    reliability: float = 1.0
    confidence: float = 1.0
    weight: float = 1.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = _clip(self.score)
        self.reliability = _clip(self.reliability)
        self.confidence = _clip(self.confidence)
        self.weight = max(0.0, float(self.weight))

    @property
    def effective_strength(self) -> float:
        return self.score * self.reliability * self.confidence * self.weight


@dataclass
class FusionConfig:
    monitor_threshold: float = 0.25
    warning_threshold: float = 0.45
    block_threshold: float = 0.65
    safe_mode_threshold: float = 0.80
    emergency_stop_threshold: float = 0.92
    minimum_total_evidence: float = 0.10
    agreement_bonus: float = 0.10
    disagreement_penalty: float = 0.05
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "dt1_residual": 1.00,
        "dt2_disagreement": 1.15,
        "hazard_function": 1.10,
        "hazard_probability": 1.15,
        "trajectory": 1.00,
        "safety_constraint": 1.35,
        "attack": 1.30,
        "sensor_integrity": 1.15,
        "controller_integrity": 1.20,
        "pump_integrity": 1.30,
        "communication": 0.90,
        "manual": 1.00,
        "unknown": 0.50,
    })


@dataclass
class FusionResult:
    fused_score: float
    confidence: float
    state: FusionState
    total_evidence: float
    hazard_support: float
    safety_support: float
    evidence_count: int
    supporting_sources: List[str] = field(default_factory=list)
    safety_sources: List[str] = field(default_factory=list)
    violated_constraints: List[str] = field(default_factory=list)
    attack_indicators: List[str] = field(default_factory=list)
    source_scores: Dict[str, float] = field(default_factory=dict)
    contributions: Dict[str, float] = field(default_factory=dict)
    rationale: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_actionable(self) -> bool:
        return self.state not in {FusionState.NORMAL, FusionState.UNKNOWN}

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["state"] = self.state.value
        return d


class EvidenceFusionEngine:
    """Interpretable fusion layer for DT/security evidence."""

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self.config = config or FusionConfig()

    def _source_weight(self, item: EvidenceItem) -> float:
        key = item.evidence_type.value if isinstance(item.evidence_type, EvidenceType) else str(item.evidence_type)
        return self.config.source_weights.get(key, self.config.source_weights["unknown"])

    def add_evidence(self, evidence: EvidenceItem, collection: Optional[List[EvidenceItem]] = None) -> List[EvidenceItem]:
        if collection is None:
            collection = []
        collection.append(evidence)
        return collection

    def fuse(self, evidence: Sequence[EvidenceItem] | Iterable[EvidenceItem], *, metadata: Optional[Mapping[str, Any]] = None) -> FusionResult:
        items = list(evidence)
        if not items:
            return FusionResult(0.0, 0.0, FusionState.UNKNOWN, 0.0, 0.0, 0.0, 0,
                                rationale=["No evidence was supplied."], metadata=dict(metadata or {}))

        hazard_support = 0.0
        safety_support = 0.0
        hazard_items, safety_items = [], []
        source_scores, contributions = {}, {}
        violated, attacks = [], []

        for item in items:
            contribution = item.effective_strength * self._source_weight(item)
            polarity = item.polarity.value if isinstance(item.polarity, EvidencePolarity) else str(item.polarity)
            if polarity == EvidencePolarity.SUPPORTS_SAFETY.value:
                safety_support += contribution
                safety_items.append(item)
            elif polarity == EvidencePolarity.SUPPORTS_HAZARD.value:
                hazard_support += contribution
                hazard_items.append(item)

            key = item.source or str(item.evidence_type)
            contributions[key] = contributions.get(key, 0.0) + contribution
            etype = item.evidence_type.value if isinstance(item.evidence_type, EvidenceType) else str(item.evidence_type)
            source_scores[etype] = max(source_scores.get(etype, 0.0), contribution)

            if etype == EvidenceType.SAFETY_CONSTRAINT.value:
                name = item.metadata.get("constraint_id") or item.metadata.get("constraint") or item.description or item.source
                if name:
                    violated.append(str(name))
            if etype == EvidenceType.ATTACK.value:
                name = item.metadata.get("attack_type") or item.metadata.get("attack_id") or item.description or item.source
                if name:
                    attacks.append(str(name))

        total = hazard_support + safety_support
        if total:
            score = ((hazard_support - safety_support) / total + 1.0) / 2.0
        else:
            score = 0.0

        types = {item.evidence_type.value if isinstance(item.evidence_type, EvidenceType) else str(item.evidence_type) for item in hazard_items}
        if len(types) >= 2 and hazard_support > safety_support:
            score += self.config.agreement_bonus
        if hazard_items and safety_items:
            score -= self.config.disagreement_penalty
        score = _clip(score)

        mean_conf = sum(i.confidence for i in items) / len(items)
        diversity = min(1.0, len(types) / 3.0)
        confidence = _clip(0.7 * mean_conf + 0.3 * diversity)

        state = self._state_from_score(score) if total >= self.config.minimum_total_evidence else FusionState.UNKNOWN
        if state == FusionState.UNKNOWN:
            confidence *= 0.5

        rationale = [f"Fused hazard score={score:.3f} from {len(items)} evidence item(s)."]
        rationale.append(f"Hazard support={hazard_support:.3f}; safety support={safety_support:.3f}.")
        if len(types) >= 2:
            rationale.append("Multiple independent evidence categories support the assessment.")
        if violated:
            rationale.append(f"{len(set(violated))} safety constraint violation(s) recorded.")
        if attacks:
            rationale.append(f"{len(set(attacks))} attack indicator(s) recorded.")
        rationale.append(f"Fusion state={state.value}.")

        return FusionResult(
            fused_score=score, confidence=confidence, state=state,
            total_evidence=total, hazard_support=hazard_support, safety_support=safety_support,
            evidence_count=len(items),
            supporting_sources=[i.source for i in hazard_items],
            safety_sources=[i.source for i in safety_items],
            violated_constraints=_unique(violated), attack_indicators=_unique(attacks),
            source_scores=source_scores, contributions=contributions,
            rationale=rationale, metadata=dict(metadata or {}),
        )

    def _state_from_score(self, score: float) -> FusionState:
        if score >= self.config.emergency_stop_threshold:
            return FusionState.EMERGENCY_STOP
        if score >= self.config.safe_mode_threshold:
            return FusionState.SAFE_MODE
        if score >= self.config.block_threshold:
            return FusionState.BLOCK
        if score >= self.config.warning_threshold:
            return FusionState.WARNING
        if score >= self.config.monitor_threshold:
            return FusionState.MONITOR
        return FusionState.NORMAL


def make_dt1_residual_evidence(residual_score: float, *, confidence: float = 1.0, reliability: float = 1.0, description: str = "") -> EvidenceItem:
    return EvidenceItem("DT1", EvidenceType.DT1_RESIDUAL, residual_score, confidence=confidence, reliability=reliability,
                        description=description or "DT1 predictive-model residual")


def make_dt2_disagreement_evidence(disagreement_score: float, *, confidence: float = 1.0, reliability: float = 1.0, description: str = "") -> EvidenceItem:
    return EvidenceItem("DT1_vs_DT2", EvidenceType.DT2_DISAGREEMENT, disagreement_score, confidence=confidence, reliability=reliability,
                        description=description or "Predictive/reactive twin disagreement")


def make_hazard_probability_evidence(probability: float, *, confidence: float = 1.0, reliability: float = 1.0, description: str = "") -> EvidenceItem:
    return EvidenceItem("HazardProbability", EvidenceType.HAZARD_PROBABILITY, probability, confidence=confidence, reliability=reliability,
                        description=description or "Estimated hazard probability")


def make_constraint_violation_evidence(violation_score: float = 1.0, *, constraint_id: str = "", confidence: float = 1.0, description: str = "") -> EvidenceItem:
    return EvidenceItem(constraint_id or "SafetyConstraint", EvidenceType.SAFETY_CONSTRAINT, violation_score,
                        confidence=confidence, description=description or "Safety constraint violation",
                        metadata={"constraint_id": constraint_id})


def make_attack_evidence(attack_score: float = 1.0, *, attack_type: str = "", confidence: float = 1.0, reliability: float = 1.0, description: str = "") -> EvidenceItem:
    return EvidenceItem(attack_type or "AttackDetector", EvidenceType.ATTACK, attack_score, confidence=confidence,
                        reliability=reliability, description=description or "Cyber/attack indicator",
                        metadata={"attack_type": attack_type})


def fuse_evidence(evidence: Sequence[EvidenceItem] | Iterable[EvidenceItem], config: Optional[FusionConfig] = None) -> FusionResult:
    return EvidenceFusionEngine(config).fuse(evidence)


__all__ = [
    "EvidenceType", "EvidencePolarity", "FusionState", "EvidenceItem", "FusionConfig",
    "FusionResult", "EvidenceFusionEngine", "make_dt1_residual_evidence",
    "make_dt2_disagreement_evidence", "make_hazard_probability_evidence",
    "make_constraint_violation_evidence", "make_attack_evidence", "fuse_evidence",
]
