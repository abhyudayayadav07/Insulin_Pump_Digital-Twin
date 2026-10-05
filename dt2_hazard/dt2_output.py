"""
DT2 output interface.

Canonical structured output from the hazard-function + survival-function
digital twin. DT2 produces safety evidence for the downstream decision
engine; it does not directly command insulin delivery.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class DT2RiskLevel(str, Enum):
    NORMAL = "normal"
    MONITOR = "monitor"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class DT2Urgency(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    IMMEDIATE = "immediate"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SurvivalOutput:
    """Survival/time-to-event evidence for one event."""
    event_id: str
    horizon_minutes: float
    hazard: float
    cumulative_hazard: float
    survival_probability: float
    event_probability: float
    time_to_event_minutes: Optional[float] = None
    median_time_to_event_minutes: Optional[float] = None


@dataclass(frozen=True)
class TrajectoryRiskOutput:
    """Risk information extracted from a predicted future trajectory."""
    horizon_minutes: float
    max_hazard_score: float
    first_unsafe_time_minutes: Optional[float] = None
    unsafe_fraction: float = 0.0
    trajectory_is_safe: bool = True
    predicted_min_glucose: Optional[float] = None
    predicted_max_glucose: Optional[float] = None


@dataclass(frozen=True)
class MitigationOutputSummary:
    """Abstract mitigation recommendation, not an insulin dose."""
    action: str = "monitor"
    deadline_minutes: Optional[float] = None
    authority_restricted: bool = False
    safe_twin_recommended: bool = False
    emergency: bool = False
    reason: str = ""


@dataclass
class DT2Output:
    """
    Canonical DT2 output.

    It separates:
      - current hazard evidence,
      - survival/event probability,
      - time-to-event,
      - future trajectory risk,
      - mitigation urgency,
      - integrity evidence.
    """

    timestamp: Any
    event_id: Optional[str] = None

    hazard_score: float = 0.0
    hazard_probability: float = 0.0
    risk_level: DT2RiskLevel = DT2RiskLevel.NORMAL

    survival_probability: Optional[float] = None
    event_probability: Optional[float] = None
    cumulative_hazard: Optional[float] = None
    time_to_event_minutes: Optional[float] = None
    median_time_to_event_minutes: Optional[float] = None

    trajectory_risk: Optional[TrajectoryRiskOutput] = None
    survival_forecasts: List[SurvivalOutput] = field(default_factory=list)

    mitigation_deadline_minutes: Optional[float] = None
    urgency: DT2Urgency = DT2Urgency.NONE
    mitigation: Optional[MitigationOutputSummary] = None

    sensor_integrity: float = 1.0
    controller_integrity: float = 1.0
    pump_integrity: float = 1.0
    communication_integrity: float = 1.0
    attack_indicator: float = 0.0

    confidence: float = 0.0
    model_version: str = "dt2-0.1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in (
            "hazard_score",
            "hazard_probability",
            "sensor_integrity",
            "controller_integrity",
            "pump_integrity",
            "communication_integrity",
            "attack_indicator",
            "confidence",
        ):
            setattr(self, name, _clip01(getattr(self, name)))

        if self.survival_probability is not None:
            self.survival_probability = _clip01(self.survival_probability)
        if self.event_probability is not None:
            self.event_probability = _clip01(self.event_probability)

    @property
    def predictive_authority_allowed(self) -> bool:
        """Whether current DT2 evidence permits predictive authority."""
        if self.attack_indicator >= 0.5:
            return False
        if min(
            self.sensor_integrity,
            self.controller_integrity,
            self.pump_integrity,
            self.communication_integrity,
        ) < 0.5:
            return False
        return self.risk_level not in (DT2RiskLevel.HIGH, DT2RiskLevel.CRITICAL)

    @property
    def emergency(self) -> bool:
        return (
            self.risk_level == DT2RiskLevel.CRITICAL
            or self.urgency == DT2Urgency.IMMEDIATE
            or (
                self.mitigation is not None
                and self.mitigation.emergency
            )
        )

    def add_survival_forecast(self, forecast: SurvivalOutput) -> None:
        self.survival_forecasts.append(forecast)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["risk_level"] = self.risk_level.value
        result["urgency"] = self.urgency.value
        return result

    def summary(self) -> Dict[str, Any]:
        """Compact evidence representation for the decision engine."""
        return {
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "hazard_score": self.hazard_score,
            "hazard_probability": self.hazard_probability,
            "risk_level": self.risk_level.value,
            "survival_probability": self.survival_probability,
            "event_probability": self.event_probability,
            "time_to_event_minutes": self.time_to_event_minutes,
            "mitigation_deadline_minutes": self.mitigation_deadline_minutes,
            "urgency": self.urgency.value,
            "predictive_authority_allowed": self.predictive_authority_allowed,
            "emergency": self.emergency,
        }


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def risk_level_from_hazard(
    hazard_score: float,
    warning_threshold: float = 0.30,
    high_threshold: float = 0.60,
    critical_threshold: float = 0.85,
) -> DT2RiskLevel:
    """Map hazard intensity to a qualitative risk level."""
    score = _clip01(hazard_score)
    if score >= critical_threshold:
        return DT2RiskLevel.CRITICAL
    if score >= high_threshold:
        return DT2RiskLevel.HIGH
    if score >= warning_threshold:
        return DT2RiskLevel.WARNING
    if score > 0.0:
        return DT2RiskLevel.MONITOR
    return DT2RiskLevel.NORMAL


def urgency_from_time_to_event(
    time_to_event_minutes: Optional[float],
    warning_minutes: float = 30.0,
    high_minutes: float = 15.0,
    immediate_minutes: float = 5.0,
) -> DT2Urgency:
    """Map time-to-event into a qualitative urgency category."""
    if time_to_event_minutes is None:
        return DT2Urgency.NONE
    t = float(time_to_event_minutes)
    if t <= immediate_minutes:
        return DT2Urgency.IMMEDIATE
    if t <= high_minutes:
        return DT2Urgency.HIGH
    if t <= warning_minutes:
        return DT2Urgency.MODERATE
    return DT2Urgency.LOW


def build_dt2_output(
    timestamp: Any,
    hazard_score: float,
    hazard_probability: float = 0.0,
    event_id: Optional[str] = None,
    survival_probability: Optional[float] = None,
    event_probability: Optional[float] = None,
    cumulative_hazard: Optional[float] = None,
    time_to_event_minutes: Optional[float] = None,
    median_time_to_event_minutes: Optional[float] = None,
    mitigation_deadline_minutes: Optional[float] = None,
    confidence: float = 0.0,
    sensor_integrity: float = 1.0,
    controller_integrity: float = 1.0,
    pump_integrity: float = 1.0,
    communication_integrity: float = 1.0,
    attack_indicator: float = 0.0,
    trajectory_risk: Optional[TrajectoryRiskOutput] = None,
    mitigation: Optional[MitigationOutputSummary] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> DT2Output:
    """Construct the standard DT2 evidence object."""
    risk = risk_level_from_hazard(hazard_score)

    if attack_indicator >= 0.5:
        risk = DT2RiskLevel.CRITICAL
    elif min(
        sensor_integrity,
        controller_integrity,
        pump_integrity,
        communication_integrity,
    ) < 0.5:
        if risk in (DT2RiskLevel.NORMAL, DT2RiskLevel.MONITOR, DT2RiskLevel.WARNING):
            risk = DT2RiskLevel.HIGH

    urgency = urgency_from_time_to_event(
        mitigation_deadline_minutes
        if mitigation_deadline_minutes is not None
        else time_to_event_minutes
    )

    return DT2Output(
        timestamp=timestamp,
        event_id=event_id,
        hazard_score=hazard_score,
        hazard_probability=hazard_probability,
        risk_level=risk,
        survival_probability=survival_probability,
        event_probability=event_probability,
        cumulative_hazard=cumulative_hazard,
        time_to_event_minutes=time_to_event_minutes,
        median_time_to_event_minutes=median_time_to_event_minutes,
        mitigation_deadline_minutes=mitigation_deadline_minutes,
        urgency=urgency,
        trajectory_risk=trajectory_risk,
        mitigation=mitigation,
        sensor_integrity=sensor_integrity,
        controller_integrity=controller_integrity,
        pump_integrity=pump_integrity,
        communication_integrity=communication_integrity,
        attack_indicator=attack_indicator,
        confidence=confidence,
        metadata=dict(metadata or {}),
    )
