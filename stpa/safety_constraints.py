"""
Safety constraints for the insulin-pump Digital Twin hazard model.

This module provides the STPA-derived safety-constraint knowledge base used by
DT2. It is intentionally separate from the final decision engine:

    Hazards -> UCAs -> Causal Scenarios -> Safety Constraints -> Runtime Checks

The default constraints describe what the controller, sensor, pump,
communication layer, configuration layer, and safety supervisor must or must
not do under defined conditions.

The runtime checker is deliberately generic. It can evaluate simple numeric
conditions from a signal dictionary, while the richer semantic interpretation
can be performed by the decision engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from datetime import datetime


class ConstraintType(str, Enum):
    """Category of safety constraint."""

    STATE = "state"
    CONTROL_ACTION = "control_action"
    TIMING = "timing"
    RANGE = "range"
    AUTHENTICATION = "authentication"
    COMMUNICATION = "communication"
    SENSOR_VALIDATION = "sensor_validation"
    ACTUATION = "actuation"
    SAFETY_RESPONSE = "safety_response"
    CONFIGURATION = "configuration"
    MODEL_CONSISTENCY = "model_consistency"


class ConstraintStatus(str, Enum):
    """Result of evaluating a safety constraint."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


class ConstraintPriority(str, Enum):
    """Operational priority of a safety constraint."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyConstraint:
    """
    STPA-derived safety constraint.

    `condition` describes when the constraint applies.
    `required_behavior` describes the safe behavior that must occur.
    `prohibited_behavior` describes behavior that must not occur.
    """

    constraint_id: str
    name: str
    description: str
    constraint_type: ConstraintType
    condition: str
    required_behavior: str
    prohibited_behavior: str = ""

    related_hazards: List[str] = field(default_factory=list)
    related_uca_ids: List[str] = field(default_factory=list)
    related_scenario_ids: List[str] = field(default_factory=list)

    monitored_signals: List[str] = field(default_factory=list)
    thresholds: Dict[str, Any] = field(default_factory=dict)

    priority: ConstraintPriority = ConstraintPriority.HIGH
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the constraint to a JSON-compatible dictionary."""
        return {
            "constraint_id": self.constraint_id,
            "name": self.name,
            "description": self.description,
            "constraint_type": self.constraint_type.value,
            "condition": self.condition,
            "required_behavior": self.required_behavior,
            "prohibited_behavior": self.prohibited_behavior,
            "related_hazards": list(self.related_hazards),
            "related_uca_ids": list(self.related_uca_ids),
            "related_scenario_ids": list(self.related_scenario_ids),
            "monitored_signals": list(self.monitored_signals),
            "thresholds": dict(self.thresholds),
            "priority": self.priority.value,
            "enabled": self.enabled,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SafetyConstraint":
        """Create a constraint from a dictionary."""
        return cls(
            constraint_id=str(data["constraint_id"]),
            name=str(data["name"]),
            description=str(data["description"]),
            constraint_type=ConstraintType(data["constraint_type"]),
            condition=str(data.get("condition", "")),
            required_behavior=str(data.get("required_behavior", "")),
            prohibited_behavior=str(data.get("prohibited_behavior", "")),
            related_hazards=list(data.get("related_hazards", [])),
            related_uca_ids=list(data.get("related_uca_ids", [])),
            related_scenario_ids=list(data.get("related_scenario_ids", [])),
            monitored_signals=list(data.get("monitored_signals", [])),
            thresholds=dict(data.get("thresholds", {})),
            priority=ConstraintPriority(
                data.get("priority", ConstraintPriority.HIGH.value)
            ),
            enabled=bool(data.get("enabled", True)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ConstraintCheckResult:
    """Result of a runtime safety-constraint evaluation."""

    constraint_id: str
    status: ConstraintStatus
    violation_score: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    timestamp: Optional[datetime] = None

    @property
    def violated(self) -> bool:
        return self.status == ConstraintStatus.VIOLATED

    @property
    def satisfied(self) -> bool:
        return self.status == ConstraintStatus.SATISFIED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "status": self.status.value,
            "violation_score": float(self.violation_score),
            "evidence": dict(self.evidence),
            "message": self.message,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


def _check_numeric_threshold(
    value: Any,
    threshold: Any,
    operator: str,
) -> Optional[bool]:
    """Evaluate a simple numeric threshold expression."""
    try:
        value = float(value)
        threshold = float(threshold)
    except (TypeError, ValueError):
        return None

    operators = {
        "<": value < threshold,
        "<=": value <= threshold,
        ">": value > threshold,
        ">=": value >= threshold,
        "==": value == threshold,
        "!=": value != threshold,
    }
    return operators.get(operator)


def check_threshold(
    constraint: SafetyConstraint,
    signals: Mapping[str, Any],
) -> ConstraintCheckResult:
    """
    Evaluate a generic threshold/range constraint.

    Supported threshold specifications:

        {"signal": "glucose", "operator": "<=", "value": 70}
        {"signal": "insulin_command", "operator": "<=", "value": 5}

    For constraints without a machine-readable threshold, UNKNOWN is returned.
    """
    timestamp = datetime.utcnow()

    threshold = constraint.thresholds
    signal = threshold.get("signal")

    if not signal:
        return ConstraintCheckResult(
            constraint_id=constraint.constraint_id,
            status=ConstraintStatus.UNKNOWN,
            message="No machine-readable signal threshold is defined.",
            timestamp=timestamp,
        )

    if signal not in signals:
        return ConstraintCheckResult(
            constraint_id=constraint.constraint_id,
            status=ConstraintStatus.UNKNOWN,
            message=f"Required signal '{signal}' is unavailable.",
            evidence={"available_signals": list(signals.keys())},
            timestamp=timestamp,
        )

    operator = str(threshold.get("operator", "<="))
    limit = threshold.get("value")

    result = _check_numeric_threshold(signals[signal], limit, operator)

    if result is None:
        return ConstraintCheckResult(
            constraint_id=constraint.constraint_id,
            status=ConstraintStatus.UNKNOWN,
            message=f"Signal '{signal}' or threshold is not numeric.",
            evidence={"signal": signal, "value": signals.get(signal), "threshold": limit},
            timestamp=timestamp,
        )

    # A threshold in this knowledge base describes the safe condition.
    status = ConstraintStatus.SATISFIED if result else ConstraintStatus.VIOLATED
    violation_score = 0.0 if result else 1.0

    return ConstraintCheckResult(
        constraint_id=constraint.constraint_id,
        status=status,
        violation_score=violation_score,
        evidence={
            "signal": signal,
            "value": signals[signal],
            "operator": operator,
            "threshold": limit,
        },
        message=(
            "Safety constraint satisfied."
            if result
            else "Safety constraint violated."
        ),
        timestamp=timestamp,
    )


class SafetyConstraintRegistry:
    """Registry and lookup interface for safety constraints."""

    def __init__(
        self,
        constraints: Optional[Iterable[SafetyConstraint]] = None,
    ):
        self._constraints: Dict[str, SafetyConstraint] = {}
        for constraint in constraints or []:
            self.register(constraint)

    def register(self, constraint: SafetyConstraint) -> None:
        if constraint.constraint_id in self._constraints:
            raise ValueError(
                f"Constraint already registered: {constraint.constraint_id}"
            )
        self._constraints[constraint.constraint_id] = constraint

    def upsert(self, constraint: SafetyConstraint) -> None:
        self._constraints[constraint.constraint_id] = constraint

    def get(self, constraint_id: str) -> Optional[SafetyConstraint]:
        return self._constraints.get(constraint_id)

    def require(self, constraint_id: str) -> SafetyConstraint:
        constraint = self.get(constraint_id)
        if constraint is None:
            raise KeyError(f"Unknown safety constraint: {constraint_id}")
        return constraint

    def all(self, enabled_only: bool = False) -> List[SafetyConstraint]:
        constraints = list(self._constraints.values())
        if enabled_only:
            constraints = [c for c in constraints if c.enabled]
        return constraints

    def by_type(
        self,
        constraint_type: ConstraintType,
        enabled_only: bool = True,
    ) -> List[SafetyConstraint]:
        return [
            c for c in self.all(enabled_only=enabled_only)
            if c.constraint_type == constraint_type
        ]

    def by_priority(
        self,
        priority: ConstraintPriority,
        enabled_only: bool = True,
    ) -> List[SafetyConstraint]:
        return [
            c for c in self.all(enabled_only=enabled_only)
            if c.priority == priority
        ]

    def by_hazard(
        self,
        hazard_id: str,
        enabled_only: bool = True,
    ) -> List[SafetyConstraint]:
        return [
            c for c in self.all(enabled_only=enabled_only)
            if hazard_id in c.related_hazards
        ]

    def by_uca(
        self,
        uca_id: str,
        enabled_only: bool = True,
    ) -> List[SafetyConstraint]:
        return [
            c for c in self.all(enabled_only=enabled_only)
            if uca_id in c.related_uca_ids
        ]

    def by_scenario(
        self,
        scenario_id: str,
        enabled_only: bool = True,
    ) -> List[SafetyConstraint]:
        return [
            c for c in self.all(enabled_only=enabled_only)
            if scenario_id in c.related_scenario_ids
        ]

    def check(
        self,
        signals: Mapping[str, Any],
        constraint_ids: Optional[Sequence[str]] = None,
    ) -> List[ConstraintCheckResult]:
        """
        Evaluate selected machine-readable constraints.

        Constraints without a numeric threshold return UNKNOWN rather than
        pretending that a semantic safety condition was evaluated.
        """
        constraints = (
            [self.require(cid) for cid in constraint_ids]
            if constraint_ids is not None
            else self.all(enabled_only=True)
        )
        return [check_threshold(c, signals) for c in constraints]

    def violated(
        self,
        signals: Mapping[str, Any],
    ) -> List[ConstraintCheckResult]:
        return [
            result
            for result in self.check(signals)
            if result.status == ConstraintStatus.VIOLATED
        ]

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            constraint_id: constraint.to_dict()
            for constraint_id, constraint in self._constraints.items()
        }

    def __len__(self) -> int:
        return len(self._constraints)

    def __iter__(self):
        return iter(self._constraints.values())


# ---------------------------------------------------------------------------
# Default STPA-derived safety constraints
# ---------------------------------------------------------------------------

DEFAULT_SAFETY_CONSTRAINTS: List[SafetyConstraint] = [
    SafetyConstraint(
        constraint_id="SC01",
        name="Prevent insulin delivery during unsafe low glucose",
        description=(
            "Insulin must not be delivered when current glucose or validated "
            "trajectory indicates a clinically unsafe low-glucose condition."
        ),
        constraint_type=ConstraintType.CONTROL_ACTION,
        condition="Low glucose hazard is active or impending.",
        required_behavior=(
            "Withhold or appropriately reduce insulin and enter an appropriate "
            "safety response."
        ),
        prohibited_behavior="Deliver insulin without a valid safety basis.",
        related_hazards=["H01", "H03"],
        related_uca_ids=["UCA02", "UCA12"],
        related_scenario_ids=["CS01", "CS03", "CS04", "CS11"],
        monitored_signals=["glucose", "predicted_glucose", "insulin_command"],
        priority=ConstraintPriority.CRITICAL,
    ),
    SafetyConstraint(
        constraint_id="SC02",
        name="Prevent insulin overdelivery",
        description=(
            "The commanded insulin dose must remain within the configured "
            "safe operating envelope."
        ),
        constraint_type=ConstraintType.RANGE,
        condition="An insulin command is generated.",
        required_behavior="Keep insulin delivery within validated dose limits.",
        prohibited_behavior="Issue a dose above the configured safe limit.",
        related_hazards=["H03"],
        related_uca_ids=["UCA02", "UCA05", "UCA08"],
        related_scenario_ids=["CS04", "CS06", "CS10"],
        monitored_signals=["insulin_command"],
        thresholds={
            "signal": "insulin_command",
            "operator": "<=",
            "value": 5.0,
            "unit": "U",
        },
        priority=ConstraintPriority.CRITICAL,
    ),
    SafetyConstraint(
        constraint_id="SC03",
        name="Provide insulin when required",
        description=(
            "When validated physiological state requires insulin, the control "
            "system must not silently suppress a required delivery."
        ),
        constraint_type=ConstraintType.CONTROL_ACTION,
        condition="Validated hyperglycemic state requires insulin.",
        required_behavior="Provide an appropriate insulin command.",
        prohibited_behavior="Withhold required insulin without a safety reason.",
        related_hazards=["H02", "H04"],
        related_uca_ids=["UCA01"],
        related_scenario_ids=["CS07", "CS08"],
        monitored_signals=["glucose", "insulin_command"],
        priority=ConstraintPriority.HIGH,
    ),
    SafetyConstraint(
        constraint_id="SC04",
        name="Validate CGM measurements before control",
        description=(
            "CGM values used by the controller must pass plausibility and "
            "freshness checks before influencing insulin delivery."
        ),
        constraint_type=ConstraintType.SENSOR_VALIDATION,
        condition="A new CGM measurement is received.",
        required_behavior="Validate freshness, plausibility, continuity, and integrity.",
        prohibited_behavior="Use known invalid, stale, or spoofed CGM data for control.",
        related_hazards=["H05", "H01", "H02"],
        related_uca_ids=["UCA09"],
        related_scenario_ids=["CS01", "CS02", "CS03", "CS12"],
        monitored_signals=["cgm", "cgm_age", "cgm_quality"],
        priority=ConstraintPriority.CRITICAL,
    ),
    SafetyConstraint(
        constraint_id="SC05",
        name="Reject stale CGM data",
        description="Stale sensor observations must not be treated as current measurements.",
        constraint_type=ConstraintType.SENSOR_VALIDATION,
        condition="CGM observation age exceeds the freshness limit.",
        required_behavior="Reject or quarantine the stale observation.",
        prohibited_behavior="Feed stale CGM data directly into safety-critical control.",
        related_hazards=["H05", "H01", "H02"],
        related_uca_ids=["UCA09"],
        related_scenario_ids=["CS02", "CS12"],
        monitored_signals=["cgm_age"],
        thresholds={
            "signal": "cgm_age",
            "operator": "<=",
            "value": 10.0,
            "unit": "minutes",
        },
        priority=ConstraintPriority.HIGH,
    ),
    SafetyConstraint(
        constraint_id="SC06",
        name="Bound insulin delivery rate",
        description=(
            "Actual pump delivery must remain within the validated physical "
            "and configured delivery envelope."
        ),
        constraint_type=ConstraintType.ACTUATION,
        condition="The pump is operating.",
        required_behavior="Deliver insulin within the validated actuator envelope.",
        prohibited_behavior="Continue uncontrolled overdelivery or unexpected delivery.",
        related_hazards=["H03", "H07"],
        related_uca_ids=["UCA02", "UCA05"],
        related_scenario_ids=["CS06", "CS07"],
        monitored_signals=["insulin_delivery_rate"],
        thresholds={
            "signal": "insulin_delivery_rate",
            "operator": "<=",
            "value": 5.0,
            "unit": "U/h",
        },
        priority=ConstraintPriority.CRITICAL,
    ),
    SafetyConstraint(
        constraint_id="SC07",
        name="Detect commanded-versus-actual delivery mismatch",
        description=(
            "A persistent mismatch between commanded insulin and observed "
            "delivery must trigger monitoring and appropriate safety handling."
        ),
        constraint_type=ConstraintType.ACTUATION,
        condition="Commanded and observed delivery are available.",
        required_behavior="Detect significant persistent delivery mismatch.",
        prohibited_behavior="Ignore sustained actuator underdelivery or overdelivery.",
        related_hazards=["H03", "H04", "H07"],
        related_uca_ids=["UCA01", "UCA02"],
        related_scenario_ids=["CS06", "CS07"],
        monitored_signals=["insulin_command", "insulin_actual", "delivery_residual"],
        priority=ConstraintPriority.HIGH,
    ),
    SafetyConstraint(
        constraint_id="SC08",
        name="Fail safely on controller malfunction",
        description=(
            "Controller faults or inconsistent control outputs must not result "
            "in unrestricted insulin delivery."
        ),
        constraint_type=ConstraintType.SAFETY_RESPONSE,
        condition="Controller malfunction or unsafe control behavior is detected.",
        required_behavior="Transition to monitoring, safe mode, or pump stop as required.",
        prohibited_behavior="Continue unrestricted autonomous control after a critical fault.",
        related_hazards=["H06", "H01", "H03", "H04"],
        related_uca_ids=["UCA12"],
        related_scenario_ids=["CS04", "CS05", "CS12"],
        monitored_signals=["controller_fault", "controller_residual", "hazard_score"],
        priority=ConstraintPriority.CRITICAL,
    ),
    SafetyConstraint(
        constraint_id="SC09",
        name="Authenticate safety-critical commands",
        description=(
            "Safety-critical commands and configuration changes must originate "
            "from an authorized source."
        ),
        constraint_type=ConstraintType.AUTHENTICATION,
        condition="A safety-critical command or configuration update is received.",
        required_behavior="Verify source authorization and command integrity.",
        prohibited_behavior="Accept unauthenticated or unauthorized safety-critical commands.",
        related_hazards=["H08", "H06", "H07"],
        related_uca_ids=["UCA11", "UCA14"],
        related_scenario_ids=["CS03", "CS09", "CS10"],
        monitored_signals=["command_authenticated", "command_integrity"],
        priority=ConstraintPriority.CRITICAL,
    ),
    SafetyConstraint(
        constraint_id="SC10",
        name="Reject replayed or stale control commands",
        description=(
            "Previously valid commands must not be accepted again outside "
            "their valid execution window."
        ),
        constraint_type=ConstraintType.TIMING,
        condition="A remote or communication-mediated control command is received.",
        required_behavior="Validate freshness, sequence, nonce, or equivalent replay protection.",
        prohibited_behavior="Execute an old command solely because its format is valid.",
        related_hazards=["H08"],
        related_uca_ids=["UCA14"],
        related_scenario_ids=["CS09"],
        monitored_signals=["command_age", "sequence_valid", "nonce_valid"],
        priority=ConstraintPriority.CRITICAL,
    ),
    SafetyConstraint(
        constraint_id="SC11",
        name="Protect against communication loss",
        description=(
            "Loss or corruption of safety-critical communication must not leave "
            "the system in an uncontrolled state."
        ),
        constraint_type=ConstraintType.COMMUNICATION,
        condition="Communication becomes unavailable or unreliable.",
        required_behavior="Detect the communication fault and transition to a safe behavior.",
        prohibited_behavior="Assume missing or corrupted commands are valid.",
        related_hazards=["H08", "H06", "H07"],
        related_uca_ids=["UCA01", "UCA14"],
        related_scenario_ids=["CS08"],
        monitored_signals=["communication_available", "packet_loss", "message_integrity"],
        priority=ConstraintPriority.HIGH,
    ),
    SafetyConstraint(
        constraint_id="SC12",
        name="Control configuration changes",
        description=(
            "Safety-relevant configuration changes must be validated before "
            "they influence the controller."
        ),
        constraint_type=ConstraintType.CONFIGURATION,
        condition="A safety-relevant configuration update is requested.",
        required_behavior="Authenticate, validate, log, and apply only permitted changes.",
        prohibited_behavior="Apply arbitrary or unauthorized configuration changes.",
        related_hazards=["H06", "H08"],
        related_uca_ids=["UCA11"],
        related_scenario_ids=["CS10"],
        monitored_signals=["configuration_authenticated", "configuration_valid"],
        priority=ConstraintPriority.CRITICAL,
    ),
    SafetyConstraint(
        constraint_id="SC13",
        name="Issue timely safety stop",
        description=(
            "When evidence indicates a critical unsafe state, the safety "
            "supervisor must issue the appropriate stop or safe-mode action "
            "without an unjustified delay."
        ),
        constraint_type=ConstraintType.SAFETY_RESPONSE,
        condition="Critical hazard evidence exceeds the safety-response threshold.",
        required_behavior="Issue the required safety response within the validated response time.",
        prohibited_behavior="Continue unsafe operation after a confirmed critical hazard.",
        related_hazards=["H01", "H03", "H06", "H07", "H08"],
        related_uca_ids=["UCA12"],
        related_scenario_ids=["CS12"],
        monitored_signals=["hazard_probability", "hazard_state", "response_latency"],
        thresholds={
            "signal": "response_latency",
            "operator": "<=",
            "value": 30.0,
            "unit": "seconds",
        },
        priority=ConstraintPriority.CRITICAL,
    ),
    SafetyConstraint(
        constraint_id="SC14",
        name="Maintain DT1 model consistency",
        description=(
            "A large persistent discrepancy between observed glucose behavior "
            "and DT1 predictions must be treated as evidence requiring monitoring."
        ),
        constraint_type=ConstraintType.MODEL_CONSISTENCY,
        condition="DT1 prediction and actual glucose are both available.",
        required_behavior="Monitor residuals and escalate persistent abnormal behavior.",
        prohibited_behavior="Treat a persistently inconsistent model as trustworthy without qualification.",
        related_hazards=["H01", "H02", "H05", "H06", "H08"],
        related_uca_ids=["UCA09", "UCA12"],
        related_scenario_ids=["CS03", "CS04", "CS11"],
        monitored_signals=["glucose", "predicted_glucose", "dt1_residual"],
        priority=ConstraintPriority.HIGH,
    ),
    SafetyConstraint(
        constraint_id="SC15",
        name="Respect hazard trajectory escalation",
        description=(
            "When predicted hazard probability or intensity is trending toward "
            "a critical state, the system must increase monitoring and prepare "
            "the corresponding safety response."
        ),
        constraint_type=ConstraintType.SAFETY_RESPONSE,
        condition="Hazard trajectory predicts escalation toward an unsafe state.",
        required_behavior="Increase monitoring and apply the validated escalation policy.",
        prohibited_behavior="Ignore a sustained predicted escalation in hazard state.",
        related_hazards=["H01", "H02", "H03", "H04", "H08"],
        related_uca_ids=["UCA12"],
        related_scenario_ids=["CS11", "CS12"],
        monitored_signals=["hazard_probability", "predicted_hazard_probability"],
        priority=ConstraintPriority.HIGH,
    ),
]


def get_default_safety_constraints() -> List[SafetyConstraint]:
    """Return independent copies of the default constraints."""
    return [
        SafetyConstraint.from_dict(constraint.to_dict())
        for constraint in DEFAULT_SAFETY_CONSTRAINTS
    ]


def get_safety_constraint_registry() -> SafetyConstraintRegistry:
    """Return a registry populated with the default STPA constraints."""
    return SafetyConstraintRegistry(get_default_safety_constraints())


def constraints_to_dict(
    constraints: Optional[Iterable[SafetyConstraint]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Serialize constraints into a dictionary keyed by constraint ID."""
    if constraints is None:
        constraints = DEFAULT_SAFETY_CONSTRAINTS

    return {
        constraint.constraint_id: constraint.to_dict()
        for constraint in constraints
    }


__all__ = [
    "ConstraintType",
    "ConstraintStatus",
    "ConstraintPriority",
    "SafetyConstraint",
    "ConstraintCheckResult",
    "SafetyConstraintRegistry",
    "check_threshold",
    "DEFAULT_SAFETY_CONSTRAINTS",
    "get_default_safety_constraints",
    "get_safety_constraint_registry",
    "constraints_to_dict",
]
