"""
DT2 Hazard Definition Module
============================

Defines the safety hazards monitored by Digital Twin 2 (DT2).

DT2 is responsible for hazard/risk reasoning, not glucose prediction.
This module provides a structured representation of hazards so that
later modules can estimate hazard state, probability, and risk.

The definitions are intentionally configurable and independent of any
specific ML model or simulator implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional


class HazardSeverity(str, Enum):
    """Qualitative severity levels for a hazard."""

    NEGLIGIBLE = "negligible"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class HazardType(str, Enum):
    """High-level categories of insulin-pump safety hazards."""

    HYPOGLYCEMIA = "hypoglycemia"
    HYPERGLYCEMIA = "hyperglycemia"
    INSULIN_OVERDELIVERY = "insulin_overdelivery"
    INSULIN_UNDERDELIVERY = "insulin_underdelivery"
    SENSOR_FAILURE = "sensor_failure"
    CONTROLLER_FAILURE = "controller_failure"
    PUMP_FAILURE = "pump_failure"
    COMMUNICATION_FAILURE = "communication_failure"
    CYBER_ATTACK = "cyber_attack"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HazardDefinition:
    """
    Formal definition of one safety hazard.

    Parameters
    ----------
    hazard_id:
        Unique identifier, e.g. ``H01``.
    name:
        Human-readable hazard name.
    hazard_type:
        Category of the hazard.
    description:
        What the hazard represents.
    unsafe_state:
        Observable unsafe system/physiological state.
    potential_consequences:
        Possible adverse consequences if the hazard is not mitigated.
    indicators:
        Signals/features that can provide evidence for the hazard.
    severity:
        Qualitative severity level.
    enabled:
        Whether DT2 should monitor this hazard.
    metadata:
        Optional additional information such as STPA references,
        thresholds, provenance, or scenario labels.

    Notes
    -----
    This class does not calculate probability or risk. Those operations
    belong to ``hazard_probability.py`` and ``risk/`` modules.
    """

    hazard_id: str
    name: str
    hazard_type: HazardType
    description: str
    unsafe_state: str
    potential_consequences: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    severity: HazardSeverity = HazardSeverity.MODERATE
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hazard_id.strip():
            raise ValueError("hazard_id must not be empty.")
        if not self.name.strip():
            raise ValueError("name must not be empty.")
        if not self.description.strip():
            raise ValueError("description must not be empty.")
        if not self.unsafe_state.strip():
            raise ValueError("unsafe_state must not be empty.")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON/YAML-friendly dictionary representation."""
        data = asdict(self)
        data["hazard_type"] = self.hazard_type.value
        data["severity"] = self.severity.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HazardDefinition":
        """Construct a hazard definition from a dictionary."""
        values = dict(data)

        hazard_type = values.get("hazard_type", HazardType.UNKNOWN)
        if not isinstance(hazard_type, HazardType):
            hazard_type = HazardType(str(hazard_type))

        severity = values.get("severity", HazardSeverity.MODERATE)
        if not isinstance(severity, HazardSeverity):
            severity = HazardSeverity(str(severity))

        return cls(
            hazard_id=str(values["hazard_id"]),
            name=str(values["name"]),
            hazard_type=hazard_type,
            description=str(values["description"]),
            unsafe_state=str(values["unsafe_state"]),
            potential_consequences=list(
                values.get("potential_consequences", [])
            ),
            indicators=list(values.get("indicators", [])),
            severity=severity,
            enabled=bool(values.get("enabled", True)),
            metadata=dict(values.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Default insulin-pump safety hazards
# ---------------------------------------------------------------------------

DEFAULT_HAZARDS: List[HazardDefinition] = [
    HazardDefinition(
        hazard_id="H01",
        name="Severe Hypoglycemia",
        hazard_type=HazardType.HYPOGLYCEMIA,
        description=(
            "The patient glucose trajectory enters or is predicted to enter "
            "a dangerously low range."
        ),
        unsafe_state=(
            "Current or predicted glucose is below the configured "
            "hypoglycemia safety boundary."
        ),
        potential_consequences=[
            "Loss of consciousness",
            "Seizure",
            "Neurological injury",
            "Severe adverse patient outcome",
        ],
        indicators=[
            "glucose",
            "cgm",
            "predicted_glucose",
            "glucose_rate_of_change",
        ],
        severity=HazardSeverity.CRITICAL,
        metadata={"default_glucose_boundary_mg_dl": 70.0},
    ),
    HazardDefinition(
        hazard_id="H02",
        name="Severe Hyperglycemia",
        hazard_type=HazardType.HYPERGLYCEMIA,
        description=(
            "The patient glucose trajectory enters or is predicted to enter "
            "a dangerously high range."
        ),
        unsafe_state=(
            "Current or predicted glucose exceeds the configured "
            "hyperglycemia safety boundary."
        ),
        potential_consequences=[
            "Prolonged hyperglycemia",
            "Metabolic deterioration",
            "Potential diabetic ketoacidosis",
            "Severe adverse patient outcome",
        ],
        indicators=[
            "glucose",
            "cgm",
            "predicted_glucose",
            "glucose_rate_of_change",
        ],
        severity=HazardSeverity.CRITICAL,
        metadata={"default_glucose_boundary_mg_dl": 250.0},
    ),
    HazardDefinition(
        hazard_id="H03",
        name="Insulin Overdelivery",
        hazard_type=HazardType.INSULIN_OVERDELIVERY,
        description=(
            "Insulin delivered by the pump is substantially greater than "
            "the expected or authorized insulin delivery."
        ),
        unsafe_state=(
            "Observed insulin delivery deviates beyond the configured "
            "safe delivery tolerance."
        ),
        potential_consequences=[
            "Rapid glucose reduction",
            "Hypoglycemia",
            "Loss of glucose control",
        ],
        indicators=[
            "insulin",
            "commanded_insulin",
            "delivered_insulin",
            "insulin_residual",
            "predicted_glucose",
        ],
        severity=HazardSeverity.CRITICAL,
    ),
    HazardDefinition(
        hazard_id="H04",
        name="Insulin Underdelivery",
        hazard_type=HazardType.INSULIN_UNDERDELIVERY,
        description=(
            "The pump delivers substantially less insulin than the "
            "expected or authorized delivery."
        ),
        unsafe_state=(
            "Observed insulin delivery is below the configured safe "
            "delivery tolerance."
        ),
        potential_consequences=[
            "Persistent hyperglycemia",
            "Loss of glucose control",
            "Potential metabolic deterioration",
        ],
        indicators=[
            "insulin",
            "commanded_insulin",
            "delivered_insulin",
            "predicted_glucose",
        ],
        severity=HazardSeverity.MAJOR,
    ),
    HazardDefinition(
        hazard_id="H05",
        name="Glucose Sensor Failure",
        hazard_type=HazardType.SENSOR_FAILURE,
        description=(
            "The glucose sensor provides missing, implausible, stale, or "
            "inconsistent measurements."
        ),
        unsafe_state=(
            "The controller receives glucose information that is unreliable "
            "for safe insulin control."
        ),
        potential_consequences=[
            "Incorrect insulin decision",
            "Delayed detection of glucose deterioration",
            "Hypoglycemia or hyperglycemia",
        ],
        indicators=[
            "cgm",
            "glucose",
            "cgm_glucose_residual",
            "missing_sensor_data",
            "sensor_age",
        ],
        severity=HazardSeverity.MAJOR,
    ),
    HazardDefinition(
        hazard_id="H06",
        name="Controller Malfunction",
        hazard_type=HazardType.CONTROLLER_FAILURE,
        description=(
            "The insulin controller produces an incorrect, unsafe, or "
            "unexpected control command."
        ),
        unsafe_state=(
            "Controller output violates configured control or safety "
            "constraints."
        ),
        potential_consequences=[
            "Insulin overdelivery",
            "Insulin underdelivery",
            "Unsafe glucose trajectory",
        ],
        indicators=[
            "commanded_insulin",
            "controller_output",
            "control_residual",
            "predicted_glucose",
        ],
        severity=HazardSeverity.CRITICAL,
    ),
    HazardDefinition(
        hazard_id="H07",
        name="Pump Delivery Failure",
        hazard_type=HazardType.PUMP_FAILURE,
        description=(
            "The physical pump does not deliver the insulin commanded by "
            "the controller."
        ),
        unsafe_state=(
            "Actual pump delivery differs materially from commanded "
            "delivery."
        ),
        potential_consequences=[
            "Insulin underdelivery",
            "Insulin overdelivery",
            "Loss of glucose regulation",
        ],
        indicators=[
            "commanded_insulin",
            "delivered_insulin",
            "pump_delivery_residual",
        ],
        severity=HazardSeverity.MAJOR,
    ),
    HazardDefinition(
        hazard_id="H08",
        name="Cyber-Induced Unsafe Control",
        hazard_type=HazardType.CYBER_ATTACK,
        description=(
            "A cyber-induced manipulation causes the insulin delivery "
            "system to enter or approach an unsafe state."
        ),
        unsafe_state=(
            "A malicious or unauthorized modification of sensing, control, "
            "communication, or actuation creates unsafe system behavior."
        ),
        potential_consequences=[
            "Insulin overdelivery",
            "Insulin underdelivery",
            "Unsafe glucose trajectory",
            "Patient harm",
        ],
        indicators=[
            "sensor_attack",
            "controller_attack",
            "pump_attack",
            "command_deviation",
            "model_residual",
            "hazard_probability",
        ],
        severity=HazardSeverity.CRITICAL,
    ),
]


class HazardRegistry:
    """Registry for creating, retrieving, filtering, and managing hazards."""

    def __init__(
        self,
        hazards: Optional[Iterable[HazardDefinition]] = None,
    ) -> None:
        self._hazards: Dict[str, HazardDefinition] = {}

        for hazard in hazards or DEFAULT_HAZARDS:
            self.register(hazard)

    def register(self, hazard: HazardDefinition) -> None:
        """Add a hazard definition to the registry."""
        if hazard.hazard_id in self._hazards:
            raise ValueError(
                f"Hazard '{hazard.hazard_id}' is already registered."
            )
        self._hazards[hazard.hazard_id] = hazard

    def get(self, hazard_id: str) -> HazardDefinition:
        """Retrieve a hazard by ID."""
        try:
            return self._hazards[hazard_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown hazard_id: '{hazard_id}'. "
                f"Available IDs: {list(self._hazards)}"
            ) from exc

    def get_all(self, enabled_only: bool = False) -> List[HazardDefinition]:
        """Return all registered hazards."""
        hazards = list(self._hazards.values())
        if enabled_only:
            hazards = [hazard for hazard in hazards if hazard.enabled]
        return hazards

    def get_by_type(
        self,
        hazard_type: HazardType,
        enabled_only: bool = False,
    ) -> List[HazardDefinition]:
        """Return hazards belonging to a specific category."""
        return [
            hazard
            for hazard in self.get_all(enabled_only=enabled_only)
            if hazard.hazard_type == hazard_type
        ]

    def get_by_severity(
        self,
        severity: HazardSeverity,
        enabled_only: bool = False,
    ) -> List[HazardDefinition]:
        """Return hazards with the requested severity."""
        return [
            hazard
            for hazard in self.get_all(enabled_only=enabled_only)
            if hazard.severity == severity
        ]

    def __len__(self) -> int:
        return len(self._hazards)

    def __contains__(self, hazard_id: str) -> bool:
        return hazard_id in self._hazards


def get_default_hazards() -> List[HazardDefinition]:
    """Return fresh copies of the default hazard definitions."""
    return [
        HazardDefinition.from_dict(hazard.to_dict())
        for hazard in DEFAULT_HAZARDS
    ]


def get_hazard_registry() -> HazardRegistry:
    """Create a registry populated with the default hazards."""
    return HazardRegistry(get_default_hazards())


def hazards_to_dict(
    hazards: Iterable[HazardDefinition],
) -> List[Dict[str, Any]]:
    """Convert hazard definitions to serializable dictionaries."""
    return [hazard.to_dict() for hazard in hazards]


__all__ = [
    "HazardSeverity",
    "HazardType",
    "HazardDefinition",
    "HazardRegistry",
    "DEFAULT_HAZARDS",
    "get_default_hazards",
    "get_hazard_registry",
    "hazards_to_dict",
]
