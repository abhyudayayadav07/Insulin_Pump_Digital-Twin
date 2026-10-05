"""
DT2 STPA — Unsafe Control Actions
=================================

Defines Unsafe Control Actions (UCAs) for the insulin-pump cyber-physical
system using an STPA-oriented representation.

Purpose
-------
An Unsafe Control Action describes a control action that can contribute to
a hazardous system state.

For the insulin-pump digital twin, this module provides the bridge between:

    STPA safety analysis
            ↓
    Unsafe Control Actions
            ↓
    Causal scenarios
            ↓
    Hazard function / hazard predictor
            ↓
    Risk and safety response

This module defines UCAs. It does not determine whether a UCA is currently
occurring; runtime evidence and hazard-state modules perform that task.

The definitions are intentionally configurable and can be extended as the
system architecture becomes more detailed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional


class UCAType(str, Enum):
    """
    STPA-oriented categories of unsafe control actions.

    The four primary categories are:

    1. NOT_PROVIDED_WHEN_NEEDED
    2. PROVIDED_WHEN_NOT_APPROPRIATE
    3. PROVIDED_TOO_EARLY_TOO_LATE_OR_WRONG_ORDER
    4. APPLIED_TOO_LONG_OR_STOPPED_TOO_SOON
    """

    NOT_PROVIDED_WHEN_NEEDED = "not_provided_when_needed"
    PROVIDED_WHEN_NOT_APPROPRIATE = "provided_when_not_appropriate"
    PROVIDED_TOO_EARLY_LATE_WRONG_ORDER = (
        "provided_too_early_too_late_or_wrong_order"
    )
    APPLIED_TOO_LONG_OR_STOPPED_TOO_SOON = (
        "applied_too_long_or_stopped_too_soon"
    )


class ControlActionType(str, Enum):
    """Control actions relevant to the insulin-pump system."""

    DELIVER_INSULIN = "deliver_insulin"
    WITHHOLD_INSULIN = "withhold_insulin"
    MODIFY_BASAL_RATE = "modify_basal_rate"
    DELIVER_BOLUS = "deliver_bolus"
    UPDATE_CONTROLLER_TARGET = "update_controller_target"
    ACCEPT_CGM_VALUE = "accept_cgm_value"
    REJECT_CGM_VALUE = "reject_cgm_value"
    ENTER_SAFE_MODE = "enter_safe_mode"
    STOP_PUMP = "stop_pump"
    RESUME_PUMP = "resume_pump"
    COMMUNICATE_COMMAND = "communicate_command"
    UPDATE_CONFIGURATION = "update_configuration"


class UCASource(str, Enum):
    """
    Potential source/context of an unsafe control action.

    These labels are useful for connecting STPA analysis to cyber-attack
    modules and system faults.
    """

    CONTROLLER = "controller"
    SENSOR = "sensor"
    PUMP = "pump"
    COMMUNICATION = "communication"
    CONFIGURATION = "configuration"
    CYBER_ATTACK = "cyber_attack"
    HUMAN_OPERATOR = "human_operator"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class UnsafeControlAction:
    """
    Formal representation of one Unsafe Control Action.

    Parameters
    ----------
    uca_id:
        Unique identifier, e.g. ``UCA01``.
    name:
        Human-readable UCA name.
    action_type:
        Control action involved.
    uca_type:
        One of the four primary STPA UCA categories.
    controller:
        Component responsible for issuing or applying the control action.
    description:
        Concrete description of the unsafe action.
    unsafe_context:
        Context in which the control action becomes unsafe.
    related_hazards:
        Hazard IDs from hazard_definition.py.
    source:
        Potential source of the unsafe action.
    safety_constraints:
        Constraints that should prevent the UCA.
    severity:
        Optional qualitative severity label.
    enabled:
        Whether this UCA is currently considered in the analysis.
    metadata:
        Additional STPA/system information.
    """

    uca_id: str
    name: str
    action_type: ControlActionType
    uca_type: UCAType
    controller: str
    description: str
    unsafe_context: str
    related_hazards: List[str] = field(default_factory=list)
    source: UCASource = UCASource.CONTROLLER
    safety_constraints: List[str] = field(default_factory=list)
    severity: str = "major"
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.uca_id.strip():
            raise ValueError("uca_id must not be empty.")

        if not self.name.strip():
            raise ValueError("name must not be empty.")

        if not self.controller.strip():
            raise ValueError("controller must not be empty.")

        if not self.description.strip():
            raise ValueError("description must not be empty.")

        if not self.unsafe_context.strip():
            raise ValueError("unsafe_context must not be empty.")

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary."""
        data = asdict(self)
        data["action_type"] = self.action_type.value
        data["uca_type"] = self.uca_type.value
        data["source"] = self.source.value
        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "UnsafeControlAction":
        """Create a UCA from a dictionary."""
        values = dict(data)

        action_type = values.get(
            "action_type",
            ControlActionType.DELIVER_INSULIN,
        )
        if not isinstance(action_type, ControlActionType):
            action_type = ControlActionType(str(action_type))

        uca_type = values.get(
            "uca_type",
            UCAType.NOT_PROVIDED_WHEN_NEEDED,
        )
        if not isinstance(uca_type, UCAType):
            uca_type = UCAType(str(uca_type))

        source = values.get("source", UCASource.UNKNOWN)
        if not isinstance(source, UCASource):
            source = UCASource(str(source))

        return cls(
            uca_id=str(values["uca_id"]),
            name=str(values["name"]),
            action_type=action_type,
            uca_type=uca_type,
            controller=str(values["controller"]),
            description=str(values["description"]),
            unsafe_context=str(values["unsafe_context"]),
            related_hazards=list(values.get("related_hazards", [])),
            source=source,
            safety_constraints=list(
                values.get("safety_constraints", [])
            ),
            severity=str(values.get("severity", "major")),
            enabled=bool(values.get("enabled", True)),
            metadata=dict(values.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Default STPA-style insulin-pump UCAs
# ---------------------------------------------------------------------------

DEFAULT_UCAS: List[UnsafeControlAction] = [
    UnsafeControlAction(
        uca_id="UCA01",
        name="Insulin Not Delivered When Required",
        action_type=ControlActionType.DELIVER_INSULIN,
        uca_type=UCAType.NOT_PROVIDED_WHEN_NEEDED,
        controller="Insulin Controller",
        description=(
            "The controller fails to command or the pump fails to apply "
            "required insulin delivery when glucose regulation requires it."
        ),
        unsafe_context=(
            "Glucose is elevated or predicted to become elevated and "
            "insulin delivery is required under the active control policy."
        ),
        related_hazards=[
            "H02",
            "H04",
            "H07",
        ],
        source=UCASource.CONTROLLER,
        safety_constraints=[
            "Required insulin delivery shall not be omitted when valid "
            "control conditions require delivery.",
            "The system shall detect prolonged underdelivery.",
        ],
        severity="major",
    ),
    UnsafeControlAction(
        uca_id="UCA02",
        name="Insulin Delivered When Not Appropriate",
        action_type=ControlActionType.DELIVER_INSULIN,
        uca_type=UCAType.PROVIDED_WHEN_NOT_APPROPRIATE,
        controller="Insulin Controller",
        description=(
            "The controller commands insulin when delivery is unsafe or "
            "not supported by the current physiological state."
        ),
        unsafe_context=(
            "Glucose is low, rapidly decreasing, or the predicted trajectory "
            "indicates elevated hypoglycemia risk."
        ),
        related_hazards=[
            "H01",
            "H03",
            "H06",
        ],
        source=UCASource.CONTROLLER,
        safety_constraints=[
            "Insulin delivery shall respect configured glucose safety limits.",
            "The controller shall consider current and predicted glucose "
            "trajectory before delivery.",
        ],
        severity="critical",
    ),
    UnsafeControlAction(
        uca_id="UCA03",
        name="Insulin Command Issued Too Early",
        action_type=ControlActionType.DELIVER_BOLUS,
        uca_type=UCAType.PROVIDED_TOO_EARLY_LATE_WRONG_ORDER,
        controller="Insulin Controller",
        description=(
            "A bolus or increased insulin command is issued before the "
            "control context supports the action."
        ),
        unsafe_context=(
            "The controller reacts before sufficient physiological or sensor "
            "evidence is available."
        ),
        related_hazards=[
            "H01",
            "H03",
        ],
        source=UCASource.CONTROLLER,
        safety_constraints=[
            "Control actions shall use sufficiently recent and valid sensor "
            "information.",
            "Rapid repeated commands shall be subject to timing constraints.",
        ],
        severity="major",
    ),
    UnsafeControlAction(
        uca_id="UCA04",
        name="Insulin Command Issued Too Late",
        action_type=ControlActionType.DELIVER_INSULIN,
        uca_type=UCAType.PROVIDED_TOO_EARLY_LATE_WRONG_ORDER,
        controller="Insulin Controller",
        description=(
            "Required insulin delivery is commanded after the appropriate "
            "control window has been missed."
        ),
        unsafe_context=(
            "Glucose or predicted glucose has already entered a dangerous "
            "trajectory and the control response is delayed."
        ),
        related_hazards=[
            "H02",
            "H04",
        ],
        source=UCASource.CONTROLLER,
        safety_constraints=[
            "The controller shall respond within the defined control period.",
            "Delayed control responses shall be detected.",
        ],
        severity="major",
    ),
    UnsafeControlAction(
        uca_id="UCA05",
        name="Insulin Delivered For Too Long",
        action_type=ControlActionType.DELIVER_INSULIN,
        uca_type=UCAType.APPLIED_TOO_LONG_OR_STOPPED_TOO_SOON,
        controller="Insulin Pump",
        description=(
            "Insulin delivery continues beyond the interval or dose "
            "required by the control command."
        ),
        unsafe_context=(
            "The commanded delivery duration has elapsed or a stop "
            "condition has been reached."
        ),
        related_hazards=[
            "H01",
            "H03",
            "H07",
        ],
        source=UCASource.PUMP,
        safety_constraints=[
            "Delivery shall terminate when the commanded dose or duration "
            "has been reached.",
            "Actual delivery shall be monitored against commanded delivery.",
        ],
        severity="critical",
    ),
    UnsafeControlAction(
        uca_id="UCA06",
        name="Insulin Stopped Too Soon",
        action_type=ControlActionType.DELIVER_INSULIN,
        uca_type=UCAType.APPLIED_TOO_LONG_OR_STOPPED_TOO_SOON,
        controller="Insulin Pump",
        description=(
            "Insulin delivery terminates before the commanded dose or "
            "required delivery duration is completed."
        ),
        unsafe_context=(
            "The patient requires continued insulin delivery and the pump "
            "terminates the action prematurely."
        ),
        related_hazards=[
            "H02",
            "H04",
            "H07",
        ],
        source=UCASource.PUMP,
        safety_constraints=[
            "The pump shall complete authorized delivery unless a valid "
            "higher-priority safety stop is active.",
        ],
        severity="major",
    ),
    UnsafeControlAction(
        uca_id="UCA07",
        name="Unsafe Basal Rate Modification",
        action_type=ControlActionType.MODIFY_BASAL_RATE,
        uca_type=UCAType.PROVIDED_WHEN_NOT_APPROPRIATE,
        controller="Insulin Controller",
        description=(
            "The controller modifies basal insulin to a value outside "
            "the allowed safety envelope."
        ),
        unsafe_context=(
            "The requested basal rate is inconsistent with safety limits "
            "or current physiological evidence."
        ),
        related_hazards=[
            "H01",
            "H02",
            "H03",
            "H04",
        ],
        source=UCASource.CONTROLLER,
        safety_constraints=[
            "Basal rate shall remain within configured safe bounds.",
            "Abrupt changes shall be constrained by the control policy.",
        ],
        severity="critical",
    ),
    UnsafeControlAction(
        uca_id="UCA08",
        name="Unsafe Bolus Command",
        action_type=ControlActionType.DELIVER_BOLUS,
        uca_type=UCAType.PROVIDED_WHEN_NOT_APPROPRIATE,
        controller="Insulin Controller",
        description=(
            "The controller commands a bolus that is excessive, "
            "unauthorized, or inconsistent with the current state."
        ),
        unsafe_context=(
            "The bolus exceeds configured limits or is inconsistent with "
            "glucose and meal information."
        ),
        related_hazards=[
            "H01",
            "H03",
        ],
        source=UCASource.CONTROLLER,
        safety_constraints=[
            "Bolus dose shall remain within configured limits.",
            "Bolus commands shall be checked against recent insulin history.",
        ],
        severity="critical",
    ),
    UnsafeControlAction(
        uca_id="UCA09",
        name="Invalid CGM Value Accepted",
        action_type=ControlActionType.ACCEPT_CGM_VALUE,
        uca_type=UCAType.PROVIDED_WHEN_NOT_APPROPRIATE,
        controller="Insulin Controller",
        description=(
            "The controller accepts an invalid, stale, implausible, or "
            "maliciously manipulated CGM measurement."
        ),
        unsafe_context=(
            "The sensor value violates validity, freshness, physiological, "
            "or consistency checks."
        ),
        related_hazards=[
            "H01",
            "H02",
            "H05",
            "H08",
        ],
        source=UCASource.SENSOR,
        safety_constraints=[
            "Invalid sensor values shall not directly drive insulin control.",
            "Sensor freshness and plausibility shall be checked.",
        ],
        severity="critical",
    ),
    UnsafeControlAction(
        uca_id="UCA10",
        name="Invalid CGM Value Rejected",
        action_type=ControlActionType.REJECT_CGM_VALUE,
        uca_type=UCAType.NOT_PROVIDED_WHEN_NEEDED,
        controller="Insulin Controller",
        description=(
            "A valid CGM measurement is incorrectly rejected, depriving "
            "the controller of required physiological information."
        ),
        unsafe_context=(
            "A valid and timely sensor measurement is available but is "
            "discarded by the sensing or validation pipeline."
        ),
        related_hazards=[
            "H02",
            "H05",
        ],
        source=UCASource.SENSOR,
        safety_constraints=[
            "Valid measurements shall remain available to the controller.",
            "Sensor rejection logic shall be traceable.",
        ],
        severity="major",
    ),
    UnsafeControlAction(
        uca_id="UCA11",
        name="Unsafe Configuration Update",
        action_type=ControlActionType.UPDATE_CONFIGURATION,
        uca_type=UCAType.PROVIDED_WHEN_NOT_APPROPRIATE,
        controller="Pump Configuration Interface",
        description=(
            "A configuration or control parameter is changed to an unsafe "
            "value."
        ),
        unsafe_context=(
            "The requested configuration exceeds an authorized safety "
            "boundary or is introduced without required validation."
        ),
        related_hazards=[
            "H01",
            "H02",
            "H03",
            "H04",
            "H08",
        ],
        source=UCASource.CONFIGURATION,
        safety_constraints=[
            "Configuration changes shall be authenticated and authorized.",
            "Safety-critical parameters shall be range-checked.",
        ],
        severity="critical",
    ),
    UnsafeControlAction(
        uca_id="UCA12",
        name="Safety Stop Not Issued",
        action_type=ControlActionType.STOP_PUMP,
        uca_type=UCAType.NOT_PROVIDED_WHEN_NEEDED,
        controller="Safety Monitor",
        description=(
            "The safety monitor fails to issue a pump stop when evidence "
            "indicates that continued delivery is unsafe."
        ),
        unsafe_context=(
            "Critical hazard evidence is present and the configured "
            "emergency-stop conditions have been met."
        ),
        related_hazards=[
            "H01",
            "H03",
            "H06",
            "H08",
        ],
        source=UCASource.CONTROLLER,
        safety_constraints=[
            "Critical unsafe conditions shall trigger the defined safety "
            "response.",
            "Stop authority shall remain independent of ordinary delivery "
            "logic.",
        ],
        severity="critical",
    ),
    UnsafeControlAction(
        uca_id="UCA13",
        name="Safety Stop Issued Without Valid Basis",
        action_type=ControlActionType.STOP_PUMP,
        uca_type=UCAType.PROVIDED_WHEN_NOT_APPROPRIATE,
        controller="Safety Monitor",
        description=(
            "The safety monitor stops insulin delivery without sufficient "
            "evidence or without satisfying the defined stop conditions."
        ),
        unsafe_context=(
            "The stop command is generated from invalid, stale, or "
            "insufficient evidence."
        ),
        related_hazards=[
            "H02",
            "H04",
            "H06",
        ],
        source=UCASource.CONTROLLER,
        safety_constraints=[
            "Safety stops shall be based on validated evidence.",
            "False or stale safety evidence shall be detectable.",
        ],
        severity="major",
    ),
    UnsafeControlAction(
        uca_id="UCA14",
        name="Malicious Control Command Accepted",
        action_type=ControlActionType.COMMUNICATE_COMMAND,
        uca_type=UCAType.PROVIDED_WHEN_NOT_APPROPRIATE,
        controller="Communication Interface",
        description=(
            "An unauthorized or manipulated control command reaches the "
            "insulin-delivery path and is accepted."
        ),
        unsafe_context=(
            "Command authenticity, integrity, authorization, or sequence "
            "validation fails."
        ),
        related_hazards=[
            "H03",
            "H04",
            "H08",
        ],
        source=UCASource.CYBER_ATTACK,
        safety_constraints=[
            "Safety-critical commands shall be authenticated.",
            "Command integrity and authorization shall be verified.",
            "Rejected commands shall not reach actuation.",
        ],
        severity="critical",
    ),
]


class UCARegistry:
    """Registry for STPA unsafe control actions."""

    def __init__(
        self,
        ucas: Optional[Iterable[UnsafeControlAction]] = None,
    ) -> None:
        self._ucas: Dict[str, UnsafeControlAction] = {}

        for uca in ucas or DEFAULT_UCAS:
            self.register(uca)

    def register(self, uca: UnsafeControlAction) -> None:
        """Register a UCA."""
        if uca.uca_id in self._ucas:
            raise ValueError(
                f"UCA '{uca.uca_id}' is already registered."
            )

        self._ucas[uca.uca_id] = uca

    def get(self, uca_id: str) -> UnsafeControlAction:
        """Retrieve one UCA by ID."""
        try:
            return self._ucas[uca_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown UCA '{uca_id}'. "
                f"Available IDs: {list(self._ucas)}"
            ) from exc

    def get_all(
        self,
        *,
        enabled_only: bool = False,
    ) -> List[UnsafeControlAction]:
        """Return all registered UCAs."""
        values = list(self._ucas.values())

        if enabled_only:
            values = [
                uca for uca in values
                if uca.enabled
            ]

        return values

    def get_by_type(
        self,
        uca_type: UCAType,
        *,
        enabled_only: bool = False,
    ) -> List[UnsafeControlAction]:
        """Return UCAs belonging to one STPA category."""
        return [
            uca
            for uca in self.get_all(enabled_only=enabled_only)
            if uca.uca_type == uca_type
        ]

    def get_by_action(
        self,
        action_type: ControlActionType,
        *,
        enabled_only: bool = False,
    ) -> List[UnsafeControlAction]:
        """Return UCAs associated with one control action."""
        return [
            uca
            for uca in self.get_all(enabled_only=enabled_only)
            if uca.action_type == action_type
        ]

    def get_by_hazard(
        self,
        hazard_id: str,
        *,
        enabled_only: bool = False,
    ) -> List[UnsafeControlAction]:
        """Return UCAs associated with a hazard."""
        return [
            uca
            for uca in self.get_all(enabled_only=enabled_only)
            if hazard_id in uca.related_hazards
        ]

    def get_by_source(
        self,
        source: UCASource,
        *,
        enabled_only: bool = False,
    ) -> List[UnsafeControlAction]:
        """Return UCAs associated with one source."""
        return [
            uca
            for uca in self.get_all(enabled_only=enabled_only)
            if uca.source == source
        ]

    def __len__(self) -> int:
        return len(self._ucas)

    def __contains__(self, uca_id: str) -> bool:
        return uca_id in self._ucas


def get_default_ucas() -> List[UnsafeControlAction]:
    """Return independent copies of the default UCAs."""
    return [
        UnsafeControlAction.from_dict(uca.to_dict())
        for uca in DEFAULT_UCAS
    ]


def get_uca_registry() -> UCARegistry:
    """Create a registry containing the default UCAs."""
    return UCARegistry(get_default_ucas())


def ucas_to_dict(
    ucas: Iterable[UnsafeControlAction],
) -> List[Dict[str, Any]]:
    """Convert UCAs into serializable dictionaries."""
    return [uca.to_dict() for uca in ucas]


__all__ = [
    "UCAType",
    "ControlActionType",
    "UCASource",
    "UnsafeControlAction",
    "UCARegistry",
    "DEFAULT_UCAS",
    "get_default_ucas",
    "get_uca_registry",
    "ucas_to_dict",
]
