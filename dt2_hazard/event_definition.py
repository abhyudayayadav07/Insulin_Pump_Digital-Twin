"""
Event definitions for the hazard/survival Digital Twin 2.

DT2 is a hazard- and survival-function-based digital twin.  This module
defines the events whose occurrence is modeled by cause-specific hazard
functions and survival functions.

The module separates:

    Event definition
        ↓
    event-specific state/threshold logic
        ↓
    hazard function h_k(t | x)
        ↓
    survival function S_k(t)
        ↓
    time-to-event analysis

These definitions are research/engineering abstractions.  Thresholds and
clinical interpretations must be validated against the intended study
protocol and domain expertise before any clinical use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math


class EventCategory(str, Enum):
    """Broad classes of DT2 events."""

    PHYSIOLOGICAL = "physiological"
    INSULIN = "insulin"
    SENSOR = "sensor"
    CONTROLLER = "controller"
    PUMP = "pump"
    COMMUNICATION = "communication"
    CYBER = "cyber"
    SAFETY = "safety"
    UNKNOWN = "unknown"


class EventSeverity(str, Enum):
    """Qualitative event severity."""

    NEGLIGIBLE = "negligible"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class EventDirection(str, Enum):
    """Physiological direction associated with an event."""

    LOW = "low"
    HIGH = "high"
    OVER = "over"
    UNDER = "under"
    FAILURE = "failure"
    INTEGRITY_LOSS = "integrity_loss"
    NONE = "none"


@dataclass(frozen=True)
class EventDefinition:
    """
    Canonical definition of a DT2 event.

    `event_id` is the stable identifier used by hazard, survival,
    prediction, and decision-engine layers.
    """

    event_id: str
    name: str
    category: EventCategory
    description: str

    severity: EventSeverity = EventSeverity.UNKNOWN
    direction: EventDirection = EventDirection.NONE

    enabled: bool = True

    # Optional physiological thresholds.  These are configuration
    # values, not universal clinical definitions.
    glucose_lower: Optional[float] = None
    glucose_upper: Optional[float] = None

    # Optional integrity threshold.  An integrity value below this
    # threshold can be treated as evidence for the event.
    integrity_threshold: Optional[float] = None

    # Optional minimum attack evidence.
    attack_threshold: Optional[float] = None

    # Metadata can carry study-specific information without changing
    # the event interface.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty.")

        if not self.name.strip():
            raise ValueError("name must not be empty.")

        if self.glucose_lower is not None and self.glucose_upper is not None:
            if self.glucose_lower >= self.glucose_upper:
                raise ValueError(
                    "glucose_lower must be smaller than glucose_upper."
                )

        if self.integrity_threshold is not None:
            if not 0 <= self.integrity_threshold <= 1:
                raise ValueError(
                    "integrity_threshold must be between 0 and 1."
                )

        if self.attack_threshold is not None:
            if not 0 <= self.attack_threshold <= 1:
                raise ValueError(
                    "attack_threshold must be between 0 and 1."
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "severity": self.severity.value,
            "direction": self.direction.value,
            "enabled": self.enabled,
            "glucose_lower": self.glucose_lower,
            "glucose_upper": self.glucose_upper,
            "integrity_threshold": self.integrity_threshold,
            "attack_threshold": self.attack_threshold,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EventObservation:
    """
    Observation of evidence relevant to one event.

    This class deliberately stores evidence rather than a final
    probability.  The hazard model remains responsible for mapping
    state/evidence into h(t | x).
    """

    event_id: str
    timestamp: Optional[Any] = None

    triggered: bool = False
    evidence_score: float = 0.0
    confidence: float = 1.0

    values: Dict[str, float] = field(default_factory=dict)
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty.")

        if not 0 <= self.evidence_score <= 1:
            raise ValueError(
                "evidence_score must be between 0 and 1."
            )

        if not 0 <= self.confidence <= 1:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        for name, value in self.values.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Observation value '{name}' must be numeric."
                ) from exc

            if not math.isfinite(numeric):
                raise ValueError(
                    f"Observation value '{name}' must be finite."
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "triggered": self.triggered,
            "evidence_score": self.evidence_score,
            "confidence": self.confidence,
            "values": dict(self.values),
            "source": self.source,
            "metadata": dict(self.metadata),
        }


class EventRegistry:
    """Registry of DT2 event definitions."""

    def __init__(
        self,
        events: Optional[Iterable[EventDefinition]] = None,
    ) -> None:
        self._events: Dict[str, EventDefinition] = {}

        if events is not None:
            for event in events:
                self.register(event)

    def register(
        self,
        event: EventDefinition,
        *,
        overwrite: bool = False,
    ) -> None:
        if (
            event.event_id in self._events
            and not overwrite
        ):
            raise ValueError(
                f"Event '{event.event_id}' is already registered."
            )

        self._events[event.event_id] = event

    def get(
        self,
        event_id: str,
    ) -> EventDefinition:
        try:
            return self._events[event_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown DT2 event: {event_id}"
            ) from exc

    def get_optional(
        self,
        event_id: str,
    ) -> Optional[EventDefinition]:
        return self._events.get(event_id)

    def remove(
        self,
        event_id: str,
    ) -> None:
        if event_id not in self._events:
            raise KeyError(
                f"Unknown DT2 event: {event_id}"
            )
        del self._events[event_id]

    def all(
        self,
        *,
        enabled_only: bool = False,
    ) -> Tuple[EventDefinition, ...]:
        events = tuple(self._events.values())

        if enabled_only:
            events = tuple(
                event
                for event in events
                if event.enabled
            )

        return events

    def by_category(
        self,
        category: EventCategory,
        *,
        enabled_only: bool = True,
    ) -> Tuple[EventDefinition, ...]:
        return tuple(
            event
            for event in self.all(
                enabled_only=enabled_only
            )
            if event.category == category
        )

    def __contains__(
        self,
        event_id: str,
    ) -> bool:
        return event_id in self._events

    def __len__(self) -> int:
        return len(self._events)

    def ids(
        self,
        *,
        enabled_only: bool = False,
    ) -> Tuple[str, ...]:
        return tuple(
            event.event_id
            for event in self.all(
                enabled_only=enabled_only
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            event.event_id: event.to_dict()
            for event in self.all()
        }


# ---------------------------------------------------------------------------
# Default DT2 event definitions
# ---------------------------------------------------------------------------

def default_event_definitions() -> Tuple[EventDefinition, ...]:
    """
    Return the default event set for DT2.

    These are deliberately event-oriented rather than identical to the
    existing hazard-definition module.  Hazard definitions describe
    hazards/risk semantics; these definitions identify the event whose
    time-to-event distribution is modeled.
    """

    return (
        EventDefinition(
            event_id="E01",
            name="Severe Hypoglycemia",
            category=EventCategory.PHYSIOLOGICAL,
            description=(
                "Future low-glucose event of predefined severity."
            ),
            severity=EventSeverity.CRITICAL,
            direction=EventDirection.LOW,
            glucose_lower=70.0,
            metadata={
                "unit": "mg/dL",
                "threshold_type": "configurable",
            },
        ),
        EventDefinition(
            event_id="E02",
            name="Severe Hyperglycemia",
            category=EventCategory.PHYSIOLOGICAL,
            description=(
                "Future high-glucose event of predefined severity."
            ),
            severity=EventSeverity.CRITICAL,
            direction=EventDirection.HIGH,
            glucose_upper=180.0,
            metadata={
                "unit": "mg/dL",
                "threshold_type": "configurable",
            },
        ),
        EventDefinition(
            event_id="E03",
            name="Insulin Overdelivery",
            category=EventCategory.INSULIN,
            description=(
                "Unsafe insulin delivery above the configured safe range."
            ),
            severity=EventSeverity.CRITICAL,
            direction=EventDirection.OVER,
            metadata={
                "threshold_type": "configured_by_controller_or_study",
            },
        ),
        EventDefinition(
            event_id="E04",
            name="Insulin Underdelivery",
            category=EventCategory.INSULIN,
            description=(
                "Insulin delivery below a required or commanded level."
            ),
            severity=EventSeverity.MAJOR,
            direction=EventDirection.UNDER,
            metadata={
                "threshold_type": "configured_by_controller_or_study",
            },
        ),
        EventDefinition(
            event_id="E05",
            name="Sensor Integrity Failure",
            category=EventCategory.SENSOR,
            description=(
                "Loss of confidence in the glucose sensor or its data."
            ),
            severity=EventSeverity.MAJOR,
            direction=EventDirection.INTEGRITY_LOSS,
            integrity_threshold=0.5,
        ),
        EventDefinition(
            event_id="E06",
            name="Controller Integrity Failure",
            category=EventCategory.CONTROLLER,
            description=(
                "Loss of confidence in controller computation or command "
                "generation."
            ),
            severity=EventSeverity.MAJOR,
            direction=EventDirection.FAILURE,
            integrity_threshold=0.5,
        ),
        EventDefinition(
            event_id="E07",
            name="Pump Actuation Failure",
            category=EventCategory.PUMP,
            description=(
                "Pump/actuator behavior deviates from the expected "
                "delivery behavior."
            ),
            severity=EventSeverity.MAJOR,
            direction=EventDirection.FAILURE,
            integrity_threshold=0.5,
        ),
        EventDefinition(
            event_id="E08",
            name="Communication Failure",
            category=EventCategory.COMMUNICATION,
            description=(
                "Loss or degradation of a communication channel required "
                "for safe operation."
            ),
            severity=EventSeverity.MAJOR,
            direction=EventDirection.FAILURE,
            integrity_threshold=0.5,
        ),
        EventDefinition(
            event_id="E09",
            name="Cyber-Induced Unsafe Control",
            category=EventCategory.CYBER,
            description=(
                "Cyber evidence indicates a future unsafe control or "
                "actuation event."
            ),
            severity=EventSeverity.CRITICAL,
            direction=EventDirection.FAILURE,
            attack_threshold=0.5,
        ),
        EventDefinition(
            event_id="E10",
            name="DT1 Predictive Divergence",
            category=EventCategory.SAFETY,
            description=(
                "Predictive DT1 evidence diverges from the physiological "
                "state strongly enough to increase future-event risk."
            ),
            severity=EventSeverity.MAJOR,
            direction=EventDirection.FAILURE,
            metadata={
                "evidence_sources": [
                    "dt1_disagreement",
                    "geco_residual",
                    "cusum",
                ],
            },
        ),
    )


def create_default_event_registry() -> EventRegistry:
    """Create a registry containing the default DT2 event definitions."""
    return EventRegistry(
        default_event_definitions()
    )


# ---------------------------------------------------------------------------
# Event evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_event(
    event: EventDefinition,
    state: Mapping[str, Any],
) -> EventObservation:
    """
    Evaluate simple event evidence from a DT2 state.

    This function provides deterministic baseline evidence only.  It is
    NOT the hazard model.  Complex temporal/event logic should be handled
    by hazard_function.py or a dedicated event evaluator.
    """

    values: Dict[str, float] = {}
    evidence_components: List[float] = []

    # Glucose evidence.
    glucose = state.get("glucose")

    if glucose is not None:
        try:
            glucose = float(glucose)
        except (TypeError, ValueError):
            glucose = None

    if glucose is not None and math.isfinite(glucose):
        values["glucose"] = glucose

        if (
            event.glucose_lower is not None
            and glucose < event.glucose_lower
        ):
            # Evidence rises as glucose moves below the threshold.
            evidence = min(
                1.0,
                (
                    event.glucose_lower - glucose
                ) / max(
                    1.0,
                    event.glucose_lower,
                ),
            )
            evidence_components.append(
                max(0.0, evidence)
            )

        if (
            event.glucose_upper is not None
            and glucose > event.glucose_upper
        ):
            evidence = min(
                1.0,
                (
                    glucose - event.glucose_upper
                ) / max(
                    1.0,
                    event.glucose_upper,
                ),
            )
            evidence_components.append(
                max(0.0, evidence)
            )

    # Integrity evidence.
    integrity_field_by_category = {
        EventCategory.SENSOR: "sensor_integrity",
        EventCategory.CONTROLLER: "controller_integrity",
        EventCategory.PUMP: "pump_integrity",
        EventCategory.COMMUNICATION: "communication_integrity",
    }

    integrity_field = integrity_field_by_category.get(
        event.category
    )

    if (
        integrity_field is not None
        and event.integrity_threshold is not None
    ):
        raw_integrity = state.get(
            integrity_field
        )

        if raw_integrity is not None:
            try:
                integrity = float(raw_integrity)
            except (TypeError, ValueError):
                integrity = math.nan

            if math.isfinite(integrity):
                values[integrity_field] = integrity

                if integrity < event.integrity_threshold:
                    evidence = (
                        event.integrity_threshold - integrity
                    ) / max(
                        event.integrity_threshold,
                        1e-6,
                    )
                    evidence_components.append(
                        max(0.0, min(1.0, evidence))
                    )

    # Cyber evidence.
    if (
        event.category == EventCategory.CYBER
        and event.attack_threshold is not None
    ):
        raw_attack = state.get(
            "attack_indicator"
        )

        if raw_attack is not None:
            try:
                attack = float(raw_attack)
            except (TypeError, ValueError):
                attack = math.nan

            if math.isfinite(attack):
                values["attack_indicator"] = attack

                if attack >= event.attack_threshold:
                    evidence_components.append(
                        min(
                            1.0,
                            attack,
                        )
                    )

    # Predictive divergence evidence.
    if event.event_id == "E10":
        disagreement = state.get(
            "dt1_disagreement"
        )
        residual = state.get(
            "geco_residual"
        )
        cusum = state.get(
            "cusum"
        )

        if disagreement is not None:
            try:
                disagreement = abs(float(disagreement))
                if math.isfinite(disagreement):
                    values["dt1_disagreement"] = disagreement
                    evidence_components.append(
                        min(1.0, disagreement / 70.0)
                    )
            except (TypeError, ValueError):
                pass

        if residual is not None:
            try:
                residual = abs(float(residual))
                if math.isfinite(residual):
                    values["geco_residual"] = residual
                    evidence_components.append(
                        min(1.0, residual / 60.0)
                    )
            except (TypeError, ValueError):
                pass

        if cusum is not None:
            try:
                cusum = abs(float(cusum))
                if math.isfinite(cusum):
                    values["cusum"] = cusum
                    evidence_components.append(
                        min(1.0, cusum / 8.0)
                    )
            except (TypeError, ValueError):
                pass

    evidence_score = (
        max(evidence_components)
        if evidence_components
        else 0.0
    )

    return EventObservation(
        event_id=event.event_id,
        triggered=evidence_score > 0.0,
        evidence_score=evidence_score,
        confidence=1.0,
        values=values,
        source="baseline_event_evaluator",
    )


def evaluate_all_events(
    state: Mapping[str, Any],
    registry: Optional[EventRegistry] = None,
) -> Tuple[EventObservation, ...]:
    """Evaluate baseline evidence for all enabled DT2 events."""
    registry = registry or create_default_event_registry()

    return tuple(
        evaluate_event(
            event,
            state,
        )
        for event in registry.all(
            enabled_only=True
        )
    )


def event_ids(
    registry: Optional[EventRegistry] = None,
) -> Tuple[str, ...]:
    """Return stable IDs for enabled events."""
    registry = registry or create_default_event_registry()
    return registry.ids(enabled_only=True)


__all__ = [
    "EventCategory",
    "EventSeverity",
    "EventDirection",
    "EventDefinition",
    "EventObservation",
    "EventRegistry",
    "default_event_definitions",
    "create_default_event_registry",
    "evaluate_event",
    "evaluate_all_events",
    "event_ids",
]
