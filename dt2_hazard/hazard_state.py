"""
DT2 Hazard State Module
=======================

Maintains the current safety state of the insulin-pump digital twin.

This module converts observed physiological/device signals and DT1 outputs
into a structured, continuously updateable hazard state.

It does NOT estimate statistical hazard probability or calculate final risk.
Those responsibilities belong to hazard_probability.py, hazard_predictor.py,
and risk/ modules respectively.

Expected flow:

    Simulation / DT1 outputs
              |
              v
       HazardStateManager
              |
              v
       Current HazardState
              |
        +-----+-----+
        |           |
        v           v
 hazard_function  hazard_probability
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    from .hazard_definition import (
        HazardDefinition,
        HazardRegistry,
        HazardSeverity,
        HazardType,
        get_hazard_registry,
    )
except ImportError:
    from hazard_definition import (
        HazardDefinition,
        HazardRegistry,
        HazardSeverity,
        HazardType,
        get_hazard_registry,
    )


class HazardStateLabel(str, Enum):
    """Current qualitative state of a hazard."""

    INACTIVE = "inactive"
    MONITORING = "monitoring"
    WARNING = "warning"
    ACTIVE = "active"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HazardObservation:
    """
    Observation/evidence used to update one hazard state.

    Parameters
    ----------
    hazard_id:
        ID from ``hazard_definition.py``.
    timestamp:
        Observation timestamp. ISO-8601 strings are also accepted.
    evidence:
        Named evidence values, e.g. ``glucose``, ``predicted_glucose``,
        ``residual``, ``commanded_insulin``.
    evidence_score:
        Optional normalized evidence strength in [0, 1].
    triggered:
        Whether the observation explicitly indicates a hazard condition.
    source:
        Source of the evidence, such as ``simulator``, ``dt1``, or
        ``attack_detector``.
    metadata:
        Additional observation information.
    """

    hazard_id: str
    timestamp: Any
    evidence: Dict[str, Any] = field(default_factory=dict)
    evidence_score: Optional[float] = None
    triggered: Optional[bool] = None
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hazard_id.strip():
            raise ValueError("hazard_id must not be empty.")

        if self.evidence_score is not None:
            self.evidence_score = float(self.evidence_score)
            if not 0.0 <= self.evidence_score <= 1.0:
                raise ValueError("evidence_score must be between 0 and 1.")

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        return asdict(self)


@dataclass
class HazardState:
    """
    Runtime state of one hazard.

    ``HazardDefinition`` describes what a hazard IS; ``HazardState`` describes
    what is happening with that hazard NOW.
    """

    hazard_id: str
    label: HazardStateLabel = HazardStateLabel.UNKNOWN
    active: bool = False
    severity: HazardSeverity = HazardSeverity.MODERATE

    evidence_score: float = 0.0
    confidence: float = 0.0

    first_detected_at: Optional[Any] = None
    last_updated_at: Optional[Any] = None
    duration_seconds: float = 0.0

    observation_count: int = 0
    consecutive_trigger_count: int = 0

    latest_evidence: Dict[str, Any] = field(default_factory=dict)
    evidence_history: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        observation: HazardObservation,
        *,
        warning_threshold: float = 0.30,
        active_threshold: float = 0.60,
        critical_threshold: float = 0.85,
        history_limit: int = 100,
    ) -> "HazardState":
        """
        Update this state from a new observation.

        The thresholds classify evidence strength only. They are NOT a
        probability model and should not be interpreted as risk probability.
        """

        if not 0.0 <= warning_threshold <= 1.0:
            raise ValueError("warning_threshold must be in [0, 1].")
        if not 0.0 <= active_threshold <= 1.0:
            raise ValueError("active_threshold must be in [0, 1].")
        if not 0.0 <= critical_threshold <= 1.0:
            raise ValueError("critical_threshold must be in [0, 1].")

        if not (
            warning_threshold <= active_threshold <= critical_threshold
        ):
            raise ValueError(
                "Thresholds must satisfy warning <= active <= critical."
            )

        score = (
            float(observation.evidence_score)
            if observation.evidence_score is not None
            else 1.0 if observation.triggered else 0.0
        )
        score = max(0.0, min(1.0, score))

        previous_active = self.active

        self.hazard_id = observation.hazard_id
        self.evidence_score = score
        self.confidence = score
        self.latest_evidence = dict(observation.evidence)
        self.source = observation.source
        self.observation_count += 1
        self.last_updated_at = observation.timestamp

        if observation.triggered is True:
            self.consecutive_trigger_count += 1
        elif observation.triggered is False:
            self.consecutive_trigger_count = 0

        if score >= critical_threshold:
            self.label = HazardStateLabel.CRITICAL
            self.active = True
        elif score >= active_threshold:
            self.label = HazardStateLabel.ACTIVE
            self.active = True
        elif score >= warning_threshold:
            self.label = HazardStateLabel.WARNING
            self.active = False
        else:
            self.label = (
                HazardStateLabel.MONITORING
                if score > 0.0
                else HazardStateLabel.INACTIVE
            )
            self.active = False

        if self.active and not previous_active:
            self.first_detected_at = observation.timestamp

        self._update_duration(observation.timestamp)

        self.evidence_history.append(observation.to_dict())
        if history_limit >= 0 and len(self.evidence_history) > history_limit:
            self.evidence_history = self.evidence_history[-history_limit:]

        return self

    def _update_duration(self, timestamp: Any) -> None:
        """Update active duration when timestamps can be interpreted."""
        if not self.active or self.first_detected_at is None:
            if not self.active:
                self.duration_seconds = 0.0
            return

        start = _timestamp_to_datetime(self.first_detected_at)
        end = _timestamp_to_datetime(timestamp)

        if start is not None and end is not None:
            self.duration_seconds = max(0.0, (end - start).total_seconds())

    def reset(self) -> None:
        """Reset runtime evidence while preserving the hazard identity."""
        self.label = HazardStateLabel.INACTIVE
        self.active = False
        self.evidence_score = 0.0
        self.confidence = 0.0
        self.first_detected_at = None
        self.last_updated_at = None
        self.duration_seconds = 0.0
        self.observation_count = 0
        self.consecutive_trigger_count = 0
        self.latest_evidence = {}
        self.evidence_history = []
        self.source = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation."""
        data = asdict(self)
        data["label"] = self.label.value
        data["severity"] = self.severity.value
        return data


class HazardStateManager:
    """
    Maintains states for all registered hazards.

    This manager is the main interface DT2 can use during a simulation loop.
    """

    def __init__(
        self,
        registry: Optional[HazardRegistry] = None,
        *,
        warning_threshold: float = 0.30,
        active_threshold: float = 0.60,
        critical_threshold: float = 0.85,
        history_limit: int = 100,
    ) -> None:
        self.registry = registry or get_hazard_registry()

        if not (
            0.0 <= warning_threshold
            <= active_threshold
            <= critical_threshold
            <= 1.0
        ):
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= warning <= active <= critical <= 1."
            )

        self.warning_threshold = warning_threshold
        self.active_threshold = active_threshold
        self.critical_threshold = critical_threshold
        self.history_limit = history_limit

        self._states: Dict[str, HazardState] = {}

        for hazard in self.registry.get_all(enabled_only=True):
            self._states[hazard.hazard_id] = HazardState(
                hazard_id=hazard.hazard_id,
                severity=hazard.severity,
                label=HazardStateLabel.INACTIVE,
                active=False,
            )

    def get_state(self, hazard_id: str) -> HazardState:
        """Return the current state for one hazard."""
        if hazard_id not in self._states:
            raise KeyError(f"No runtime state exists for hazard '{hazard_id}'.")
        return self._states[hazard_id]

    def update(
        self,
        observation: HazardObservation,
    ) -> HazardState:
        """Update one hazard from a new observation."""
        if observation.hazard_id not in self._states:
            hazard = self.registry.get(observation.hazard_id)

            if not hazard.enabled:
                raise ValueError(
                    f"Hazard '{observation.hazard_id}' is disabled."
                )

            self._states[observation.hazard_id] = HazardState(
                hazard_id=hazard.hazard_id,
                severity=hazard.severity,
            )

        return self._states[observation.hazard_id].update(
            observation,
            warning_threshold=self.warning_threshold,
            active_threshold=self.active_threshold,
            critical_threshold=self.critical_threshold,
            history_limit=self.history_limit,
        )

    def update_from_values(
        self,
        hazard_id: str,
        *,
        timestamp: Any = None,
        evidence: Optional[Mapping[str, Any]] = None,
        evidence_score: Optional[float] = None,
        triggered: Optional[bool] = None,
        source: str = "unknown",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> HazardState:
        """Convenience method for creating and applying an observation."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        observation = HazardObservation(
            hazard_id=hazard_id,
            timestamp=timestamp,
            evidence=dict(evidence or {}),
            evidence_score=evidence_score,
            triggered=triggered,
            source=source,
            metadata=dict(metadata or {}),
        )
        return self.update(observation)

    def get_all_states(self) -> List[HazardState]:
        """Return states for all hazards."""
        return list(self._states.values())

    def get_active_states(self) -> List[HazardState]:
        """Return currently active/critical hazards."""
        return [
            state
            for state in self._states.values()
            if state.active
        ]

    def get_states_by_label(
        self,
        label: HazardStateLabel,
    ) -> List[HazardState]:
        """Return states with a particular label."""
        return [
            state
            for state in self._states.values()
            if state.label == label
        ]

    def any_active(self) -> bool:
        """Return True if at least one hazard is active."""
        return any(state.active for state in self._states.values())

    def highest_evidence_state(self) -> Optional[HazardState]:
        """Return the state with the strongest current evidence."""
        states = self.get_all_states()
        if not states:
            return None
        return max(states, key=lambda state: state.evidence_score)

    def reset(self, hazard_id: Optional[str] = None) -> None:
        """
        Reset one hazard or all hazard states.
        """
        if hazard_id is not None:
            self.get_state(hazard_id).reset()
            return

        for state in self._states.values():
            state.reset()

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return a dictionary snapshot of the entire DT2 hazard state."""
        return {
            hazard_id: state.to_dict()
            for hazard_id, state in self._states.items()
        }


def _timestamp_to_datetime(value: Any) -> Optional[datetime]:
    """Convert common timestamp representations to timezone-aware datetime."""
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            pass

    return None


def create_hazard_state_manager(
    registry: Optional[HazardRegistry] = None,
    **kwargs: Any,
) -> HazardStateManager:
    """Factory for creating a configured hazard state manager."""
    return HazardStateManager(registry=registry, **kwargs)


__all__ = [
    "HazardStateLabel",
    "HazardObservation",
    "HazardState",
    "HazardStateManager",
    "create_hazard_state_manager",
]
