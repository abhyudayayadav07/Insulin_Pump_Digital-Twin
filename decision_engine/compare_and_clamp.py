"""
Trusted compare-and-clamp safety layer for the insulin digital twin.

Purpose
-------
This module implements the final authorization layer between:

    DT1 predictive recommendation
    DT2 reactive-safe recommendation
    security / hazard / attack evidence

and the final insulin command.

Conceptual flow:

    DT1 recommendation
             |
             |\
             | \
             |  +----------------------+
             |                         |
             v                         v
       DT1 command              DT2 safe command
             \                         /
              \                       /
               +--------+------------+
                        |
                        v
                Compare & Clamp
                        |
          +-------------+-------------+
          |             |             |
        ALLOW         CLAMP        SAFE MODE
          |             |             |
          +-------------+-------------+
                        |
                        v
                 final command

The design is inspired by the trusted safety-bound / compare-and-clamp
principle used in GlucOS-style architectures, but this implementation
also accepts security evidence such as hazard state, risk, attack
evidence, sensor/controller/pump integrity, and safety constraints.

Important
---------
This is a research/simulation component. It is NOT a clinically validated
insulin dosing controller and must not be connected directly to a real
insulin pump.

The module does not itself calculate a therapeutic insulin dose. It only
authorizes, bounds, or rejects a proposed DT1 command relative to a DT2
safe recommendation and the supplied safety constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional

import math


class ClampAction(str, Enum):
    """Final action selected by the trusted safety layer."""

    ALLOW_DT1 = "allow_dt1"
    CLAMP_TO_SAFE_RANGE = "clamp_to_safe_range"
    USE_DT2 = "use_dt2"
    SAFE_MODE = "safe_mode"
    EMERGENCY_STOP = "emergency_stop"
    REJECT_COMMAND = "reject_command"


class SafetyLevel(str, Enum):
    """Normalized safety state used by compare-and-clamp."""

    NORMAL = "normal"
    MONITOR = "monitor"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CompareClampConfig:
    """
    Configuration for the trusted comparison layer.

    Parameters
    ----------
    max_deviation_from_dt2:
        Maximum absolute deviation allowed between the predictive DT1
        command and the DT2 command before clamping.

    max_relative_deviation:
        Optional relative bound. For example, 0.25 means DT1 can differ
        from DT2 by at most 25%.

    min_command:
        Lower command bound used by the simulation safety layer.

    max_command:
        Upper command bound used by the simulation safety layer.

    disagreement_warning:
        DT1/DT2 command difference at which monitoring begins.

    disagreement_high:
        Difference at which predictive authority is restricted.

    disagreement_critical:
        Difference at which the trusted layer falls back to DT2 or safe
        mode.

    hazard_warning / hazard_high / hazard_critical:
        Optional hazard-score thresholds.

    risk_high / risk_critical:
        Optional normalized risk thresholds.

    require_dt2:
        If True, DT1 cannot receive final authority without a DT2
        reference command.

    block_on_attack:
        If True, strong attack evidence prevents DT1 authority.

    block_on_constraint_violation:
        If True, a critical safety-constraint violation prevents DT1
        authority.

    emergency_on_critical:
        If True, critical security evidence escalates to emergency stop.
        For research simulations this should normally be False unless the
        scenario explicitly models an emergency-stop condition.
    """

    max_deviation_from_dt2: float = 0.30
    max_relative_deviation: Optional[float] = 0.25

    min_command: float = 0.0
    max_command: float = 5.0

    disagreement_warning: float = 0.20
    disagreement_high: float = 0.50
    disagreement_critical: float = 1.00

    hazard_warning: float = 0.30
    hazard_high: float = 0.60
    hazard_critical: float = 0.85

    risk_high: float = 0.60
    risk_critical: float = 0.85

    require_dt2: bool = True
    block_on_attack: bool = True
    block_on_constraint_violation: bool = True
    emergency_on_critical: bool = False

    def __post_init__(self) -> None:
        if self.max_deviation_from_dt2 < 0:
            raise ValueError(
                "max_deviation_from_dt2 must be >= 0."
            )

        if (
            self.max_relative_deviation is not None
            and self.max_relative_deviation < 0
        ):
            raise ValueError(
                "max_relative_deviation must be >= 0."
            )

        if self.min_command < 0:
            raise ValueError(
                "min_command must be >= 0."
            )

        if self.max_command < self.min_command:
            raise ValueError(
                "max_command must be >= min_command."
            )

        for name, value in (
            ("disagreement_warning", self.disagreement_warning),
            ("disagreement_high", self.disagreement_high),
            ("disagreement_critical", self.disagreement_critical),
        ):
            if value < 0:
                raise ValueError(
                    f"{name} must be >= 0."
                )

        if not (
            self.disagreement_warning
            <= self.disagreement_high
            <= self.disagreement_critical
        ):
            raise ValueError(
                "Disagreement thresholds must be ordered warning <= high <= critical."
            )

        for name, value in (
            ("hazard_warning", self.hazard_warning),
            ("hazard_high", self.hazard_high),
            ("hazard_critical", self.hazard_critical),
            ("risk_high", self.risk_high),
            ("risk_critical", self.risk_critical),
        ):
            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

        if not (
            self.hazard_warning
            <= self.hazard_high
            <= self.hazard_critical
        ):
            raise ValueError(
                "Hazard thresholds must be ordered warning <= high <= critical."
            )

        if self.risk_high > self.risk_critical:
            raise ValueError(
                "risk_high must be <= risk_critical."
            )


@dataclass(frozen=True)
class SafetyEvidence:
    """
    Security/safety evidence supplied to compare-and-clamp.

    All probabilities/scores are treated as evidence scores unless the
    caller explicitly states that they are calibrated probabilities.
    """

    hazard_score: Optional[float] = None
    hazard_probability: Optional[float] = None
    risk_score: Optional[float] = None

    attack_detected: bool = False
    attack_strength: float = 0.0

    constraint_violation: bool = False
    critical_constraint_violation: bool = False

    sensor_integrity_ok: bool = True
    controller_integrity_ok: bool = True
    pump_integrity_ok: bool = True
    communication_ok: bool = True

    dt1_anomaly: bool = False
    dt1_critical: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("hazard_score", self.hazard_score),
            ("hazard_probability", self.hazard_probability),
            ("risk_score", self.risk_score),
            ("attack_strength", self.attack_strength),
        ):
            if value is not None:
                if not math.isfinite(float(value)):
                    raise ValueError(
                        f"{name} must be finite."
                    )

        for name, value in (
            ("hazard_score", self.hazard_score),
            ("hazard_probability", self.hazard_probability),
            ("risk_score", self.risk_score),
            ("attack_strength", self.attack_strength),
        ):
            if value is not None and not 0 <= float(value) <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )


@dataclass(frozen=True)
class CompareClampResult:
    """Complete result of one compare-and-clamp evaluation."""

    dt1_command: Optional[float]
    dt2_command: Optional[float]

    safe_lower_bound: Optional[float]
    safe_upper_bound: Optional[float]

    final_command: Optional[float]

    action: ClampAction
    safety_level: SafetyLevel

    disagreement: Optional[float]
    relative_disagreement: Optional[float]

    clamped: bool
    predictive_authority: bool

    reason: str
    evidence: SafetyEvidence

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def command_difference(self) -> Optional[float]:
        """Signed DT1 - DT2 command difference."""
        if (
            self.dt1_command is None
            or self.dt2_command is None
        ):
            return None

        return self.dt1_command - self.dt2_command

    @property
    def command_rejected(self) -> bool:
        return self.action in (
            ClampAction.SAFE_MODE,
            ClampAction.EMERGENCY_STOP,
            ClampAction.REJECT_COMMAND,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result."""
        return {
            "dt1_command": self.dt1_command,
            "dt2_command": self.dt2_command,
            "safe_lower_bound": self.safe_lower_bound,
            "safe_upper_bound": self.safe_upper_bound,
            "final_command": self.final_command,
            "action": self.action.value,
            "safety_level": self.safety_level.value,
            "disagreement": self.disagreement,
            "relative_disagreement": self.relative_disagreement,
            "clamped": self.clamped,
            "predictive_authority": self.predictive_authority,
            "reason": self.reason,
            "evidence": {
                "hazard_score": self.evidence.hazard_score,
                "hazard_probability": self.evidence.hazard_probability,
                "risk_score": self.evidence.risk_score,
                "attack_detected": self.evidence.attack_detected,
                "attack_strength": self.evidence.attack_strength,
                "constraint_violation": (
                    self.evidence.constraint_violation
                ),
                "critical_constraint_violation": (
                    self.evidence.critical_constraint_violation
                ),
                "sensor_integrity_ok": (
                    self.evidence.sensor_integrity_ok
                ),
                "controller_integrity_ok": (
                    self.evidence.controller_integrity_ok
                ),
                "pump_integrity_ok": (
                    self.evidence.pump_integrity_ok
                ),
                "communication_ok": (
                    self.evidence.communication_ok
                ),
                "dt1_anomaly": self.evidence.dt1_anomaly,
                "dt1_critical": self.evidence.dt1_critical,
                "metadata": dict(self.evidence.metadata),
            },
            "metadata": dict(self.metadata),
        }

    def to_record(self) -> Dict[str, Any]:
        """Return a flat record suitable for telemetry/CSV."""
        return {
            "dt1_command": self.dt1_command,
            "dt2_command": self.dt2_command,
            "safe_lower_bound": self.safe_lower_bound,
            "safe_upper_bound": self.safe_upper_bound,
            "final_command": self.final_command,
            "action": self.action.value,
            "safety_level": self.safety_level.value,
            "disagreement": self.disagreement,
            "relative_disagreement": self.relative_disagreement,
            "clamped": self.clamped,
            "predictive_authority": self.predictive_authority,
            "reason": self.reason,
        }


class CompareAndClamp:
    """
    Trusted safety comparison layer.

    The central operation is:

        DT1 command
              |
              v
        compare against DT2
              |
              v
        construct DT2-centered safety envelope
              |
              v
        apply security constraints
              |
              v
        ALLOW / CLAMP / USE DT2 / SAFE MODE / STOP

    DT1 is considered the predictive command source.
    DT2 is considered the reactive-safe reference.
    """

    def __init__(
        self,
        config: Optional[CompareClampConfig] = None,
    ) -> None:
        self.config = config or CompareClampConfig()

    @staticmethod
    def _finite_or_none(
        value: Optional[float],
    ) -> Optional[float]:
        if value is None:
            return None

        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    def _clamp_global(
        self,
        command: float,
    ) -> float:
        return max(
            self.config.min_command,
            min(
                self.config.max_command,
                command,
            ),
        )

    def _relative_bound(
        self,
        dt2_command: float,
    ) -> float:
        """
        Compute the DT2-relative deviation allowance.

        A small floor prevents a zero DT2 command from producing a
        meaningless zero-width relative interval.
        """
        if self.config.max_relative_deviation is None:
            return float("inf")

        reference = max(
            abs(dt2_command),
            self.config.min_command,
            1e-6,
        )

        return (
            reference
            * self.config.max_relative_deviation
        )

    def _build_safe_bounds(
        self,
        dt2_command: float,
    ) -> tuple[float, float]:
        absolute_bound = self.config.max_deviation_from_dt2
        relative_bound = self._relative_bound(dt2_command)

        deviation = min(
            absolute_bound,
            relative_bound,
        )

        lower = dt2_command - deviation
        upper = dt2_command + deviation

        lower = max(
            self.config.min_command,
            lower,
        )

        upper = min(
            self.config.max_command,
            upper,
        )

        if upper < lower:
            upper = lower

        return lower, upper

    def _disagreement_level(
        self,
        disagreement: float,
    ) -> SafetyLevel:
        if disagreement >= self.config.disagreement_critical:
            return SafetyLevel.CRITICAL

        if disagreement >= self.config.disagreement_high:
            return SafetyLevel.HIGH

        if disagreement >= self.config.disagreement_warning:
            return SafetyLevel.WARNING

        return SafetyLevel.NORMAL

    def _evidence_level(
        self,
        evidence: SafetyEvidence,
    ) -> SafetyLevel:
        levels = [SafetyLevel.NORMAL]

        if evidence.hazard_score is not None:
            score = float(evidence.hazard_score)

            if score >= self.config.hazard_critical:
                levels.append(SafetyLevel.CRITICAL)
            elif score >= self.config.hazard_high:
                levels.append(SafetyLevel.HIGH)
            elif score >= self.config.hazard_warning:
                levels.append(SafetyLevel.WARNING)

        if evidence.hazard_probability is not None:
            probability = float(
                evidence.hazard_probability
            )

            if probability >= self.config.hazard_critical:
                levels.append(SafetyLevel.CRITICAL)
            elif probability >= self.config.hazard_high:
                levels.append(SafetyLevel.HIGH)
            elif probability >= self.config.hazard_warning:
                levels.append(SafetyLevel.WARNING)

        if evidence.risk_score is not None:
            risk = float(evidence.risk_score)

            if risk >= self.config.risk_critical:
                levels.append(SafetyLevel.CRITICAL)
            elif risk >= self.config.risk_high:
                levels.append(SafetyLevel.HIGH)

        if evidence.attack_detected:
            if evidence.attack_strength >= 0.85:
                levels.append(SafetyLevel.CRITICAL)
            elif evidence.attack_strength >= 0.60:
                levels.append(SafetyLevel.HIGH)
            else:
                levels.append(SafetyLevel.WARNING)

        if evidence.critical_constraint_violation:
            levels.append(SafetyLevel.CRITICAL)
        elif evidence.constraint_violation:
            levels.append(SafetyLevel.HIGH)

        if evidence.dt1_critical:
            levels.append(SafetyLevel.CRITICAL)
        elif evidence.dt1_anomaly:
            levels.append(SafetyLevel.WARNING)

        if not (
            evidence.sensor_integrity_ok
            and evidence.controller_integrity_ok
            and evidence.pump_integrity_ok
            and evidence.communication_ok
        ):
            levels.append(SafetyLevel.HIGH)

        order = {
            SafetyLevel.UNKNOWN: 0,
            SafetyLevel.NORMAL: 1,
            SafetyLevel.MONITOR: 2,
            SafetyLevel.WARNING: 3,
            SafetyLevel.HIGH: 4,
            SafetyLevel.CRITICAL: 5,
        }

        return max(
            levels,
            key=lambda level: order[level],
        )

    def compare(
        self,
        dt1_command: Optional[float],
        dt2_command: Optional[float],
        evidence: Optional[SafetyEvidence] = None,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CompareClampResult:
        """
        Compare DT1 and DT2 commands and determine the final authorized
        command.

        Decision order:

        1. Validate command inputs.
        2. Establish DT2-centered safety envelope.
        3. Evaluate command disagreement.
        4. Evaluate security/safety evidence.
        5. Apply hard safety overrides.
        6. Otherwise allow DT1 if it is inside the envelope.
        7. Clamp DT1 to the envelope if it exceeds the bounds.
        8. Fall back to DT2 when predictive authority is not justified.
        """
        evidence = evidence or SafetyEvidence()
        metadata_dict = dict(metadata or {})

        dt1 = self._finite_or_none(dt1_command)
        dt2 = self._finite_or_none(dt2_command)

        if dt1 is not None:
            dt1 = self._clamp_global(dt1)

        if dt2 is not None:
            dt2 = self._clamp_global(dt2)

        if dt1 is None:
            return CompareClampResult(
                dt1_command=None,
                dt2_command=dt2,
                safe_lower_bound=None,
                safe_upper_bound=None,
                final_command=dt2,
                action=(
                    ClampAction.USE_DT2
                    if dt2 is not None
                    else ClampAction.REJECT_COMMAND
                ),
                safety_level=SafetyLevel.HIGH,
                disagreement=None,
                relative_disagreement=None,
                clamped=False,
                predictive_authority=False,
                reason=(
                    "DT1 command is unavailable or invalid; "
                    "falling back to DT2."
                ),
                evidence=evidence,
                metadata=metadata_dict,
            )

        if dt2 is None and self.config.require_dt2:
            return CompareClampResult(
                dt1_command=dt1,
                dt2_command=None,
                safe_lower_bound=None,
                safe_upper_bound=None,
                final_command=None,
                action=ClampAction.SAFE_MODE,
                safety_level=SafetyLevel.CRITICAL,
                disagreement=None,
                relative_disagreement=None,
                clamped=False,
                predictive_authority=False,
                reason=(
                    "DT2 safe reference is unavailable; "
                    "predictive authority is not permitted."
                ),
                evidence=evidence,
                metadata=metadata_dict,
            )

        if dt2 is None:
            return CompareClampResult(
                dt1_command=dt1,
                dt2_command=None,
                safe_lower_bound=self.config.min_command,
                safe_upper_bound=self.config.max_command,
                final_command=dt1,
                action=ClampAction.ALLOW_DT1,
                safety_level=SafetyLevel.WARNING,
                disagreement=None,
                relative_disagreement=None,
                clamped=False,
                predictive_authority=True,
                reason=(
                    "DT2 reference unavailable, but require_dt2 is "
                    "disabled; DT1 remains globally bounded."
                ),
                evidence=evidence,
                metadata=metadata_dict,
            )

        disagreement = abs(dt1 - dt2)

        reference = max(
            abs(dt2),
            self.config.min_command,
            1e-6,
        )

        relative_disagreement = (
            disagreement / reference
        )

        safe_lower, safe_upper = self._build_safe_bounds(dt2)

        disagreement_level = self._disagreement_level(
            disagreement
        )

        evidence_level = self._evidence_level(
            evidence
        )

        level_order = {
            SafetyLevel.UNKNOWN: 0,
            SafetyLevel.NORMAL: 1,
            SafetyLevel.MONITOR: 2,
            SafetyLevel.WARNING: 3,
            SafetyLevel.HIGH: 4,
            SafetyLevel.CRITICAL: 5,
        }

        safety_level = max(
            disagreement_level,
            evidence_level,
            key=lambda level: level_order[level],
        )

        integrity_failure = not (
            evidence.sensor_integrity_ok
            and evidence.controller_integrity_ok
            and evidence.pump_integrity_ok
            and evidence.communication_ok
        )

        critical_security = (
            evidence.critical_constraint_violation
            or (
                evidence.attack_detected
                and evidence.attack_strength >= 0.85
            )
            or evidence.dt1_critical
            or safety_level == SafetyLevel.CRITICAL
        )

        if critical_security and self.config.emergency_on_critical:
            return CompareClampResult(
                dt1_command=dt1,
                dt2_command=dt2,
                safe_lower_bound=safe_lower,
                safe_upper_bound=safe_upper,
                final_command=None,
                action=ClampAction.EMERGENCY_STOP,
                safety_level=SafetyLevel.CRITICAL,
                disagreement=disagreement,
                relative_disagreement=relative_disagreement,
                clamped=False,
                predictive_authority=False,
                reason=(
                    "Critical safety/security evidence triggered "
                    "emergency-stop policy."
                ),
                evidence=evidence,
                metadata=metadata_dict,
            )

        if (
            evidence.critical_constraint_violation
            and self.config.block_on_constraint_violation
        ):
            return CompareClampResult(
                dt1_command=dt1,
                dt2_command=dt2,
                safe_lower_bound=safe_lower,
                safe_upper_bound=safe_upper,
                final_command=dt2,
                action=ClampAction.USE_DT2,
                safety_level=SafetyLevel.CRITICAL,
                disagreement=disagreement,
                relative_disagreement=relative_disagreement,
                clamped=False,
                predictive_authority=False,
                reason=(
                    "Critical safety constraint violation; "
                    "DT1 predictive authority blocked."
                ),
                evidence=evidence,
                metadata=metadata_dict,
            )

        if (
            evidence.attack_detected
            and self.config.block_on_attack
            and evidence.attack_strength >= 0.60
        ):
            return CompareClampResult(
                dt1_command=dt1,
                dt2_command=dt2,
                safe_lower_bound=safe_lower,
                safe_upper_bound=safe_upper,
                final_command=dt2,
                action=ClampAction.USE_DT2,
                safety_level=(
                    SafetyLevel.CRITICAL
                    if evidence.attack_strength >= 0.85
                    else SafetyLevel.HIGH
                ),
                disagreement=disagreement,
                relative_disagreement=relative_disagreement,
                clamped=False,
                predictive_authority=False,
                reason=(
                    "Attack evidence is strong enough to block "
                    "DT1 predictive authority."
                ),
                evidence=evidence,
                metadata=metadata_dict,
            )

        if integrity_failure:
            return CompareClampResult(
                dt1_command=dt1,
                dt2_command=dt2,
                safe_lower_bound=safe_lower,
                safe_upper_bound=safe_upper,
                final_command=dt2,
                action=ClampAction.USE_DT2,
                safety_level=SafetyLevel.HIGH,
                disagreement=disagreement,
                relative_disagreement=relative_disagreement,
                clamped=False,
                predictive_authority=False,
                reason=(
                    "Sensor/controller/pump/communication integrity "
                    "cannot be established; DT2 retained as trusted "
                    "reference."
                ),
                evidence=evidence,
                metadata=metadata_dict,
            )

        if (
            evidence.dt1_critical
            or (
                evidence.hazard_score is not None
                and evidence.hazard_score
                >= self.config.hazard_critical
            )
            or (
                evidence.hazard_probability is not None
                and evidence.hazard_probability
                >= self.config.hazard_critical
            )
            or (
                evidence.risk_score is not None
                and evidence.risk_score
                >= self.config.risk_critical
            )
        ):
            return CompareClampResult(
                dt1_command=dt1,
                dt2_command=dt2,
                safe_lower_bound=safe_lower,
                safe_upper_bound=safe_upper,
                final_command=dt2,
                action=ClampAction.USE_DT2,
                safety_level=SafetyLevel.CRITICAL,
                disagreement=disagreement,
                relative_disagreement=relative_disagreement,
                clamped=False,
                predictive_authority=False,
                reason=(
                    "Critical DT1/hazard/risk evidence prevents "
                    "predictive command authority."
                ),
                evidence=evidence,
                metadata=metadata_dict,
            )

        if (
            disagreement >= self.config.disagreement_critical
        ):
            return CompareClampResult(
                dt1_command=dt1,
                dt2_command=dt2,
                safe_lower_bound=safe_lower,
                safe_upper_bound=safe_upper,
                final_command=dt2,
                action=ClampAction.USE_DT2,
                safety_level=SafetyLevel.CRITICAL,
                disagreement=disagreement,
                relative_disagreement=relative_disagreement,
                clamped=False,
                predictive_authority=False,
                reason=(
                    "DT1 and DT2 commands differ beyond the critical "
                    "disagreement threshold."
                ),
                evidence=evidence,
                metadata=metadata_dict,
            )

        if dt1 < safe_lower or dt1 > safe_upper:
            clamped_command = min(
                max(dt1, safe_lower),
                safe_upper,
            )

            return CompareClampResult(
                dt1_command=dt1,
                dt2_command=dt2,
                safe_lower_bound=safe_lower,
                safe_upper_bound=safe_upper,
                final_command=clamped_command,
                action=ClampAction.CLAMP_TO_SAFE_RANGE,
                safety_level=max(
                    safety_level,
                    SafetyLevel.WARNING,
                    key=lambda level: level_order[level],
                ),
                disagreement=disagreement,
                relative_disagreement=relative_disagreement,
                clamped=True,
                predictive_authority=True,
                reason=(
                    "DT1 command exceeded the DT2-centered safety "
                    "envelope and was clamped."
                ),
                evidence=evidence,
                metadata=metadata_dict,
            )

        if (
            evidence.constraint_violation
            and self.config.block_on_constraint_violation
        ):
            return CompareClampResult(
                dt1_command=dt1,
                dt2_command=dt2,
                safe_lower_bound=safe_lower,
                safe_upper_bound=safe_upper,
                final_command=dt2,
                action=ClampAction.USE_DT2,
                safety_level=SafetyLevel.HIGH,
                disagreement=disagreement,
                relative_disagreement=relative_disagreement,
                clamped=False,
                predictive_authority=False,
                reason=(
                    "Safety constraint violation prevents DT1 "
                    "predictive authority."
                ),
                evidence=evidence,
                metadata=metadata_dict,
            )

        if (
            evidence.dt1_anomaly
            or disagreement >= self.config.disagreement_high
        ):
            return CompareClampResult(
                dt1_command=dt1,
                dt2_command=dt2,
                safe_lower_bound=safe_lower,
                safe_upper_bound=safe_upper,
                final_command=dt2,
                action=ClampAction.USE_DT2,
                safety_level=SafetyLevel.HIGH,
                disagreement=disagreement,
                relative_disagreement=relative_disagreement,
                clamped=False,
                predictive_authority=False,
                reason=(
                    "DT1 anomaly or high DT1/DT2 disagreement "
                    "caused fallback to DT2."
                ),
                evidence=evidence,
                metadata=metadata_dict,
            )

        return CompareClampResult(
            dt1_command=dt1,
            dt2_command=dt2,
            safe_lower_bound=safe_lower,
            safe_upper_bound=safe_upper,
            final_command=dt1,
            action=ClampAction.ALLOW_DT1,
            safety_level=safety_level,
            disagreement=disagreement,
            relative_disagreement=relative_disagreement,
            clamped=False,
            predictive_authority=True,
            reason=(
                "DT1 command is within the DT2-centered safety envelope "
                "and no hard safety override was triggered."
            ),
            evidence=evidence,
            metadata=metadata_dict,
        )


def compare_and_clamp(
    dt1_command: Optional[float],
    dt2_command: Optional[float],
    evidence: Optional[SafetyEvidence] = None,
    config: Optional[CompareClampConfig] = None,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> CompareClampResult:
    """Functional convenience wrapper."""
    engine = CompareAndClamp(config=config)

    return engine.compare(
        dt1_command=dt1_command,
        dt2_command=dt2_command,
        evidence=evidence,
        metadata=metadata,
    )


def build_evidence_from_mapping(
    values: Mapping[str, Any],
) -> SafetyEvidence:
    """
    Construct SafetyEvidence from a dictionary.

    Unknown keys are ignored so callers can pass richer evidence records
    from the security layer.
    """
    allowed = {
        "hazard_score",
        "hazard_probability",
        "risk_score",
        "attack_detected",
        "attack_strength",
        "constraint_violation",
        "critical_constraint_violation",
        "sensor_integrity_ok",
        "controller_integrity_ok",
        "pump_integrity_ok",
        "communication_ok",
        "dt1_anomaly",
        "dt1_critical",
    }

    kwargs = {
        key: values[key]
        for key in allowed
        if key in values
    }

    metadata = {
        key: value
        for key, value in values.items()
        if key not in allowed
    }

    kwargs["metadata"] = metadata

    return SafetyEvidence(**kwargs)


__all__ = [
    "ClampAction",
    "SafetyLevel",
    "CompareClampConfig",
    "SafetyEvidence",
    "CompareClampResult",
    "CompareAndClamp",
    "compare_and_clamp",
    "build_evidence_from_mapping",
]
