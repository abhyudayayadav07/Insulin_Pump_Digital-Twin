"""
Decision engine for the insulin digital twin safety architecture.

The decision engine converts fused evidence and system context into a
high-level safety action. It does NOT calculate an insulin dose and does
NOT directly actuate the insulin pump.

Expected flow:
    DT1 / DT2 / Security evidence
              |
              v
       Evidence Fusion
              |
              v
        Decision Engine
              |
              v
       Safety Response
              |
              v
        Pump interface

Actions:
    ALLOW
    MONITOR
    WARN
    BLOCK_ML
    USE_SAFE_TWIN
    SAFE_MODE
    EMERGENCY_STOP
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional


class DecisionAction(str, Enum):
    ALLOW = "ALLOW"
    MONITOR = "MONITOR"
    WARN = "WARN"
    BLOCK_ML = "BLOCK_ML"
    USE_SAFE_TWIN = "USE_SAFE_TWIN"
    SAFE_MODE = "SAFE_MODE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class DecisionReason(str, Enum):
    NORMAL_OPERATION = "NORMAL_OPERATION"
    ELEVATED_MONITORING = "ELEVATED_MONITORING"
    HAZARD_WARNING = "HAZARD_WARNING"
    CRITICAL_HAZARD = "CRITICAL_HAZARD"
    SAFETY_CONSTRAINT = "SAFETY_CONSTRAINT"
    STRONG_ATTACK_EVIDENCE = "STRONG_ATTACK_EVIDENCE"
    DT1_DT2_DISAGREEMENT = "DT1_DT2_DISAGREEMENT"
    PUMP_MISMATCH = "PUMP_MISMATCH"
    SENSOR_INTEGRITY = "SENSOR_INTEGRITY"
    CONTROLLER_INTEGRITY = "CONTROLLER_INTEGRITY"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    UNKNOWN = "UNKNOWN"


@dataclass
class DecisionConfig:
    """Thresholds and policy settings for the decision engine."""

    monitor_threshold: float = 0.25
    warning_threshold: float = 0.50
    block_threshold: float = 0.65
    safe_mode_threshold: float = 0.80
    emergency_threshold: float = 0.90

    dt_disagreement_threshold: float = 0.50
    pump_mismatch_threshold: float = 0.50
    attack_threshold: float = 0.70

    minimum_predictive_confidence: float = 0.50
    allow_predictive_when_unknown: bool = False

    enable_hard_constraint_override: bool = True
    enable_attack_override: bool = True
    enable_pump_mismatch_override: bool = True
    enable_dt_disagreement_override: bool = True


@dataclass
class DecisionContext:
    """
    Inputs to the decision engine.

    Scores are research/engineering signals, not calibrated clinical
    probabilities.
    """

    fused_score: float = 0.0
    fused_state: Optional[str] = None

    dt1_confidence: float = 1.0
    dt1_dt2_disagreement: float = 0.0

    hazard_score: float = 0.0
    hazard_probability: float = 0.0

    attack_score: float = 0.0
    pump_mismatch_score: float = 0.0

    sensor_integrity: float = 1.0
    controller_integrity: float = 1.0

    critical_constraint_violated: bool = False
    safety_constraint_violated: bool = False

    manual_override: Optional[DecisionAction] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionResult:
    """Decision produced by the engine."""

    action: DecisionAction
    reason: DecisionReason
    confidence: float
    score: float
    predictive_authority: bool
    safe_twin_authority: bool
    emergency_stop: bool
    explanation: str
    triggered_conditions: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason.value,
            "confidence": self.confidence,
            "score": self.score,
            "predictive_authority": self.predictive_authority,
            "safe_twin_authority": self.safe_twin_authority,
            "emergency_stop": self.emergency_stop,
            "explanation": self.explanation,
            "triggered_conditions": list(self.triggered_conditions),
            "metadata": dict(self.metadata),
        }


class DecisionEngine:
    """
    High-level safety decision engine.

    Priority is intentionally conservative:

        EMERGENCY_STOP
            >
        SAFE_MODE
            >
        USE_SAFE_TWIN / BLOCK_ML
            >
        WARN
            >
        MONITOR
            >
        ALLOW

    The engine only selects authority/action. It does not issue a pump
    command or determine an insulin dose.
    """

    def __init__(self, config: Optional[DecisionConfig] = None):
        self.config = config or DecisionConfig()
        self.last_result: Optional[DecisionResult] = None

    @staticmethod
    def _clip(value: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _base_score(self, context: DecisionContext) -> float:
        scores = [
            self._clip(context.fused_score),
            self._clip(context.hazard_score),
            self._clip(context.hazard_probability),
            self._clip(context.attack_score),
            self._clip(context.pump_mismatch_score),
            self._clip(context.dt1_dt2_disagreement),
            1.0 - self._clip(context.sensor_integrity),
            1.0 - self._clip(context.controller_integrity),
        ]
        return max(scores)

    def decide(self, context: DecisionContext) -> DecisionResult:
        cfg = self.config
        score = self._base_score(context)
        triggered = []

        # Manual override has highest authority.
        if context.manual_override is not None:
            action = context.manual_override
            result = self._build_result(
                action=action,
                reason=DecisionReason.MANUAL_OVERRIDE,
                score=score,
                confidence=1.0,
                triggered=["manual_override"],
                explanation=f"Manual override requested action {action.value}.",
            )
            self.last_result = result
            return result

        # Critical safety constraint.
        if cfg.enable_hard_constraint_override and context.critical_constraint_violated:
            triggered.append("critical_constraint")
            result = self._build_result(
                DecisionAction.EMERGENCY_STOP,
                DecisionReason.SAFETY_CONSTRAINT,
                score=max(score, 1.0),
                confidence=1.0,
                triggered=triggered,
                explanation="A critical safety constraint is violated.",
            )
            self.last_result = result
            return result

        # Strong pump mismatch means commanded delivery cannot be trusted.
        if (
            cfg.enable_pump_mismatch_override
            and context.pump_mismatch_score >= cfg.pump_mismatch_threshold
        ):
            triggered.append("pump_mismatch")
            result = self._build_result(
                DecisionAction.SAFE_MODE,
                DecisionReason.PUMP_MISMATCH,
                score=max(score, context.pump_mismatch_score),
                confidence=1.0,
                triggered=triggered,
                explanation="Pump command and actual delivery show a significant mismatch.",
            )
            self.last_result = result
            return result

        # Strong attack evidence.
        if (
            cfg.enable_attack_override
            and context.attack_score >= cfg.attack_threshold
        ):
            triggered.append("strong_attack_evidence")
            result = self._build_result(
                DecisionAction.SAFE_MODE,
                DecisionReason.STRONG_ATTACK_EVIDENCE,
                score=max(score, context.attack_score),
                confidence=1.0,
                triggered=triggered,
                explanation="Strong attack evidence reduces trust in the predictive control path.",
            )
            self.last_result = result
            return result

        # DT1/DT2 disagreement: predictive model loses authority.
        if (
            cfg.enable_dt_disagreement_override
            and context.dt1_dt2_disagreement >= cfg.dt_disagreement_threshold
        ):
            triggered.append("dt1_dt2_disagreement")
            result = self._build_result(
                DecisionAction.USE_SAFE_TWIN,
                DecisionReason.DT1_DT2_DISAGREEMENT,
                score=max(score, context.dt1_dt2_disagreement),
                confidence=1.0,
                triggered=triggered,
                explanation="DT1 and DT2 disagree beyond the configured safety threshold.",
            )
            self.last_result = result
            return result

        # Invalid sensor/controller integrity blocks predictive authority.
        if context.sensor_integrity < cfg.minimum_predictive_confidence:
            triggered.append("sensor_integrity")
            result = self._build_result(
                DecisionAction.USE_SAFE_TWIN,
                DecisionReason.SENSOR_INTEGRITY,
                score=max(score, 1.0 - context.sensor_integrity),
                confidence=context.sensor_integrity,
                triggered=triggered,
                explanation="Sensor integrity is too low for predictive authority.",
            )
            self.last_result = result
            return result

        if context.controller_integrity < cfg.minimum_predictive_confidence:
            triggered.append("controller_integrity")
            result = self._build_result(
                DecisionAction.USE_SAFE_TWIN,
                DecisionReason.CONTROLLER_INTEGRITY,
                score=max(score, 1.0 - context.controller_integrity),
                confidence=context.controller_integrity,
                triggered=triggered,
                explanation="Controller integrity is too low for predictive authority.",
            )
            self.last_result = result
            return result

        # Low DT1 confidence prevents predictive authority.
        if context.dt1_confidence < cfg.minimum_predictive_confidence:
            triggered.append("low_dt1_confidence")
            result = self._build_result(
                DecisionAction.USE_SAFE_TWIN,
                DecisionReason.LOW_CONFIDENCE,
                score=max(score, 1.0 - context.dt1_confidence),
                confidence=context.dt1_confidence,
                triggered=triggered,
                explanation="DT1 confidence is below the configured predictive-authority threshold.",
            )
            self.last_result = result
            return result

        # Safety escalation based on fused score.
        if score >= cfg.emergency_threshold:
            triggered.append("emergency_threshold")
            action = DecisionAction.EMERGENCY_STOP
            reason = DecisionReason.CRITICAL_HAZARD
        elif score >= cfg.safe_mode_threshold:
            triggered.append("safe_mode_threshold")
            action = DecisionAction.SAFE_MODE
            reason = DecisionReason.CRITICAL_HAZARD
        elif score >= cfg.block_threshold:
            triggered.append("block_threshold")
            action = DecisionAction.BLOCK_ML
            reason = DecisionReason.SAFETY_CONSTRAINT
        elif score >= cfg.warning_threshold:
            triggered.append("warning_threshold")
            action = DecisionAction.WARN
            reason = DecisionReason.HAZARD_WARNING
        elif score >= cfg.monitor_threshold:
            triggered.append("monitor_threshold")
            action = DecisionAction.MONITOR
            reason = DecisionReason.ELEVATED_MONITORING
        else:
            action = DecisionAction.ALLOW
            reason = DecisionReason.NORMAL_OPERATION

        confidence = self._decision_confidence(context, score)

        # Do not grant predictive authority when confidence is insufficient.
        if (
            action == DecisionAction.ALLOW
            and confidence < cfg.minimum_predictive_confidence
            and not cfg.allow_predictive_when_unknown
        ):
            action = DecisionAction.USE_SAFE_TWIN
            reason = DecisionReason.LOW_CONFIDENCE
            triggered.append("predictive_confidence_gate")

        result = self._build_result(
            action=action,
            reason=reason,
            score=score,
            confidence=confidence,
            triggered=triggered,
            explanation=self._explanation(action, reason, score),
        )
        self.last_result = result
        return result

    def _decision_confidence(
        self, context: DecisionContext, score: float
    ) -> float:
        values = [
            self._clip(context.dt1_confidence),
            self._clip(context.sensor_integrity),
            self._clip(context.controller_integrity),
        ]
        confidence = float(np_mean(values))
        if context.hazard_probability > 0.0:
            confidence = min(
                1.0,
                0.7 * confidence + 0.3 * self._clip(context.hazard_probability),
            )
        return confidence

    def _build_result(
        self,
        action: DecisionAction,
        reason: DecisionReason,
        score: float,
        confidence: float,
        triggered: list,
        explanation: str,
    ) -> DecisionResult:
        predictive = action in {
            DecisionAction.ALLOW,
            DecisionAction.MONITOR,
            DecisionAction.WARN,
        }
        safe_twin = action in {
            DecisionAction.BLOCK_ML,
            DecisionAction.USE_SAFE_TWIN,
            DecisionAction.SAFE_MODE,
        }
        emergency = action == DecisionAction.EMERGENCY_STOP

        return DecisionResult(
            action=action,
            reason=reason,
            confidence=self._clip(confidence),
            score=self._clip(score),
            predictive_authority=predictive,
            safe_twin_authority=safe_twin,
            emergency_stop=emergency,
            explanation=explanation,
            triggered_conditions=list(triggered),
        )

    @staticmethod
    def _explanation(
        action: DecisionAction,
        reason: DecisionReason,
        score: float,
    ) -> str:
        return (
            f"Selected action={action.value}; reason={reason.value}; "
            f"decision score={score:.3f}."
        )

    def reset(self):
        self.last_result = None


def np_mean(values):
    """Small dependency-free mean helper."""
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def decide(
    context: DecisionContext,
    config: Optional[DecisionConfig] = None,
) -> DecisionResult:
    """Convenience function for a one-shot decision."""
    return DecisionEngine(config).decide(context)


__all__ = [
    "DecisionAction",
    "DecisionReason",
    "DecisionConfig",
    "DecisionContext",
    "DecisionResult",
    "DecisionEngine",
    "decide",
]
