"""Safety response layer for the insulin digital twin.

Maps DecisionResult objects to explicit, auditable safety responses.
This is a research/engineering policy layer; it does not directly actuate
a physical insulin pump or calculate insulin doses.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence

try:
    from .decision import DecisionAction, DecisionReason, DecisionResult
except ImportError:
    from decision import DecisionAction, DecisionReason, DecisionResult

class SafetyResponseType(str, Enum):
    NONE = "none"
    CONTINUE_PREDICTIVE = "continue_predictive"
    INCREASE_MONITORING = "increase_monitoring"
    ISSUE_WARNING = "issue_warning"
    BLOCK_PREDICTIVE = "block_predictive"
    USE_REACTIVE_SAFE_TWIN = "use_reactive_safe_twin"
    ENTER_SAFE_MODE = "enter_safe_mode"
    EMERGENCY_STOP = "emergency_stop"

class ControlAuthority(str, Enum):
    NONE = "none"
    DT1_PREDICTIVE = "dt1_predictive"
    DT2_REACTIVE_SAFE = "dt2_reactive_safe"
    SAFETY_LAYER = "safety_layer"
    EMERGENCY = "emergency"

@dataclass
class SafetyResponseConfig:
    allow_predictive_during_monitor: bool = True
    allow_predictive_during_warning: bool = True
    require_dt2_for_fallback: bool = True
    safe_mode_latch: bool = True
    emergency_stop_latch: bool = True
    simulation_mode: bool = True

@dataclass
class SafetyResponse:
    response_type: SafetyResponseType
    control_authority: ControlAuthority
    allow_predictive_command: bool
    block_predictive_command: bool
    use_reactive_safe_twin: bool
    enter_safe_mode: bool
    emergency_stop: bool
    warning: bool
    monitoring_level: str
    reason: DecisionReason
    rationale: list[str] = field(default_factory=list)
    decision_action: str = ""
    fused_score: float = 0.0
    confidence: float = 0.0
    latched: bool = False
    reset_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response_type": self.response_type.value,
            "control_authority": self.control_authority.value,
            "allow_predictive_command": self.allow_predictive_command,
            "block_predictive_command": self.block_predictive_command,
            "use_reactive_safe_twin": self.use_reactive_safe_twin,
            "enter_safe_mode": self.enter_safe_mode,
            "emergency_stop": self.emergency_stop,
            "warning": self.warning,
            "monitoring_level": self.monitoring_level,
            "reason": self.reason.value,
            "rationale": list(self.rationale),
            "decision_action": self.decision_action,
            "fused_score": self.fused_score,
            "confidence": self.confidence,
            "latched": self.latched,
            "reset_required": self.reset_required,
            "metadata": dict(self.metadata),
        }

class SafetyResponseEngine:
    """Stateful response policy with optional safe-mode/stop latches."""
    def __init__(self, config: Optional[SafetyResponseConfig] = None):
        self.config = config or SafetyResponseConfig()
        self.safe_mode_latched = False
        self.emergency_stop_latched = False

    def generate(self, decision: DecisionResult, *, dt2_available: bool = True,
                 metadata: Optional[Mapping[str, Any]] = None) -> SafetyResponse:
        if self.emergency_stop_latched:
            return self._emergency(decision, ["Emergency-stop latch is active.",
                "Explicit reset is required before normal operation can resume."], metadata)
        if decision.action == DecisionAction.EMERGENCY_STOP:
            if self.config.emergency_stop_latch:
                self.emergency_stop_latched = True
            return self._emergency(decision, ["Decision engine requested an emergency stop.",
                "Predictive insulin authority is removed."], metadata)
        if self.safe_mode_latched:
            return self._safe_mode(decision, ["Safe-mode latch is active.",
                "DT2 reactive-safe control remains the trusted fallback."], metadata)
        if decision.action == DecisionAction.SAFE_MODE:
            if self.config.safe_mode_latch:
                self.safe_mode_latched = True
            return self._safe_mode(decision, ["Decision engine requested safe mode.",
                "Predictive ML control is no longer trusted."], metadata)
        if decision.action in (DecisionAction.USE_SAFE_TWIN, DecisionAction.BLOCK_ML):
            if self.config.require_dt2_for_fallback and not dt2_available:
                return self._safe_mode(decision, [
                    "Predictive control was blocked, but DT2 is unavailable.",
                    "The response layer escalates to safe mode."], metadata)
            return self._safe_twin(decision, metadata)
        if decision.action == DecisionAction.WARN:
            if self.config.allow_predictive_during_warning:
                return self._warning(decision, metadata)
            if dt2_available:
                return self._safe_twin(decision, metadata,
                    ["Predictive control during warning is disabled by configuration."])
            return self._safe_mode(decision, [
                "Predictive control during warning is disabled and DT2 is unavailable."], metadata)
        if decision.action == DecisionAction.MONITOR:
            if self.config.allow_predictive_during_monitor:
                return self._monitor(decision, metadata)
            if dt2_available:
                return self._safe_twin(decision, metadata,
                    ["Predictive control during monitoring is disabled by configuration."])
            return self._safe_mode(decision, [
                "Predictive control during monitoring is disabled and DT2 is unavailable."], metadata)
        return self._predictive(decision, metadata)

    def reset_safe_mode(self) -> None:
        self.safe_mode_latched = False
    def reset_emergency_stop(self) -> None:
        self.emergency_stop_latched = False
    def reset_all(self) -> None:
        self.safe_mode_latched = False
        self.emergency_stop_latched = False
    def status(self) -> Dict[str, bool]:
        return {"safe_mode_latched": self.safe_mode_latched,
                "emergency_stop_latched": self.emergency_stop_latched}

    def _base(self, decision: DecisionResult, response_type: SafetyResponseType,
              authority: ControlAuthority, *, allow: bool, block: bool,
              safe_twin: bool, safe_mode: bool, emergency: bool, warning: bool,
              monitoring: str, rationale: Sequence[str], metadata: Optional[Mapping[str, Any]],
              latched: bool = False, reset_required: bool = False) -> SafetyResponse:
        merged = dict(decision.metadata)
        if metadata:
            merged.update(dict(metadata))
        return SafetyResponse(response_type, authority, allow, block, safe_twin,
            safe_mode, emergency, warning, monitoring, decision.reason, list(rationale),
            decision.action.value, decision.fused_score, decision.confidence,
            latched, reset_required, merged)

    def _predictive(self, d, m):
        return self._base(d, SafetyResponseType.CONTINUE_PREDICTIVE,
            ControlAuthority.DT1_PREDICTIVE, allow=True, block=False, safe_twin=False,
            safe_mode=False, emergency=False, warning=False, monitoring="normal",
            rationale=["Predictive DT1 control is authorized.",
                       "No higher-priority safety response was triggered."], metadata=m)
    def _monitor(self, d, m):
        return self._base(d, SafetyResponseType.INCREASE_MONITORING,
            ControlAuthority.DT1_PREDICTIVE, allow=True, block=False, safe_twin=False,
            safe_mode=False, emergency=False, warning=False, monitoring="elevated",
            rationale=["Predictive DT1 control remains authorized.",
                       "Evidence is elevated; monitoring should be increased."], metadata=m)
    def _warning(self, d, m):
        return self._base(d, SafetyResponseType.ISSUE_WARNING,
            ControlAuthority.DT1_PREDICTIVE, allow=True, block=False, safe_twin=False,
            safe_mode=False, emergency=False, warning=True, monitoring="high",
            rationale=["Predictive DT1 control remains authorized by configuration.",
                       "A warning is raised because hazard evidence is elevated."], metadata=m)
    def _safe_twin(self, d, m, extra: Optional[Sequence[str]] = None):
        rationale = ["Predictive DT1 control is blocked.",
                      "DT2 reactive-safe twin becomes the trusted control authority."]
        if extra: rationale.extend(extra)
        return self._base(d, SafetyResponseType.USE_REACTIVE_SAFE_TWIN,
            ControlAuthority.DT2_REACTIVE_SAFE, allow=False, block=True, safe_twin=True,
            safe_mode=False, emergency=False, warning=True, monitoring="high",
            rationale=rationale, metadata=m)
    def _safe_mode(self, d, rationale, m):
        return self._base(d, SafetyResponseType.ENTER_SAFE_MODE,
            ControlAuthority.DT2_REACTIVE_SAFE, allow=False, block=True, safe_twin=True,
            safe_mode=True, emergency=False, warning=True, monitoring="critical",
            rationale=rationale, metadata=m, latched=self.config.safe_mode_latch,
            reset_required=self.config.safe_mode_latch)
    def _emergency(self, d, rationale, m):
        return self._base(d, SafetyResponseType.EMERGENCY_STOP,
            ControlAuthority.EMERGENCY, allow=False, block=True, safe_twin=False,
            safe_mode=False, emergency=True, warning=True, monitoring="emergency",
            rationale=rationale, metadata=m, latched=self.config.emergency_stop_latch,
            reset_required=self.config.emergency_stop_latch)

def generate_safety_response(decision: DecisionResult, *,
    config: Optional[SafetyResponseConfig] = None,
    dt2_available: bool = True) -> SafetyResponse:
    """One-shot convenience wrapper."""
    return SafetyResponseEngine(config).generate(decision, dt2_available=dt2_available)

__all__ = ["SafetyResponseType", "ControlAuthority", "SafetyResponseConfig",
           "SafetyResponse", "SafetyResponseEngine", "generate_safety_response"]
