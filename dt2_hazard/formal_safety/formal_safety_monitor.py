"""Runtime formal-safety monitor for DT2."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional

from .formal_safety_checker import FormalSafetyChecker, FormalSafetyReport
from .safety_invariants import SafetyInvariant, check_invariants


class MonitorState(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class FormalSafetyMonitorResult:
    state: MonitorState
    formal_report: Optional[FormalSafetyReport]
    invariant_results: Dict[str, bool]
    violation_count: int
    critical: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class FormalSafetyMonitor:
    def __init__(self, checker: Optional[FormalSafetyChecker] = None, invariants=None):
        self.checker = checker or FormalSafetyChecker()
        self.invariants = list(invariants) if invariants is not None else []

    def evaluate(self, context: Mapping[str, Any]) -> FormalSafetyMonitorResult:
        report = self.checker.check(context)
        inv_results = check_invariants(context, self.invariants) if self.invariants else {}
        invariant_failures = sum(not value for value in inv_results.values())

        total_violations = report.violation_count + invariant_failures
        critical = report.critical_violation

        if critical:
            state = MonitorState.CRITICAL
        elif total_violations > 0:
            state = MonitorState.VIOLATION
        else:
            state = MonitorState.SAFE

        return FormalSafetyMonitorResult(
            state=state,
            formal_report=report,
            invariant_results=inv_results,
            violation_count=total_violations,
            critical=critical,
        )
