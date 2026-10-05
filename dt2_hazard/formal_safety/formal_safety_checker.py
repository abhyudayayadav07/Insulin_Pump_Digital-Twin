"""Evaluation engine for formal safety specifications."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

from .safety_specification import (
    SafetySpecification,
    SafetySpecificationRegistry,
    SpecificationEvaluation,
)


@dataclass
class FormalSafetyReport:
    evaluations: list[SpecificationEvaluation] = field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        return all(item.satisfied for item in self.evaluations)

    @property
    def violations(self):
        return [item for item in self.evaluations if not item.satisfied]

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    @property
    def critical_violation(self) -> bool:
        return any(
            item.severity.value == "critical" and not item.satisfied
            for item in self.evaluations
        )


class FormalSafetyChecker:
    def __init__(self, registry: Optional[SafetySpecificationRegistry] = None):
        self.registry = registry or SafetySpecificationRegistry()

    def check(self, context: Mapping[str, Any]) -> FormalSafetyReport:
        report = FormalSafetyReport()
        for spec in self.registry.all():
            report.evaluations.append(
                SpecificationEvaluation(
                    spec_id=spec.spec_id,
                    satisfied=spec.evaluate(context),
                    severity=spec.severity,
                    message="satisfied" if spec.evaluate(context) else "violated",
                )
            )
        return report


def check_formal_safety(
    context: Mapping[str, Any],
    specifications: Iterable[SafetySpecification],
) -> FormalSafetyReport:
    registry = SafetySpecificationRegistry(specifications)
    return FormalSafetyChecker(registry).check(context)
