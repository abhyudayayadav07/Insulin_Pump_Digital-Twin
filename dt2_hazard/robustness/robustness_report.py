"""Structured reporting for robustness experiments."""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from .robustness_metrics import RobustnessMetrics


@dataclass
class RobustnessReport:
    experiment_name: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    scenarios: List[Dict[str, Any]] = field(default_factory=list)

    def add_scenario(self, scenario_id: str, **results):
        self.scenarios.append({"scenario_id": scenario_id, **results})

    def summary(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "scenario_count": len(self.scenarios),
            "metrics": dict(self.metrics),
        }

    def to_dict(self):
        return {
            "experiment_name": self.experiment_name,
            "metrics": dict(self.metrics),
            "scenarios": list(self.scenarios),
        }


def build_report(experiment_name: str, metrics=None, scenarios=None):
    report = RobustnessReport(
        experiment_name=experiment_name,
        metrics=dict(metrics or {}),
    )
    for scenario in scenarios or []:
        scenario = dict(scenario)
        scenario_id = scenario.pop("scenario_id", "unknown")
        report.add_scenario(scenario_id, **scenario)
    return report
