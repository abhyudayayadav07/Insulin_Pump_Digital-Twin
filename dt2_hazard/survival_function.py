"""
Survival-function component for the hazard/survival Digital Twin 2.

DT2 is intentionally different from the GlucOS reactive-safe controller.
Its purpose is to estimate how long the system is expected to remain
free from a specified hazardous event.

Core quantities
---------------
Hazard function:
    h(t | x)

Cumulative hazard:
    H(t | x) = integral_0^t h(u | x) du

Survival function:
    S(t | x) = exp(-H(t | x))

Event probability by horizon:
    F(t | x) = 1 - S(t | x)

This is a mathematical/engineering survival-analysis component.
It does not claim clinical calibration by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math


class SurvivalStatus(str, Enum):
    VERY_SAFE = "very_safe"
    SAFE = "safe"
    MONITOR = "monitor"
    WARNING = "warning"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SurvivalConfig:
    """Configuration for survival-function calculations."""

    min_hazard: float = 0.0
    max_hazard: float = 100.0

    very_safe_event_probability: float = 0.01
    safe_event_probability: float = 0.05
    monitor_event_probability: float = 0.15
    warning_event_probability: float = 0.30
    high_risk_event_probability: float = 0.60
    critical_event_probability: float = 0.85

    integration_method: str = "trapezoid"

    def __post_init__(self) -> None:
        if self.min_hazard < 0:
            raise ValueError("min_hazard must be >= 0.")
        if self.max_hazard < self.min_hazard:
            raise ValueError("max_hazard must be >= min_hazard.")

        thresholds = (
            self.very_safe_event_probability,
            self.safe_event_probability,
            self.monitor_event_probability,
            self.warning_event_probability,
            self.high_risk_event_probability,
            self.critical_event_probability,
        )

        if any(not 0 <= x <= 1 for x in thresholds):
            raise ValueError(
                "Event-probability thresholds must be between 0 and 1."
            )

        if any(
            thresholds[i] > thresholds[i + 1]
            for i in range(len(thresholds) - 1)
        ):
            raise ValueError(
                "Event-probability thresholds must be non-decreasing."
            )

        if self.integration_method not in {"trapezoid", "left", "right"}:
            raise ValueError(
                "integration_method must be 'trapezoid', 'left', or 'right'."
            )


@dataclass(frozen=True)
class SurvivalPoint:
    """Survival state at one time horizon."""

    time: float
    hazard: float
    cumulative_hazard: float
    survival_probability: float
    event_probability: float
    status: SurvivalStatus

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "hazard": self.hazard,
            "cumulative_hazard": self.cumulative_hazard,
            "survival_probability": self.survival_probability,
            "event_probability": self.event_probability,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class SurvivalResult:
    """Complete survival calculation over a time horizon."""

    times: Tuple[float, ...]
    hazards: Tuple[float, ...]
    cumulative_hazard: Tuple[float, ...]
    survival_probability: Tuple[float, ...]
    event_probability: Tuple[float, ...]
    points: Tuple[SurvivalPoint, ...]
    event_name: str = "hazard_event"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def final_survival(self) -> float:
        return self.survival_probability[-1]

    @property
    def final_event_probability(self) -> float:
        return self.event_probability[-1]

    @property
    def final_cumulative_hazard(self) -> float:
        return self.cumulative_hazard[-1]

    @property
    def max_hazard(self) -> float:
        return max(self.hazards)

    @property
    def minimum_survival(self) -> float:
        return min(self.survival_probability)

    @property
    def final_status(self) -> SurvivalStatus:
        return self.points[-1].status

    def point_at(self, horizon: float) -> SurvivalPoint:
        if not self.points:
            raise ValueError("SurvivalResult contains no points.")
        return min(self.points, key=lambda p: abs(p.time - horizon))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_name": self.event_name,
            "times": list(self.times),
            "hazards": list(self.hazards),
            "cumulative_hazard": list(self.cumulative_hazard),
            "survival_probability": list(self.survival_probability),
            "event_probability": list(self.event_probability),
            "points": [p.to_dict() for p in self.points],
            "metadata": dict(self.metadata),
        }

    def to_records(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.points]


@dataclass(frozen=True)
class MultiEventSurvivalResult:
    """Separate survival curves for several event types."""

    results: Dict[str, SurvivalResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def event_probabilities_at(self, horizon: float) -> Dict[str, float]:
        return {
            name: result.point_at(horizon).event_probability
            for name, result in self.results.items()
        }

    def highest_event_probability(
        self,
        horizon: float,
    ) -> Tuple[Optional[str], float]:
        values = self.event_probabilities_at(horizon)
        if not values:
            return None, 0.0
        name = max(values, key=values.get)
        return name, values[name]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": {
                name: result.to_dict()
                for name, result in self.results.items()
            },
            "metadata": dict(self.metadata),
        }


class SurvivalFunction:
    """
    Numerical survival-function calculator.

    `hazard` may be:
      - a scalar hazard rate,
      - a sequence aligned with `times`,
      - a callable h(t).

    For a continuous hazard:
        H(t) = integral h(u) du
        S(t) = exp(-H(t))
        F(t) = 1 - S(t)
    """

    def __init__(self, config: Optional[SurvivalConfig] = None) -> None:
        self.config = config or SurvivalConfig()

    def _sanitize_hazard(self, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Hazard values must be finite.")
        return max(
            self.config.min_hazard,
            min(self.config.max_hazard, value),
        )

    @staticmethod
    def _validate_times(times: Sequence[float]) -> List[float]:
        values = [float(t) for t in times]
        if not values:
            raise ValueError("At least one time point is required.")
        if any(not math.isfinite(t) for t in values):
            raise ValueError("All time points must be finite.")
        if values[0] < 0:
            raise ValueError("Time points must be >= 0.")
        if any(b < a for a, b in zip(values, values[1:])):
            raise ValueError("Time points must be non-decreasing.")
        return values

    def _resolve_hazards(
        self,
        times: Sequence[float],
        hazard: Any,
    ) -> List[float]:
        if callable(hazard):
            return [
                self._sanitize_hazard(hazard(t))
                for t in times
            ]

        if isinstance(hazard, (int, float)):
            value = self._sanitize_hazard(hazard)
            return [value for _ in times]

        values = [self._sanitize_hazard(v) for v in hazard]
        if len(values) != len(times):
            raise ValueError(
                "Hazard sequence length must match times length."
            )
        return values

    def cumulative_hazard(
        self,
        times: Sequence[float],
        hazards: Sequence[float],
    ) -> List[float]:
        """Numerically integrate h(t) to obtain H(t)."""
        t = self._validate_times(times)
        if len(hazards) != len(t):
            raise ValueError(
                "Hazard sequence length must match times length."
            )

        h = [self._sanitize_hazard(v) for v in hazards]
        cumulative = [0.0]

        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]

            if self.config.integration_method == "trapezoid":
                increment = 0.5 * (h[i] + h[i - 1]) * dt
            elif self.config.integration_method == "left":
                increment = h[i - 1] * dt
            else:
                increment = h[i] * dt

            cumulative.append(
                cumulative[-1] + max(0.0, increment)
            )

        return cumulative

    @staticmethod
    def _survival_from_cumulative_hazard(
        cumulative_hazard: float,
    ) -> float:
        if cumulative_hazard <= 0:
            return 1.0
        if cumulative_hazard >= 745:
            return 0.0
        return max(
            0.0,
            min(1.0, math.exp(-cumulative_hazard)),
        )

    def classify_event_probability(
        self,
        event_probability: float,
    ) -> SurvivalStatus:
        p = max(0.0, min(1.0, float(event_probability)))

        if p < self.config.very_safe_event_probability:
            return SurvivalStatus.VERY_SAFE
        if p < self.config.safe_event_probability:
            return SurvivalStatus.SAFE
        if p < self.config.monitor_event_probability:
            return SurvivalStatus.MONITOR
        if p < self.config.warning_event_probability:
            return SurvivalStatus.WARNING
        if p < self.config.high_risk_event_probability:
            return SurvivalStatus.HIGH_RISK
        return SurvivalStatus.CRITICAL

    def calculate(
        self,
        times: Sequence[float],
        hazard: Any,
        *,
        event_name: str = "hazard_event",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SurvivalResult:
        """
        Calculate H(t), S(t), and F(t) from a supplied hazard.

        Recommended DT2 time unit: minutes.
        """
        t = self._validate_times(times)
        h = self._resolve_hazards(t, hazard)
        H = self.cumulative_hazard(t, h)

        survival = [
            self._survival_from_cumulative_hazard(v)
            for v in H
        ]

        event_probability = [
            max(0.0, min(1.0, 1.0 - s))
            for s in survival
        ]

        points = tuple(
            SurvivalPoint(
                time=t_i,
                hazard=h_i,
                cumulative_hazard=H_i,
                survival_probability=S_i,
                event_probability=F_i,
                status=self.classify_event_probability(F_i),
            )
            for t_i, h_i, H_i, S_i, F_i in zip(
                t, h, H, survival, event_probability
            )
        )

        return SurvivalResult(
            times=tuple(t),
            hazards=tuple(h),
            cumulative_hazard=tuple(H),
            survival_probability=tuple(survival),
            event_probability=tuple(event_probability),
            points=points,
            event_name=event_name,
            metadata=dict(metadata or {}),
        )

    def calculate_constant_hazard(
        self,
        times: Sequence[float],
        hazard_rate: float,
        *,
        event_name: str = "hazard_event",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SurvivalResult:
        return self.calculate(
            times,
            hazard_rate,
            event_name=event_name,
            metadata=metadata,
        )

    def calculate_from_callable(
        self,
        times: Sequence[float],
        hazard_function: Callable[[float], float],
        *,
        event_name: str = "hazard_event",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SurvivalResult:
        return self.calculate(
            times,
            hazard_function,
            event_name=event_name,
            metadata=metadata,
        )

    def calculate_multiple_events(
        self,
        times: Sequence[float],
        hazards: Mapping[str, Any],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MultiEventSurvivalResult:
        """
        Calculate separate event-specific survival curves.

        Event probabilities are intentionally NOT summed here. Summing
        cause-specific event probabilities requires a competing-risks
        formulation and cause-specific cumulative incidence.
        """
        results = {
            name: self.calculate(
                times,
                event_hazard,
                event_name=name,
            )
            for name, event_hazard in hazards.items()
        }
        return MultiEventSurvivalResult(
            results=results,
            metadata=dict(metadata or {}),
        )

    def survival_at(
        self,
        horizon: float,
        hazard: Any,
        *,
        step: float = 1.0,
        event_name: str = "hazard_event",
    ) -> SurvivalPoint:
        """Calculate survival at an exact requested horizon."""
        horizon = float(horizon)
        step = float(step)

        if horizon < 0:
            raise ValueError("horizon must be >= 0.")
        if step <= 0:
            raise ValueError("step must be > 0.")

        if horizon == 0:
            times = [0.0]
        else:
            times = [0.0]
            current = step
            while current < horizon:
                times.append(current)
                current += step
            if times[-1] != horizon:
                times.append(horizon)

        result = self.calculate(
            times,
            hazard,
            event_name=event_name,
        )
        return result.points[-1]

    def event_probability_at(
        self,
        horizon: float,
        hazard: Any,
        *,
        step: float = 1.0,
    ) -> float:
        """Return P(T <= horizon)."""
        return self.survival_at(
            horizon,
            hazard,
            step=step,
        ).event_probability

    def estimate_mean_event_time(
        self,
        result: SurvivalResult,
    ) -> Optional[float]:
        """
        Estimate E[T] over the available finite horizon.

        E[T] = integral_0^infinity S(t) dt

        With finite simulation data this is a truncated estimate; no
        unvalidated tail extrapolation is performed.
        """
        if len(result.times) < 2:
            return None

        total = 0.0
        for i in range(1, len(result.times)):
            dt = result.times[i] - result.times[i - 1]
            total += (
                0.5
                * (
                    result.survival_probability[i]
                    + result.survival_probability[i - 1]
                )
                * dt
            )
        return total


def calculate_survival_function(
    times: Sequence[float],
    hazard: Any,
    *,
    event_name: str = "hazard_event",
    config: Optional[SurvivalConfig] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> SurvivalResult:
    """Functional convenience wrapper."""
    return SurvivalFunction(config=config).calculate(
        times,
        hazard,
        event_name=event_name,
        metadata=metadata,
    )


def survival_from_cumulative_hazard(
    cumulative_hazard: Iterable[float],
) -> List[float]:
    """Convert H(t) directly to S(t)=exp(-H(t))."""
    output: List[float] = []

    for value in cumulative_hazard:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(
                "Cumulative hazard values must be finite."
            )

        value = max(0.0, value)
        survival = 0.0 if value >= 745 else math.exp(-value)
        output.append(max(0.0, min(1.0, survival)))

    return output


def event_probability_from_survival(
    survival_probability: Iterable[float],
) -> List[float]:
    """Convert S(t) to F(t)=1-S(t)."""
    output: List[float] = []

    for value in survival_probability:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(
                "Survival probabilities must be finite."
            )

        value = max(0.0, min(1.0, value))
        output.append(1.0 - value)

    return output


__all__ = [
    "SurvivalStatus",
    "SurvivalConfig",
    "SurvivalPoint",
    "SurvivalResult",
    "MultiEventSurvivalResult",
    "SurvivalFunction",
    "calculate_survival_function",
    "survival_from_cumulative_hazard",
    "event_probability_from_survival",
]
