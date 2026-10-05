"""
Time-to-event analysis component for DT2.

Converts DT2 survival/hazard outputs into explicit estimates of when a
hazardous event reaches specified probability thresholds.

Research/engineering component; not clinically calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math

try:
    from .survival_function import SurvivalFunction, SurvivalResult
except ImportError:
    from survival_function import SurvivalFunction, SurvivalResult  # type: ignore


class EventTimeStatus(str, Enum):
    NO_EVENT_IN_HORIZON = "no_event_in_horizon"
    DISTANT = "distant"
    UPCOMING = "upcoming"
    NEAR_TERM = "near_term"
    IMMINENT = "imminent"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TimeToEventConfig:
    warning_probability: float = 0.30
    high_risk_probability: float = 0.60
    critical_probability: float = 0.85

    warning_horizon_minutes: float = 30.0
    near_term_horizon_minutes: float = 15.0
    imminent_horizon_minutes: float = 5.0

    median_probability: float = 0.50
    quantiles: Tuple[float, ...] = (0.25, 0.50, 0.75, 0.90)

    def __post_init__(self) -> None:
        probabilities = (
            self.warning_probability,
            self.high_risk_probability,
            self.critical_probability,
            self.median_probability,
            *self.quantiles,
        )
        if any(not 0.0 <= float(p) <= 1.0 for p in probabilities):
            raise ValueError("All probabilities must be between 0 and 1.")
        if not (
            self.warning_probability
            <= self.high_risk_probability
            <= self.critical_probability
        ):
            raise ValueError("Risk thresholds must be ordered.")
        if not (
            self.warning_horizon_minutes > 0
            and self.near_term_horizon_minutes > 0
            and self.imminent_horizon_minutes > 0
        ):
            raise ValueError("Time horizons must be > 0.")
        if any(
            self.quantiles[i] >= self.quantiles[i + 1]
            for i in range(len(self.quantiles) - 1)
        ):
            raise ValueError("Quantiles must be strictly increasing.")


@dataclass(frozen=True)
class TimeToEventEstimate:
    event_name: str
    probability_level: float
    time_minutes: Optional[float]
    status: EventTimeStatus
    interpolated: bool = False
    reached_within_horizon: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_name": self.event_name,
            "probability_level": self.probability_level,
            "time_minutes": self.time_minutes,
            "status": self.status.value,
            "interpolated": self.interpolated,
            "reached_within_horizon": self.reached_within_horizon,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TimeToEventResult:
    event_name: str
    horizon_minutes: float
    current_event_probability: float
    final_event_probability: float
    median_time_minutes: Optional[float]
    expected_time_minutes: Optional[float]
    quantile_times: Dict[float, Optional[float]]
    warning_time_minutes: Optional[float]
    high_risk_time_minutes: Optional[float]
    critical_time_minutes: Optional[float]
    urgency: EventTimeStatus
    reached_warning: bool
    reached_high_risk: bool
    reached_critical: bool
    survival_result: Optional[SurvivalResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_name": self.event_name,
            "horizon_minutes": self.horizon_minutes,
            "current_event_probability": self.current_event_probability,
            "final_event_probability": self.final_event_probability,
            "median_time_minutes": self.median_time_minutes,
            "expected_time_minutes": self.expected_time_minutes,
            "quantile_times": {str(k): v for k, v in self.quantile_times.items()},
            "warning_time_minutes": self.warning_time_minutes,
            "high_risk_time_minutes": self.high_risk_time_minutes,
            "critical_time_minutes": self.critical_time_minutes,
            "urgency": self.urgency.value,
            "reached_warning": self.reached_warning,
            "reached_high_risk": self.reached_high_risk,
            "reached_critical": self.reached_critical,
            "metadata": dict(self.metadata),
        }


class TimeToEventAnalyzer:
    """Inverse-survival analyzer for estimating time-to-hazard."""

    def __init__(self, config: Optional[TimeToEventConfig] = None) -> None:
        self.config = config or TimeToEventConfig()

    @staticmethod
    def _validate_result(result: SurvivalResult) -> None:
        if not result.times:
            raise ValueError("SurvivalResult contains no time points.")
        if not (
            len(result.times)
            == len(result.event_probability)
            == len(result.survival_probability)
        ):
            raise ValueError("SurvivalResult arrays must have equal lengths.")

    @staticmethod
    def _interpolate_crossing(
        t0: float,
        p0: float,
        t1: float,
        p1: float,
        target_probability: float,
    ) -> Tuple[float, bool]:
        if p0 >= target_probability:
            return t0, False
        if p1 < target_probability:
            raise ValueError("No crossing exists in the supplied interval.")
        if p1 == p0:
            return t1, False
        fraction = (target_probability - p0) / (p1 - p0)
        fraction = max(0.0, min(1.0, fraction))
        return t0 + fraction * (t1 - t0), True

    def classify_time(
        self,
        time_minutes: Optional[float],
        probability: float,
    ) -> EventTimeStatus:
        if time_minutes is None:
            return EventTimeStatus.NO_EVENT_IN_HORIZON
        if probability >= self.config.critical_probability:
            return EventTimeStatus.CRITICAL
        if time_minutes <= self.config.imminent_horizon_minutes:
            return EventTimeStatus.IMMINENT
        if time_minutes <= self.config.near_term_horizon_minutes:
            return EventTimeStatus.NEAR_TERM
        if (
            probability >= self.config.high_risk_probability
            or time_minutes <= self.config.warning_horizon_minutes
        ):
            return EventTimeStatus.UPCOMING
        return EventTimeStatus.DISTANT

    def first_probability_crossing(
        self,
        result: SurvivalResult,
        probability: float,
    ) -> TimeToEventEstimate:
        """Estimate the first t for which F(t) >= probability."""
        self._validate_result(result)
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1.")

        if probability == 0.0:
            return TimeToEventEstimate(
                event_name=result.event_name,
                probability_level=0.0,
                time_minutes=result.times[0],
                status=EventTimeStatus.DISTANT,
                reached_within_horizon=True,
            )

        for i, p in enumerate(result.event_probability):
            if p >= probability:
                if i == 0:
                    time = result.times[0]
                    interpolated = False
                else:
                    time, interpolated = self._interpolate_crossing(
                        result.times[i - 1],
                        result.event_probability[i - 1],
                        result.times[i],
                        p,
                        probability,
                    )
                return TimeToEventEstimate(
                    event_name=result.event_name,
                    probability_level=probability,
                    time_minutes=time,
                    status=self.classify_time(time, probability),
                    interpolated=interpolated,
                    reached_within_horizon=True,
                )

        return TimeToEventEstimate(
            event_name=result.event_name,
            probability_level=probability,
            time_minutes=None,
            status=EventTimeStatus.NO_EVENT_IN_HORIZON,
            reached_within_horizon=False,
            metadata={"final_event_probability": result.final_event_probability},
        )

    def expected_time(self, result: SurvivalResult) -> Optional[float]:
        """Finite-horizon estimate of E[T] = integral S(t) dt."""
        self._validate_result(result)
        if len(result.times) < 2:
            return None

        total = 0.0
        for i in range(1, len(result.times)):
            dt = result.times[i] - result.times[i - 1]
            total += 0.5 * (
                result.survival_probability[i]
                + result.survival_probability[i - 1]
            ) * dt
        return total

    def estimate_all_quantiles(
        self,
        result: SurvivalResult,
        quantiles: Optional[Iterable[float]] = None,
    ) -> Dict[float, Optional[float]]:
        requested = (
            self.config.quantiles
            if quantiles is None
            else tuple(float(q) for q in quantiles)
        )
        return {
            q: self.first_probability_crossing(result, q).time_minutes
            for q in requested
        }

    def analyze(
        self,
        result: SurvivalResult,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TimeToEventResult:
        self._validate_result(result)

        quantile_times = self.estimate_all_quantiles(result)

        warning = self.first_probability_crossing(
            result, self.config.warning_probability
        )
        high_risk = self.first_probability_crossing(
            result, self.config.high_risk_probability
        )
        critical = self.first_probability_crossing(
            result, self.config.critical_probability
        )
        median = self.first_probability_crossing(
            result, self.config.median_probability
        )

        if critical.time_minutes is not None:
            urgency = EventTimeStatus.CRITICAL
        elif high_risk.time_minutes is not None:
            urgency = self.classify_time(
                high_risk.time_minutes,
                self.config.high_risk_probability,
            )
        elif warning.time_minutes is not None:
            urgency = self.classify_time(
                warning.time_minutes,
                self.config.warning_probability,
            )
        elif result.final_event_probability > 0:
            urgency = EventTimeStatus.DISTANT
        else:
            urgency = EventTimeStatus.NO_EVENT_IN_HORIZON

        return TimeToEventResult(
            event_name=result.event_name,
            horizon_minutes=float(result.times[-1]),
            current_event_probability=float(result.event_probability[0]),
            final_event_probability=float(result.final_event_probability),
            median_time_minutes=median.time_minutes,
            expected_time_minutes=self.expected_time(result),
            quantile_times=quantile_times,
            warning_time_minutes=warning.time_minutes,
            high_risk_time_minutes=high_risk.time_minutes,
            critical_time_minutes=critical.time_minutes,
            urgency=urgency,
            reached_warning=warning.time_minutes is not None,
            reached_high_risk=high_risk.time_minutes is not None,
            reached_critical=critical.time_minutes is not None,
            survival_result=result,
            metadata=dict(metadata or {}),
        )

    def analyze_hazard(
        self,
        times: Sequence[float],
        hazard: Any,
        *,
        event_name: str = "hazard_event",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TimeToEventResult:
        """Calculate a survival curve from hazard, then analyze time-to-event."""
        survival = SurvivalFunction()
        result = survival.calculate(
            times=times,
            hazard=hazard,
            event_name=event_name,
            metadata=metadata,
        )
        return self.analyze(result, metadata=metadata)


def time_to_event(
    result: SurvivalResult,
    *,
    config: Optional[TimeToEventConfig] = None,
) -> TimeToEventResult:
    return TimeToEventAnalyzer(config=config).analyze(result)


def estimate_time_to_event(
    result: SurvivalResult,
    probability: float = 0.50,
) -> Optional[float]:
    return TimeToEventAnalyzer().first_probability_crossing(
        result, probability
    ).time_minutes


__all__ = [
    "EventTimeStatus",
    "TimeToEventConfig",
    "TimeToEventEstimate",
    "TimeToEventResult",
    "TimeToEventAnalyzer",
    "time_to_event",
    "estimate_time_to_event",
]
