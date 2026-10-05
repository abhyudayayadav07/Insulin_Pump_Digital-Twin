"""
Future survival/hazard prediction component for DT2.

This module sits on top of `survival_function.py` and converts a current
DT2 hazard state into future survival and event-probability predictions.

Conceptual pipeline:

    current DT2 state
           |
           v
      hazard source
           |
           v
       h(t | x)
           |
           v
    survival_function.py
           |
           +------------------+
           |                  |
           v                  v
       S(t | x)          P(T <= t | x)
           |                  |
           +--------+---------+
                    |
                    v
              SurvivalForecast

The predictor is intentionally model-agnostic. It can consume:
    - a constant hazard,
    - a sequence of future hazards,
    - a callable future hazard,
    - a callable that uses the current state and future time.

It does not claim that the supplied hazard is clinically calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import math


# ---------------------------------------------------------------------------
# Lightweight local imports
# ---------------------------------------------------------------------------

# The module is written to work both as:
#   from .survival_function import ...
# and as a standalone file during development/testing.
try:
    from .survival_function import (
        SurvivalConfig,
        SurvivalFunction,
        SurvivalPoint,
        SurvivalResult,
        SurvivalStatus,
    )
except ImportError:
    from survival_function import (  # type: ignore
        SurvivalConfig,
        SurvivalFunction,
        SurvivalPoint,
        SurvivalResult,
        SurvivalStatus,
    )


@dataclass(frozen=True)
class SurvivalPredictorConfig:
    """Configuration for future DT2 survival prediction."""

    horizons_minutes: Tuple[float, ...] = (
        5.0,
        15.0,
        30.0,
        60.0,
    )

    integration_step_minutes: float = 1.0

    default_hazard: float = 0.01

    event_probability_warning: float = 0.30
    event_probability_high: float = 0.60
    event_probability_critical: float = 0.85

    # Optional exponential trend coefficient for the simple baseline
    # hazard extrapolator.
    trend_decay: float = 0.05

    def __post_init__(self) -> None:
        if not self.horizons_minutes:
            raise ValueError(
                "At least one prediction horizon is required."
            )

        if any(
            not math.isfinite(float(h))
            for h in self.horizons_minutes
        ):
            raise ValueError(
                "All prediction horizons must be finite."
            )

        if any(
            float(h) <= 0
            for h in self.horizons_minutes
        ):
            raise ValueError(
                "Prediction horizons must be > 0."
            )

        if any(
            current <= previous
            for previous, current in zip(
                self.horizons_minutes,
                self.horizons_minutes[1:],
            )
        ):
            raise ValueError(
                "Prediction horizons must be strictly increasing."
            )

        if self.integration_step_minutes <= 0:
            raise ValueError(
                "integration_step_minutes must be > 0."
            )

        if self.default_hazard < 0:
            raise ValueError(
                "default_hazard must be >= 0."
            )

        for name, value in (
            (
                "event_probability_warning",
                self.event_probability_warning,
            ),
            (
                "event_probability_high",
                self.event_probability_high,
            ),
            (
                "event_probability_critical",
                self.event_probability_critical,
            ),
        ):
            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

        if not (
            self.event_probability_warning
            <= self.event_probability_high
            <= self.event_probability_critical
        ):
            raise ValueError(
                "Event-probability thresholds must be ordered."
            )

        if self.trend_decay < 0:
            raise ValueError(
                "trend_decay must be >= 0."
            )


@dataclass(frozen=True)
class SurvivalForecastPoint:
    """Prediction at one future horizon."""

    horizon_minutes: float
    hazard_rate: float
    cumulative_hazard: float
    survival_probability: float
    event_probability: float
    status: SurvivalStatus
    event_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_minutes": self.horizon_minutes,
            "hazard_rate": self.hazard_rate,
            "cumulative_hazard": self.cumulative_hazard,
            "survival_probability": self.survival_probability,
            "event_probability": self.event_probability,
            "status": self.status.value,
            "event_name": self.event_name,
        }


@dataclass(frozen=True)
class SurvivalForecast:
    """Complete multi-horizon DT2 survival forecast."""

    event_name: str
    points: Tuple[SurvivalForecastPoint, ...]

    current_hazard: float
    current_survival: float
    current_event_probability: float

    most_urgent_horizon_minutes: Optional[float] = None
    first_warning_horizon_minutes: Optional[float] = None
    first_high_risk_horizon_minutes: Optional[float] = None
    first_critical_horizon_minutes: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def final_point(self) -> SurvivalForecastPoint:
        if not self.points:
            raise ValueError("SurvivalForecast contains no points.")
        return self.points[-1]

    @property
    def final_survival(self) -> float:
        return self.final_point.survival_probability

    @property
    def final_event_probability(self) -> float:
        return self.final_point.event_probability

    @property
    def maximum_event_probability(self) -> float:
        return max(
            point.event_probability
            for point in self.points
        )

    @property
    def highest_status(self) -> SurvivalStatus:
        order = {
            SurvivalStatus.UNKNOWN: 0,
            SurvivalStatus.VERY_SAFE: 1,
            SurvivalStatus.SAFE: 2,
            SurvivalStatus.MONITOR: 3,
            SurvivalStatus.WARNING: 4,
            SurvivalStatus.HIGH_RISK: 5,
            SurvivalStatus.CRITICAL: 6,
        }

        return max(
            (point.status for point in self.points),
            key=lambda status: order[status],
        )

    def point_at(
        self,
        horizon_minutes: float,
    ) -> SurvivalForecastPoint:
        return min(
            self.points,
            key=lambda point: abs(
                point.horizon_minutes - horizon_minutes
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_name": self.event_name,
            "points": [
                point.to_dict()
                for point in self.points
            ],
            "current_hazard": self.current_hazard,
            "current_survival": self.current_survival,
            "current_event_probability": (
                self.current_event_probability
            ),
            "most_urgent_horizon_minutes": (
                self.most_urgent_horizon_minutes
            ),
            "first_warning_horizon_minutes": (
                self.first_warning_horizon_minutes
            ),
            "first_high_risk_horizon_minutes": (
                self.first_high_risk_horizon_minutes
            ),
            "first_critical_horizon_minutes": (
                self.first_critical_horizon_minutes
            ),
            "metadata": dict(self.metadata),
        }

    def to_records(self) -> List[Dict[str, Any]]:
        return [
            point.to_dict()
            for point in self.points
        ]


@dataclass(frozen=True)
class MultiEventSurvivalForecast:
    """Survival forecasts for multiple hazard/event types."""

    forecasts: Dict[str, SurvivalForecast]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def event_probabilities_at(
        self,
        horizon_minutes: float,
    ) -> Dict[str, float]:
        return {
            name: forecast.point_at(
                horizon_minutes
            ).event_probability
            for name, forecast in self.forecasts.items()
        }

    def highest_risk_event(
        self,
        horizon_minutes: float,
    ) -> Tuple[Optional[str], float]:
        values = self.event_probabilities_at(
            horizon_minutes
        )

        if not values:
            return None, 0.0

        name = max(
            values,
            key=values.get,
        )

        return name, values[name]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forecasts": {
                name: forecast.to_dict()
                for name, forecast in self.forecasts.items()
            },
            "metadata": dict(self.metadata),
        }


class SurvivalPredictor:
    """
    Multi-horizon survival predictor for DT2.

    The predictor delegates survival mathematics to SurvivalFunction.

    Two modes are particularly useful:

    1. `predict_from_hazard`
       The caller already has a future hazard profile.

    2. `predict_from_state`
       A caller-supplied hazard model maps the current state and future
       time to a hazard rate.
    """

    def __init__(
        self,
        config: Optional[SurvivalPredictorConfig] = None,
        survival_config: Optional[SurvivalConfig] = None,
    ) -> None:
        self.config = config or SurvivalPredictorConfig()
        self.survival = SurvivalFunction(
            config=survival_config
        )

    def _status_from_probability(
        self,
        event_probability: float,
    ) -> SurvivalStatus:
        p = max(
            0.0,
            min(
                1.0,
                float(event_probability),
            ),
        )

        if p < self.config.event_probability_warning:
            if p < 0.05:
                return SurvivalStatus.SAFE
            return SurvivalStatus.MONITOR

        if p < self.config.event_probability_high:
            return SurvivalStatus.WARNING

        if p < self.config.event_probability_critical:
            return SurvivalStatus.HIGH_RISK

        return SurvivalStatus.CRITICAL

    def _build_forecast(
        self,
        result: SurvivalResult,
        *,
        event_name: str,
        current_hazard: float,
        current_survival: float,
        current_event_probability: float,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SurvivalForecast:
        points: List[SurvivalForecastPoint] = []

        for horizon in self.config.horizons_minutes:
            point = result.point_at(horizon)

            points.append(
                SurvivalForecastPoint(
                    horizon_minutes=point.time,
                    hazard_rate=point.hazard,
                    cumulative_hazard=point.cumulative_hazard,
                    survival_probability=(
                        point.survival_probability
                    ),
                    event_probability=point.event_probability,
                    status=self._status_from_probability(
                        point.event_probability
                    ),
                    event_name=event_name,
                )
            )

        warning_horizon = next(
            (
                point.horizon_minutes
                for point in points
                if point.event_probability
                >= self.config.event_probability_warning
            ),
            None,
        )

        high_risk_horizon = next(
            (
                point.horizon_minutes
                for point in points
                if point.event_probability
                >= self.config.event_probability_high
            ),
            None,
        )

        critical_horizon = next(
            (
                point.horizon_minutes
                for point in points
                if point.event_probability
                >= self.config.event_probability_critical
            ),
            None,
        )

        # Earliest horizon crossing a configured warning threshold is
        # the most useful urgency indicator for downstream safety logic.
        urgent_candidates = [
            horizon
            for horizon in (
                warning_horizon,
                high_risk_horizon,
                critical_horizon,
            )
            if horizon is not None
        ]

        return SurvivalForecast(
            event_name=event_name,
            points=tuple(points),
            current_hazard=float(current_hazard),
            current_survival=float(current_survival),
            current_event_probability=float(
                current_event_probability
            ),
            most_urgent_horizon_minutes=(
                min(urgent_candidates)
                if urgent_candidates
                else None
            ),
            first_warning_horizon_minutes=warning_horizon,
            first_high_risk_horizon_minutes=high_risk_horizon,
            first_critical_horizon_minutes=critical_horizon,
            metadata=dict(metadata or {}),
        )

    def _time_grid(self) -> List[float]:
        max_horizon = max(
            self.config.horizons_minutes
        )

        times = [0.0]
        current = self.config.integration_step_minutes

        while current < max_horizon:
            times.append(current)
            current += self.config.integration_step_minutes

        if times[-1] != max_horizon:
            times.append(max_horizon)

        return times

    def predict_from_hazard(
        self,
        hazard: Any,
        *,
        event_name: str = "hazard_event",
        current_hazard: Optional[float] = None,
        current_survival: float = 1.0,
        current_event_probability: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SurvivalForecast:
        """
        Predict future survival from a supplied hazard profile.

        `hazard` can be:
            scalar,
            sequence aligned with the generated time grid,
            callable h(t).
        """
        times = self._time_grid()

        result = self.survival.calculate(
            times=times,
            hazard=hazard,
            event_name=event_name,
            metadata=metadata,
        )

        if current_hazard is None:
            current_hazard = result.hazards[0]

        return self._build_forecast(
            result,
            event_name=event_name,
            current_hazard=current_hazard,
            current_survival=current_survival,
            current_event_probability=current_event_probability,
            metadata=metadata,
        )

    def predict_from_state(
        self,
        state: Mapping[str, Any],
        hazard_model: Callable[
            [Mapping[str, Any], float],
            float,
        ],
        *,
        event_name: str = "hazard_event",
        current_hazard: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SurvivalForecast:
        """
        Predict future survival from a state-conditioned hazard model.

        The model must implement:

            hazard_model(state, future_minutes) -> hazard_rate
        """
        times = self._time_grid()

        hazard_values = [
            hazard_model(
                state,
                t,
            )
            for t in times
        ]

        result = self.survival.calculate(
            times=times,
            hazard=hazard_values,
            event_name=event_name,
            metadata=metadata,
        )

        if current_hazard is None:
            current_hazard = result.hazards[0]

        return self._build_forecast(
            result,
            event_name=event_name,
            current_hazard=current_hazard,
            current_survival=1.0,
            current_event_probability=0.0,
            metadata=metadata,
        )

    def predict_multiple_events(
        self,
        event_hazards: Mapping[str, Any],
        *,
        current_hazards: Optional[
            Mapping[str, float]
        ] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MultiEventSurvivalForecast:
        """
        Generate separate survival forecasts for multiple hazards.

        This keeps cause-specific forecasts separate. A full competing-
        risks model can be added later if the project requires
        cause-specific cumulative incidence.
        """
        forecasts: Dict[str, SurvivalForecast] = {}

        for event_name, hazard in event_hazards.items():
            current_hazard = None

            if current_hazards is not None:
                current_hazard = current_hazards.get(
                    event_name
                )

            forecasts[event_name] = self.predict_from_hazard(
                hazard=hazard,
                event_name=event_name,
                current_hazard=current_hazard,
                metadata=metadata,
            )

        return MultiEventSurvivalForecast(
            forecasts=forecasts,
            metadata=dict(metadata or {}),
        )

    def baseline_exponential_hazard(
        self,
        current_hazard: float,
        trend: float = 0.0,
    ) -> Callable[[float], float]:
        """
        Build a simple research baseline for future hazard.

        Positive `trend` increases future hazard; negative trend decreases
        it. The trend is exponentially damped so the extrapolation does
        not grow without bound.

        This is a baseline extrapolator, NOT a trained survival model.
        """
        current_hazard = max(
            0.0,
            float(current_hazard),
        )
        trend = float(trend)

        def future_hazard(
            time_minutes: float,
        ) -> float:
            t = max(
                0.0,
                float(time_minutes),
            )

            if self.config.trend_decay == 0:
                adjustment = trend * t
            else:
                adjustment = (
                    trend
                    * (
                        1.0
                        - math.exp(
                            -self.config.trend_decay * t
                        )
                    )
                    / self.config.trend_decay
                )

            return max(
                0.0,
                current_hazard + adjustment,
            )

        return future_hazard


def predict_survival(
    hazard: Any,
    *,
    horizons_minutes: Sequence[float] = (
        5.0,
        15.0,
        30.0,
        60.0,
    ),
    integration_step_minutes: float = 1.0,
    event_name: str = "hazard_event",
    current_hazard: Optional[float] = None,
) -> SurvivalForecast:
    """Functional convenience wrapper."""
    config = SurvivalPredictorConfig(
        horizons_minutes=tuple(
            float(x)
            for x in horizons_minutes
        ),
        integration_step_minutes=integration_step_minutes,
    )

    predictor = SurvivalPredictor(
        config=config
    )

    return predictor.predict_from_hazard(
        hazard=hazard,
        event_name=event_name,
        current_hazard=current_hazard,
    )


__all__ = [
    "SurvivalPredictorConfig",
    "SurvivalForecastPoint",
    "SurvivalForecast",
    "MultiEventSurvivalForecast",
    "SurvivalPredictor",
    "predict_survival",
]
