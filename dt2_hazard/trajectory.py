"""
DT2 Trajectory Module
=====================

Builds and analyzes time-ordered physiological, device, hazard, and
Digital Twin trajectories for Digital Twin 2.

Purpose
-------
A hazard is usually not determined by a single observation. Its evolution
over time matters. This module provides a common trajectory representation
that can be consumed by:

    hazard_state.py
    hazard_function.py
    hazard_probability.py
    hazard_predictor.py
    risk/

The module is deliberately model-agnostic. It does not predict glucose or
hazard probability by itself. It stores trajectories and computes descriptive
trajectory features such as slope, acceleration, rolling statistics, threshold
crossings, time-above-threshold, and trend direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class TrajectoryTrend(str, Enum):
    """Qualitative direction of a trajectory."""

    STRONGLY_DECREASING = "strongly_decreasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    INCREASING = "increasing"
    STRONGLY_INCREASING = "strongly_increasing"
    UNKNOWN = "unknown"


@dataclass
class TrajectoryPoint:
    """One timestamped point in a trajectory."""

    timestamp: Any
    value: float
    predicted: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.value = float(self.value)

        if not isfinite(self.value):
            raise ValueError("Trajectory point value must be finite.")

        if self.predicted is not None:
            self.predicted = float(self.predicted)

        if self.lower_bound is not None:
            self.lower_bound = float(self.lower_bound)

        if self.upper_bound is not None:
            self.upper_bound = float(self.upper_bound)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "predicted": self.predicted,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "label": self.label,
            "metadata": dict(self.metadata),
        }


@dataclass
class TrajectoryFeatures:
    """Descriptive features extracted from a trajectory."""

    count: int
    duration: float
    start_value: Optional[float]
    end_value: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]
    mean: Optional[float]
    standard_deviation: Optional[float]
    net_change: Optional[float]
    mean_rate: Optional[float]
    latest_rate: Optional[float]
    acceleration: Optional[float]
    trend: TrajectoryTrend
    threshold_crossings: Dict[str, int] = field(default_factory=dict)
    time_above_threshold: Dict[str, float] = field(default_factory=dict)
    time_below_threshold: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        data = {
            "count": self.count,
            "duration": self.duration,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "net_change": self.net_change,
            "mean_rate": self.mean_rate,
            "latest_rate": self.latest_rate,
            "acceleration": self.acceleration,
            "trend": self.trend.value,
            "threshold_crossings": dict(self.threshold_crossings),
            "time_above_threshold": dict(self.time_above_threshold),
            "time_below_threshold": dict(self.time_below_threshold),
            "metadata": dict(self.metadata),
        }
        return data


def _timestamp_to_float(timestamp: Any) -> Optional[float]:
    """
    Convert a timestamp to numeric time.

    Numeric timestamps are assumed to already use the desired time unit.
    ISO timestamps are converted to Unix seconds.
    """
    if isinstance(timestamp, (int, float)):
        value = float(timestamp)
        return value if isfinite(value) else None

    try:
        from datetime import datetime

        if isinstance(timestamp, datetime):
            return timestamp.timestamp()

        if isinstance(timestamp, str):
            parsed = datetime.fromisoformat(
                timestamp.strip().replace("Z", "+00:00")
            )
            return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return None

    return None


def _validate_thresholds(
    thresholds: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    """Validate and normalize threshold definitions."""
    if thresholds is None:
        return {}

    result: Dict[str, float] = {}

    for name, value in thresholds.items():
        value = float(value)

        if not isfinite(value):
            raise ValueError(
                f"Threshold '{name}' must be finite."
            )

        result[str(name)] = value

    return result


def _calculate_rates(
    times: Sequence[float],
    values: Sequence[float],
) -> List[float]:
    """Calculate point-to-point rates of change."""
    rates: List[float] = []

    for index in range(1, len(values)):
        delta_time = times[index] - times[index - 1]

        if abs(delta_time) <= 1e-12:
            rates.append(0.0)
        else:
            rates.append(
                (values[index] - values[index - 1])
                / delta_time
            )

    return rates


def _classify_trend(
    rate: Optional[float],
    *,
    stable_threshold: float = 0.001,
    strong_threshold: float = 0.01,
) -> TrajectoryTrend:
    """Convert a normalized rate into a qualitative trend."""
    if rate is None or not isfinite(rate):
        return TrajectoryTrend.UNKNOWN

    if abs(rate) <= stable_threshold:
        return TrajectoryTrend.STABLE

    if rate >= strong_threshold:
        return TrajectoryTrend.STRONGLY_INCREASING

    if rate > stable_threshold:
        return TrajectoryTrend.INCREASING

    if rate <= -strong_threshold:
        return TrajectoryTrend.STRONGLY_DECREASING

    return TrajectoryTrend.DECREASING


class Trajectory:
    """
    Time-ordered trajectory container.

    A trajectory can represent:
    - glucose
    - CGM
    - insulin delivery
    - DT1 residual
    - hazard intensity
    - hazard probability
    - attack score
    - any other scalar DT2 signal
    """

    def __init__(
        self,
        name: str,
        *,
        unit: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        max_length: Optional[int] = None,
    ) -> None:
        if not name.strip():
            raise ValueError("Trajectory name must not be empty.")

        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be > 0.")

        self.name = name
        self.unit = unit
        self.metadata = dict(metadata or {})
        self.max_length = max_length
        self._points: List[TrajectoryPoint] = []

    def add_point(
        self,
        timestamp: Any,
        value: float,
        *,
        predicted: Optional[float] = None,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        label: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TrajectoryPoint:
        """Append one point to the trajectory."""
        point = TrajectoryPoint(
            timestamp=timestamp,
            value=value,
            predicted=predicted,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            label=label,
            metadata=dict(metadata or {}),
        )

        self._points.append(point)

        if self.max_length is not None:
            self._points = self._points[-self.max_length:]

        return point

    def extend(
        self,
        points: Iterable[TrajectoryPoint],
    ) -> None:
        """Append multiple trajectory points."""
        for point in points:
            self._points.append(point)

        if self.max_length is not None:
            self._points = self._points[-self.max_length:]

    def points(self) -> List[TrajectoryPoint]:
        """Return a copy of the trajectory points."""
        return list(self._points)

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self):
        return iter(self._points)

    def latest(self) -> Optional[TrajectoryPoint]:
        """Return the latest point."""
        return self._points[-1] if self._points else None

    def first(self) -> Optional[TrajectoryPoint]:
        """Return the first point."""
        return self._points[0] if self._points else None

    def values(self) -> List[float]:
        """Return trajectory values."""
        return [point.value for point in self._points]

    def timestamps(self) -> List[Any]:
        """Return trajectory timestamps."""
        return [point.timestamp for point in self._points]

    def numeric_times(self) -> List[float]:
        """Return usable numeric timestamps."""
        return [
            timestamp
            for timestamp in (
                _timestamp_to_float(point.timestamp)
                for point in self._points
            )
            if timestamp is not None
        ]

    def sorted_points(self) -> List[TrajectoryPoint]:
        """Return points sorted by timestamp when timestamps are comparable."""
        usable = []

        for index, point in enumerate(self._points):
            numeric_time = _timestamp_to_float(point.timestamp)

            if numeric_time is not None:
                usable.append((numeric_time, index, point))

        usable.sort(key=lambda item: item[0])

        return [item[2] for item in usable]

    def window(self, count: int) -> "Trajectory":
        """Return a trajectory containing the latest ``count`` points."""
        if count <= 0:
            raise ValueError("count must be > 0.")

        result = Trajectory(
            self.name,
            unit=self.unit,
            metadata=self.metadata,
            max_length=count,
        )

        result.extend(self._points[-count:])
        return result

    def time_window(
        self,
        start_time: Any,
        end_time: Any,
    ) -> "Trajectory":
        """Return points within an inclusive time window."""
        start = _timestamp_to_float(start_time)
        end = _timestamp_to_float(end_time)

        if start is None or end is None:
            raise ValueError(
                "start_time and end_time must be convertible to numeric time."
            )

        if start > end:
            raise ValueError("start_time must be <= end_time.")

        result = Trajectory(
            self.name,
            unit=self.unit,
            metadata=self.metadata,
        )

        for point in self._points:
            timestamp = _timestamp_to_float(point.timestamp)

            if timestamp is not None and start <= timestamp <= end:
                result.add_point(
                    point.timestamp,
                    point.value,
                    predicted=point.predicted,
                    lower_bound=point.lower_bound,
                    upper_bound=point.upper_bound,
                    label=point.label,
                    metadata=point.metadata,
                )

        return result

    def rates(self) -> List[float]:
        """Return point-to-point rates of change."""
        points = self.sorted_points()

        times = [
            _timestamp_to_float(point.timestamp)
            for point in points
        ]

        usable = [
            (time, point.value)
            for time, point in zip(times, points)
            if time is not None
        ]

        if len(usable) < 2:
            return []

        numeric_times = [item[0] for item in usable]
        values = [item[1] for item in usable]

        return _calculate_rates(numeric_times, values)

    def latest_rate(self) -> Optional[float]:
        """Return the latest rate of change."""
        rates = self.rates()
        return rates[-1] if rates else None

    def acceleration(self) -> Optional[float]:
        """Return the latest change in rate."""
        rates = self.rates()

        if len(rates) < 2:
            return None

        return rates[-1] - rates[-2]

    def slope(self) -> Optional[float]:
        """Estimate the overall trajectory slope."""
        points = self.sorted_points()

        usable = []

        for point in points:
            timestamp = _timestamp_to_float(point.timestamp)

            if timestamp is not None:
                usable.append((timestamp, point.value))

        if len(usable) < 2:
            return None

        times = [item[0] for item in usable]
        values = [item[1] for item in usable]

        mean_time = mean(times)
        mean_value = mean(values)

        numerator = sum(
            (time - mean_time) * (value - mean_value)
            for time, value in zip(times, values)
        )

        denominator = sum(
            (time - mean_time) ** 2
            for time in times
        )

        if denominator <= 1e-12:
            return None

        return numerator / denominator

    def trend(
        self,
        *,
        stable_threshold: float = 0.001,
        strong_threshold: float = 0.01,
    ) -> TrajectoryTrend:
        """Classify the latest trajectory trend."""
        rate = self.latest_rate()

        return _classify_trend(
            rate,
            stable_threshold=stable_threshold,
            strong_threshold=strong_threshold,
        )

    def rolling_mean(self, window: int) -> Optional[float]:
        """Return the mean of the latest ``window`` values."""
        if window <= 0:
            raise ValueError("window must be > 0.")

        values = self.values()

        if not values:
            return None

        return mean(values[-window:])

    def rolling_std(self, window: int) -> Optional[float]:
        """Return population standard deviation of latest values."""
        if window <= 0:
            raise ValueError("window must be > 0.")

        values = self.values()

        if not values:
            return None

        selected = values[-window:]

        if len(selected) == 1:
            return 0.0

        return pstdev(selected)

    def count_threshold_crossings(
        self,
        threshold: float,
        *,
        direction: str = "up",
    ) -> int:
        """
        Count crossings of a threshold.

        direction:
            ``up``   : below/equal -> above
            ``down`` : above/equal -> below
            ``both`` : either direction
        """
        threshold = float(threshold)

        if direction not in {"up", "down", "both"}:
            raise ValueError(
                "direction must be 'up', 'down', or 'both'."
            )

        values = self.values()

        if len(values) < 2:
            return 0

        count = 0

        for previous, current in zip(values[:-1], values[1:]):
            crossed_up = (
                previous <= threshold
                and current > threshold
            )

            crossed_down = (
                previous >= threshold
                and current < threshold
            )

            if direction == "up" and crossed_up:
                count += 1
            elif direction == "down" and crossed_down:
                count += 1
            elif direction == "both" and (crossed_up or crossed_down):
                count += 1

        return count

    def time_above(self, threshold: float) -> float:
        """Calculate approximate time spent above a threshold."""
        return self._time_condition(
            threshold=float(threshold),
            condition="above",
        )

    def time_below(self, threshold: float) -> float:
        """Calculate approximate time spent below a threshold."""
        return self._time_condition(
            threshold=float(threshold),
            condition="below",
        )

    def _time_condition(
        self,
        *,
        threshold: float,
        condition: str,
    ) -> float:
        points = self.sorted_points()

        if len(points) < 2:
            return 0.0

        total = 0.0

        for previous, current in zip(points[:-1], points[1:]):
            previous_time = _timestamp_to_float(previous.timestamp)
            current_time = _timestamp_to_float(current.timestamp)

            if previous_time is None or current_time is None:
                continue

            duration = max(0.0, current_time - previous_time)

            if condition == "above":
                if previous.value > threshold:
                    total += duration
            elif condition == "below":
                if previous.value < threshold:
                    total += duration
            else:
                raise ValueError(
                    "condition must be 'above' or 'below'."
                )

        return total

    def features(
        self,
        *,
        thresholds: Optional[Mapping[str, float]] = None,
        stable_threshold: float = 0.001,
        strong_threshold: float = 0.01,
    ) -> TrajectoryFeatures:
        """Extract descriptive features from the trajectory."""
        points = self.sorted_points()

        if not points:
            return TrajectoryFeatures(
                count=0,
                duration=0.0,
                start_value=None,
                end_value=None,
                minimum=None,
                maximum=None,
                mean=None,
                standard_deviation=None,
                net_change=None,
                mean_rate=None,
                latest_rate=None,
                acceleration=None,
                trend=TrajectoryTrend.UNKNOWN,
            )

        values = [point.value for point in points]
        numeric_times = [
            _timestamp_to_float(point.timestamp)
            for point in points
        ]

        usable_times = [
            value
            for value in numeric_times
            if value is not None
        ]

        if len(usable_times) >= 2:
            duration = max(
                0.0,
                usable_times[-1] - usable_times[0],
            )
        else:
            duration = 0.0

        rates = self.rates()

        mean_rate = mean(rates) if rates else None
        latest_rate = rates[-1] if rates else None
        acceleration = self.acceleration()

        threshold_values = _validate_thresholds(thresholds)

        crossings: Dict[str, int] = {}
        above: Dict[str, float] = {}
        below: Dict[str, float] = {}

        for name, threshold in threshold_values.items():
            crossings[name] = self.count_threshold_crossings(
                threshold,
                direction="both",
            )
            above[name] = self.time_above(threshold)
            below[name] = self.time_below(threshold)

        return TrajectoryFeatures(
            count=len(points),
            duration=duration,
            start_value=values[0],
            end_value=values[-1],
            minimum=min(values),
            maximum=max(values),
            mean=mean(values),
            standard_deviation=(
                pstdev(values)
                if len(values) > 1
                else 0.0
            ),
            net_change=values[-1] - values[0],
            mean_rate=mean_rate,
            latest_rate=latest_rate,
            acceleration=acceleration,
            trend=_classify_trend(
                latest_rate,
                stable_threshold=stable_threshold,
                strong_threshold=strong_threshold,
            ),
            threshold_crossings=crossings,
            time_above_threshold=above,
            time_below_threshold=below,
            metadata={
                "name": self.name,
                "unit": self.unit,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the complete trajectory."""
        return {
            "name": self.name,
            "unit": self.unit,
            "metadata": dict(self.metadata),
            "max_length": self.max_length,
            "points": [
                point.to_dict()
                for point in self._points
            ],
        }


def trajectory_from_observations(
    name: str,
    observations: Iterable[Mapping[str, Any]],
    *,
    unit: Optional[str] = None,
    max_length: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Trajectory:
    """
    Construct a trajectory from dictionaries.

    Required fields:
        timestamp
        value

    Optional fields:
        predicted
        lower_bound
        upper_bound
        label
        metadata
    """
    trajectory = Trajectory(
        name,
        unit=unit,
        max_length=max_length,
        metadata=metadata,
    )

    for observation in observations:
        trajectory.add_point(
            observation["timestamp"],
            float(observation["value"]),
            predicted=observation.get("predicted"),
            lower_bound=observation.get("lower_bound"),
            upper_bound=observation.get("upper_bound"),
            label=observation.get("label"),
            metadata=observation.get("metadata"),
        )

    return trajectory


def build_hazard_trajectory(
    hazard_id: str,
    observations: Iterable[Mapping[str, Any]],
    *,
    max_length: Optional[int] = None,
) -> Trajectory:
    """
    Convenience constructor for a DT2 hazard trajectory.

    The expected observation fields are:
        timestamp
        probability

    Optional:
        intensity
        evidence_score
    """
    trajectory = Trajectory(
        name=f"hazard:{hazard_id}",
        unit="normalized",
        max_length=max_length,
        metadata={"hazard_id": hazard_id},
    )

    for observation in observations:
        probability = float(observation["probability"])

        trajectory.add_point(
            observation["timestamp"],
            probability,
            predicted=(
                float(observation["intensity"])
                if observation.get("intensity") is not None
                else None
            ),
            label=observation.get("state"),
            metadata={
                "intensity": observation.get("intensity"),
                "evidence_score": observation.get("evidence_score"),
                **dict(observation.get("metadata") or {}),
            },
        )

    return trajectory


__all__ = [
    "TrajectoryTrend",
    "TrajectoryPoint",
    "TrajectoryFeatures",
    "Trajectory",
    "trajectory_from_observations",
    "build_hazard_trajectory",
]
