"""
DT2 Hazard Predictor Module
===========================

Predicts the future evolution of hazard intensity/probability from a
time-ordered history of DT2 hazard observations.

This module sits after hazard_function.py and hazard_probability.py:

    hazard_definition.py
            |
    hazard_state.py
            |
    hazard_function.py
            |
    hazard_probability.py
            |
    hazard_predictor.py
            |
          risk/

The predictor is deliberately separate from DT1:
- DT1 predicts physiological quantities such as future glucose.
- DT2 predicts whether a defined safety hazard is likely to become active.

The default predictor is a lightweight, interpretable trajectory model based
on recent hazard-probability trends. It is intended as a baseline that can
later be replaced by a trained ML/time-series model without changing the
rest of the DT2 architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class PredictionMethod(str, Enum):
    """Available hazard prediction methods."""

    TREND = "trend"
    EWMA_TREND = "ewma_trend"
    LINEAR = "linear"


class PredictedHazardState(str, Enum):
    """Qualitative predicted hazard state."""

    LOW = "low"
    MONITORING = "monitoring"
    WARNING = "warning"
    ACTIVE = "active"
    CRITICAL = "critical"


@dataclass
class HazardPredictorConfig:
    """
    Configuration for future hazard prediction.

    Parameters
    ----------
    horizons:
        Future horizons expressed in the same time unit as the supplied
        timestamps. For the insulin-pump simulation this can normally be
        minutes.
    history_length:
        Maximum number of recent observations used by the predictor.
    minimum_history:
        Minimum number of observations required for trend estimation.
    method:
        Trend-estimation method.
    smoothing_alpha:
        EWMA smoothing parameter.
    warning_threshold:
        Predicted probability/intensity above this value enters warning.
    active_threshold:
        Predicted probability/intensity above this value enters active.
    critical_threshold:
        Predicted probability/intensity above this value enters critical.
    """

    horizons: Tuple[float, ...] = (5.0, 15.0, 30.0)
    history_length: int = 20
    minimum_history: int = 3
    method: PredictionMethod = PredictionMethod.EWMA_TREND
    smoothing_alpha: float = 0.40

    warning_threshold: float = 0.30
    active_threshold: float = 0.60
    critical_threshold: float = 0.85

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("At least one prediction horizon is required.")

        if any(float(horizon) <= 0 for horizon in self.horizons):
            raise ValueError("All prediction horizons must be > 0.")

        if self.history_length < 2:
            raise ValueError("history_length must be at least 2.")

        if self.minimum_history < 2:
            raise ValueError("minimum_history must be at least 2.")

        if self.minimum_history > self.history_length:
            raise ValueError(
                "minimum_history cannot exceed history_length."
            )

        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1].")

        if not (
            0.0
            <= self.warning_threshold
            <= self.active_threshold
            <= self.critical_threshold
            <= 1.0
        ):
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= warning <= active <= critical <= 1."
            )


@dataclass
class HazardPrediction:
    """Prediction for one future horizon."""

    hazard_id: str
    horizon: float
    predicted_probability: float
    predicted_intensity: float
    predicted_state: PredictedHazardState
    trend: float
    confidence: float
    method: str
    history_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def predicted_probability_percent(self) -> float:
        """Probability expressed as percentage."""
        return 100.0 * self.predicted_probability

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        return {
            "hazard_id": self.hazard_id,
            "horizon": self.horizon,
            "predicted_probability": self.predicted_probability,
            "predicted_probability_percent":
                self.predicted_probability_percent,
            "predicted_intensity": self.predicted_intensity,
            "predicted_state": self.predicted_state.value,
            "trend": self.trend,
            "confidence": self.confidence,
            "method": self.method,
            "history_count": self.history_count,
            "metadata": dict(self.metadata),
        }


@dataclass
class HazardPredictionResult:
    """Collection of predictions for one hazard."""

    hazard_id: str
    predictions: List[HazardPrediction]
    current_probability: float
    current_intensity: float
    trend: float
    confidence: float
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def earliest_active_horizon(self) -> Optional[float]:
        """Return the earliest horizon predicted to be active/critical."""
        for prediction in self.predictions:
            if prediction.predicted_state in {
                PredictedHazardState.ACTIVE,
                PredictedHazardState.CRITICAL,
            }:
                return prediction.horizon
        return None

    def earliest_critical_horizon(self) -> Optional[float]:
        """Return the earliest horizon predicted to be critical."""
        for prediction in self.predictions:
            if prediction.predicted_state == PredictedHazardState.CRITICAL:
                return prediction.horizon
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        return {
            "hazard_id": self.hazard_id,
            "predictions": [
                prediction.to_dict()
                for prediction in self.predictions
            ],
            "current_probability": self.current_probability,
            "current_intensity": self.current_intensity,
            "trend": self.trend,
            "confidence": self.confidence,
            "method": self.method,
            "metadata": dict(self.metadata),
        }


@dataclass
class HazardObservationPoint:
    """
    One historical hazard observation.

    ``probability`` should normally come from hazard_probability.py and
    ``intensity`` should normally come from hazard_function.py.
    """

    timestamp: Any
    probability: float
    intensity: Optional[float] = None
    evidence_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _clamp(value: float) -> float:
    """Clamp a scalar to [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def _timestamp_to_float(timestamp: Any) -> Optional[float]:
    """
    Convert common timestamp representations to a numeric time.

    Numeric timestamps are assumed to already be in the desired unit.
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


def _linear_trend(
    values: Sequence[float],
    times: Sequence[float],
) -> float:
    """Estimate slope using ordinary least squares."""
    if len(values) != len(times):
        raise ValueError("values and times must have equal lengths.")

    if len(values) < 2:
        return 0.0

    mean_t = mean(times)
    mean_y = mean(values)

    numerator = sum(
        (time - mean_t) * (value - mean_y)
        for time, value in zip(times, values)
    )

    denominator = sum(
        (time - mean_t) ** 2
        for time in times
    )

    if denominator <= 1e-12:
        return 0.0

    return numerator / denominator


def _ewma(values: Sequence[float], alpha: float) -> float:
    """Calculate exponentially weighted moving average."""
    if not values:
        return 0.0

    result = float(values[0])

    for value in values[1:]:
        result = alpha * float(value) + (1.0 - alpha) * result

    return result


def _classify(
    value: float,
    config: HazardPredictorConfig,
) -> PredictedHazardState:
    """Classify a predicted probability/intensity."""
    value = _clamp(value)

    if value >= config.critical_threshold:
        return PredictedHazardState.CRITICAL

    if value >= config.active_threshold:
        return PredictedHazardState.ACTIVE

    if value >= config.warning_threshold:
        return PredictedHazardState.WARNING

    if value > 0.0:
        return PredictedHazardState.MONITORING

    return PredictedHazardState.LOW


def _confidence(
    history_count: int,
    minimum_history: int,
    trend: float,
) -> float:
    """
    Estimate confidence from history availability and trend stability.

    This is a heuristic confidence indicator, not a statistical confidence
    interval.
    """
    history_component = min(
        1.0,
        history_count / max(minimum_history * 2, 1),
    )

    trend_penalty = min(1.0, abs(trend) * 10.0)
    stability_component = 1.0 - 0.5 * trend_penalty

    return _clamp(
        history_component * stability_component
    )


class HazardPredictor:
    """
    Predict future hazard probability/intensity from recent DT2 history.

    The baseline model is intentionally transparent. It estimates a recent
    trend and extrapolates it over the requested horizons.
    """

    def __init__(
        self,
        config: Optional[HazardPredictorConfig] = None,
    ) -> None:
        self.config = config or HazardPredictorConfig()
        self._history: Dict[str, List[HazardObservationPoint]] = {}

    def add_observation(
        self,
        hazard_id: str,
        *,
        timestamp: Any,
        probability: float,
        intensity: Optional[float] = None,
        evidence_score: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Append a hazard observation to the predictor history."""
        probability = _clamp(probability)

        if intensity is None:
            intensity = probability
        else:
            intensity = _clamp(intensity)

        if evidence_score is not None:
            evidence_score = _clamp(evidence_score)

        point = HazardObservationPoint(
            timestamp=timestamp,
            probability=probability,
            intensity=intensity,
            evidence_score=evidence_score,
            metadata=dict(metadata or {}),
        )

        history = self._history.setdefault(hazard_id, [])
        history.append(point)

        if len(history) > self.config.history_length:
            del history[:-self.config.history_length]

    def add_probability_result(
        self,
        result: Any,
        *,
        timestamp: Any,
        intensity: Optional[float] = None,
    ) -> None:
        """
        Add a HazardProbabilityResult without requiring a hard import.

        This keeps hazard_predictor.py compatible with the independently
        generated hazard_probability.py module.
        """
        self.add_observation(
            hazard_id=str(result.hazard_id),
            timestamp=timestamp,
            probability=float(result.probability),
            intensity=intensity,
            evidence_score=float(result.evidence_score),
        )

    def history(
        self,
        hazard_id: str,
    ) -> List[HazardObservationPoint]:
        """Return a copy of the history for one hazard."""
        return list(self._history.get(hazard_id, []))

    def _prepare_history(
        self,
        hazard_id: str,
    ) -> Tuple[List[float], List[float]]:
        """Return usable times and probabilities."""
        points = self._history.get(hazard_id, [])

        usable: List[Tuple[float, float]] = []

        for point in points:
            timestamp = _timestamp_to_float(point.timestamp)
            if timestamp is None:
                continue
            usable.append((timestamp, point.probability))

        if len(usable) < 2:
            return [], []

        usable.sort(key=lambda item: item[0])

        times = [item[0] for item in usable]
        values = [item[1] for item in usable]

        return times, values

    def _estimate_trend(
        self,
        times: Sequence[float],
        values: Sequence[float],
    ) -> float:
        """Estimate recent probability trend."""
        if len(values) < 2:
            return 0.0

        if self.config.method == PredictionMethod.LINEAR:
            return _linear_trend(values, times)

        if self.config.method == PredictionMethod.TREND:
            dt = times[-1] - times[-2]
            if abs(dt) <= 1e-12:
                return 0.0
            return (values[-1] - values[-2]) / dt

        # EWMA trend
        smoothed = _ewma(
            values,
            self.config.smoothing_alpha,
        )

        if len(values) < 2:
            return 0.0

        dt = times[-1] - times[-2]

        if abs(dt) <= 1e-12:
            return 0.0

        return (values[-1] - smoothed) / dt

    def predict(
        self,
        hazard_id: str,
        *,
        current_probability: Optional[float] = None,
        current_intensity: Optional[float] = None,
        horizons: Optional[Iterable[float]] = None,
    ) -> HazardPredictionResult:
        """
        Predict future hazard probability and intensity.

        If insufficient history is available, the current value is carried
        forward with low confidence rather than inventing a trend.
        """
        points = self._history.get(hazard_id, [])

        if current_probability is None:
            if not points:
                raise ValueError(
                    f"No history exists for hazard '{hazard_id}'."
                )
            current_probability = points[-1].probability

        current_probability = _clamp(current_probability)

        if current_intensity is None:
            if points and points[-1].intensity is not None:
                current_intensity = points[-1].intensity
            else:
                current_intensity = current_probability

        current_intensity = _clamp(current_intensity)

        selected_horizons = tuple(
            float(horizon)
            for horizon in (
                self.config.horizons
                if horizons is None
                else horizons
            )
        )

        if not selected_horizons:
            raise ValueError("At least one prediction horizon is required.")

        if any(horizon <= 0 for horizon in selected_horizons):
            raise ValueError("Prediction horizons must be > 0.")

        times, values = self._prepare_history(hazard_id)

        sufficient_history = (
            len(values) >= self.config.minimum_history
        )

        if sufficient_history:
            trend = self._estimate_trend(times, values)
        else:
            trend = 0.0

        confidence = _confidence(
            len(values),
            self.config.minimum_history,
            trend,
        )

        predictions: List[HazardPrediction] = []

        # Numeric timestamps may be seconds or minutes depending on caller.
        # To avoid imposing a unit conversion, the horizon is interpreted in
        # the same unit as the observation timestamps.
        for horizon in selected_horizons:
            if sufficient_history:
                predicted_probability = _clamp(
                    current_probability + trend * horizon
                )

                # Probability trend and intensity trend are intentionally
                # kept conservative: the same normalized trajectory is used
                # for both unless a future learned model replaces it.
                predicted_intensity = _clamp(
                    current_intensity + trend * horizon
                )
            else:
                predicted_probability = current_probability
                predicted_intensity = current_intensity

            state = _classify(
                predicted_probability,
                self.config,
            )

            predictions.append(
                HazardPrediction(
                    hazard_id=hazard_id,
                    horizon=horizon,
                    predicted_probability=predicted_probability,
                    predicted_intensity=predicted_intensity,
                    predicted_state=state,
                    trend=trend,
                    confidence=confidence,
                    method=self.config.method.value,
                    history_count=len(values),
                    metadata={
                        "sufficient_history": sufficient_history,
                        "history_length": len(values),
                    },
                )
            )

        return HazardPredictionResult(
            hazard_id=hazard_id,
            predictions=predictions,
            current_probability=current_probability,
            current_intensity=current_intensity,
            trend=trend,
            confidence=confidence,
            method=self.config.method.value,
            metadata={
                "sufficient_history": sufficient_history,
                "history_length": len(values),
            },
        )

    def predict_from_probability_result(
        self,
        result: Any,
        *,
        timestamp: Any,
        intensity: Optional[float] = None,
        horizons: Optional[Iterable[float]] = None,
    ) -> HazardPredictionResult:
        """Add the current probability result and predict forward."""
        self.add_probability_result(
            result,
            timestamp=timestamp,
            intensity=intensity,
        )

        return self.predict(
            str(result.hazard_id),
            current_probability=float(result.probability),
            current_intensity=intensity,
            horizons=horizons,
        )

    def reset(self, hazard_id: Optional[str] = None) -> None:
        """Reset one hazard history or all histories."""
        if hazard_id is None:
            self._history.clear()
        else:
            self._history.pop(hazard_id, None)


def predict_hazard_trajectory(
    hazard_id: str,
    observations: Sequence[Mapping[str, Any]],
    *,
    horizons: Sequence[float] = (5.0, 15.0, 30.0),
    method: PredictionMethod = PredictionMethod.EWMA_TREND,
) -> HazardPredictionResult:
    """
    Stateless convenience function.

    Each observation should contain:
        timestamp
        probability

    Optional:
        intensity
        evidence_score
        metadata
    """
    predictor = HazardPredictor(
        HazardPredictorConfig(
            horizons=tuple(horizons),
            method=method,
        )
    )

    for observation in observations:
        predictor.add_observation(
            hazard_id,
            timestamp=observation["timestamp"],
            probability=float(observation["probability"]),
            intensity=(
                float(observation["intensity"])
                if observation.get("intensity") is not None
                else None
            ),
            evidence_score=(
                float(observation["evidence_score"])
                if observation.get("evidence_score") is not None
                else None
            ),
            metadata=observation.get("metadata"),
        )

    return predictor.predict(hazard_id)


__all__ = [
    "PredictionMethod",
    "PredictedHazardState",
    "HazardPredictorConfig",
    "HazardPrediction",
    "HazardPredictionResult",
    "HazardObservationPoint",
    "HazardPredictor",
    "predict_hazard_trajectory",
]
