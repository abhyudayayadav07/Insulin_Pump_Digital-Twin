"""Standardized output objects for the DT2 temporal prediction layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PredictionPoint:
    """One predicted state at a future horizon."""
    horizon: int
    values: np.ndarray
    target_names: Optional[Tuple[str, ...]] = None
    confidence: Optional[float] = None
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    timestamp: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        object.__setattr__(self, "values", np.asarray(self.values, dtype=float))
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.lower_bound is not None:
            object.__setattr__(self, "lower_bound", np.asarray(self.lower_bound, dtype=float))
        if self.upper_bound is not None:
            object.__setattr__(self, "upper_bound", np.asarray(self.upper_bound, dtype=float))


@dataclass
class TemporalPredictionOutput:
    """Unified short-term and long-term prediction output."""
    short_term: List[PredictionPoint] = field(default_factory=list)
    long_term: List[PredictionPoint] = field(default_factory=list)
    model_name: Optional[str] = None
    generated_at: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_short_term(self, point: PredictionPoint) -> None:
        self.short_term.append(point)
        self.short_term.sort(key=lambda p: p.horizon)

    def add_long_term(self, point: PredictionPoint) -> None:
        self.long_term.append(point)
        self.long_term.sort(key=lambda p: p.horizon)

    @property
    def all_points(self) -> List[PredictionPoint]:
        """Return all points in chronological horizon order."""
        # If both predictors produce the same horizon, short-term has priority.
        merged: Dict[int, PredictionPoint] = {p.horizon: p for p in self.long_term}
        merged.update({p.horizon: p for p in self.short_term})
        return [merged[h] for h in sorted(merged)]

    @property
    def horizons(self) -> Tuple[int, ...]:
        return tuple(p.horizon for p in self.all_points)

    def at_horizon(self, horizon: int) -> PredictionPoint:
        for point in self.all_points:
            if point.horizon == horizon:
                return point
        raise KeyError(f"No prediction available for horizon {horizon}")

    def to_dict(self) -> Dict[str, Any]:
        def encode(point: PredictionPoint) -> Dict[str, Any]:
            return {
                "horizon": point.horizon,
                "values": point.values.tolist(),
                "target_names": list(point.target_names) if point.target_names else None,
                "confidence": point.confidence,
                "lower_bound": point.lower_bound.tolist() if point.lower_bound is not None else None,
                "upper_bound": point.upper_bound.tolist() if point.upper_bound is not None else None,
                "timestamp": point.timestamp,
            }
        return {
            "short_term": [encode(p) for p in self.short_term],
            "long_term": [encode(p) for p in self.long_term],
            "model_name": self.model_name,
            "generated_at": self.generated_at,
            "metadata": dict(self.metadata),
        }


def _points_from_prediction(prediction: Any, confidence_by_horizon: Optional[Mapping[int, float]] = None):
    values = np.asarray(prediction.values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != len(prediction.horizons):
        raise ValueError("Prediction must have shape (horizon, target)")
    names = getattr(prediction, "target_names", None)
    return [
        PredictionPoint(
            horizon=int(h), values=values[i], target_names=names,
            confidence=(confidence_by_horizon or {}).get(int(h)),
            timestamp=getattr(prediction, "timestamp", None),
        )
        for i, h in enumerate(prediction.horizons)
    ]


def merge_predictions(
    short_term_prediction: Any = None,
    long_term_prediction: Any = None,
    *,
    model_name: Optional[str] = None,
    generated_at: Optional[Any] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    short_term_confidence: Optional[Mapping[int, float]] = None,
    long_term_confidence: Optional[Mapping[int, float]] = None,
) -> TemporalPredictionOutput:
    """Merge short- and long-term predictions; short-term wins on overlap."""
    output = TemporalPredictionOutput(
        model_name=model_name,
        generated_at=generated_at,
        metadata=dict(metadata or {}),
    )
    if short_term_prediction is not None:
        for p in _points_from_prediction(short_term_prediction, short_term_confidence):
            output.add_short_term(p)
    if long_term_prediction is not None:
        short_horizons = {p.horizon for p in output.short_term}
        for p in _points_from_prediction(long_term_prediction, long_term_confidence):
            if p.horizon not in short_horizons:
                output.add_long_term(p)
    return output


def extract_target_trajectory(output: TemporalPredictionOutput, target_index: int = 0):
    """Return ``(horizons, values)`` for one predicted target."""
    points = output.all_points
    if not points:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    horizons = np.asarray([p.horizon for p in points], dtype=int)
    values = np.asarray([np.asarray(p.values).reshape(-1)[target_index] for p in points])
    return horizons, values


__all__ = ["PredictionPoint", "TemporalPredictionOutput", "merge_predictions", "extract_target_trajectory"]
