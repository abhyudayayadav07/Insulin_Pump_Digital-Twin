"""Model-agnostic temporal prediction interface for Digital Twin 2.

The temporal prediction layer converts a recent state sequence into one or
more future-state predictions. It produces evidence for reachability,
hazard/survival, and mitigation layers; it does not directly control the
insulin pump.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


@dataclass(frozen=True)
class TemporalPredictorConfig:
    """Shared configuration for temporal predictors."""

    history_length: int = 12
    horizons: Tuple[int, ...] = (1,)
    target_names: Optional[Tuple[str, ...]] = None
    feature_names: Optional[Tuple[str, ...]] = None
    clip_min: Optional[float] = 20.0
    clip_max: Optional[float] = 600.0
    require_finite_input: bool = True

    def __post_init__(self) -> None:
        if self.history_length <= 0:
            raise ValueError("history_length must be positive")
        if not self.horizons:
            raise ValueError("At least one horizon is required")
        horizons = tuple(sorted(set(int(h) for h in self.horizons)))
        if any(h <= 0 for h in horizons):
            raise ValueError("All horizons must be positive")
        if self.clip_min is not None and self.clip_max is not None:
            if self.clip_min >= self.clip_max:
                raise ValueError("clip_min must be smaller than clip_max")
        object.__setattr__(self, "horizons", horizons)


@dataclass
class TemporalPrediction:
    """Prediction result for one input sequence."""

    horizons: Tuple[int, ...]
    values: np.ndarray
    target_names: Optional[Tuple[str, ...]] = None
    timestamp: Optional[Any] = None
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError("values must have shape (horizons, targets)")
        if values.shape[0] != len(self.horizons):
            raise ValueError("Prediction rows must match horizons")
        self.values = values

    def at_horizon(self, horizon: int) -> np.ndarray:
        """Return prediction values for one horizon."""
        if horizon not in self.horizons:
            raise KeyError(f"Horizon {horizon} is not available")
        return self.values[self.horizons.index(horizon)]

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)


@dataclass
class TemporalPredictionBatch:
    """Prediction result for a batch of temporal sequences."""

    horizons: Tuple[int, ...]
    values: np.ndarray
    target_names: Optional[Tuple[str, ...]] = None
    timestamps: Optional[np.ndarray] = None
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim == 2:
            values = values[:, :, None]
        if values.ndim != 3:
            raise ValueError("values must have shape (batch, horizons, targets)")
        if values.shape[1] != len(self.horizons):
            raise ValueError("Prediction horizon dimension does not match horizons")
        self.values = values
        if self.timestamps is not None:
            self.timestamps = np.asarray(self.timestamps, dtype=object)
            if len(self.timestamps) != values.shape[0]:
                raise ValueError("timestamps must match batch size")

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_targets(self) -> int:
        return int(self.values.shape[2])

    def at_horizon(self, horizon: int) -> np.ndarray:
        if horizon not in self.horizons:
            raise KeyError(f"Horizon {horizon} is not available")
        return self.values[:, self.horizons.index(horizon), :]


def _as_batch(sequences: ArrayLike, history_length: Optional[int]) -> np.ndarray:
    """Validate and convert input to (batch, history, features)."""
    array = np.asarray(sequences, dtype=float)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(
            "Input must have shape (history, features) or "
            "(batch, history, features)"
        )
    if history_length is not None and array.shape[1] != history_length:
        raise ValueError(
            f"Expected history length {history_length}, got {array.shape[1]}"
        )
    return array


def _normalize_output(output: Any, batch_size: int, n_horizons: int) -> np.ndarray:
    """Normalize common model output formats to (batch, horizons, targets)."""
    if isinstance(output, dict):
        if set(output.keys()) != set(range(n_horizons)):
            # Dictionary keys may be actual horizon values; ordering is enough.
            values = list(output.values())
        else:
            values = [output[i] for i in range(n_horizons)]
        parts = []
        for value in values:
            value = np.asarray(value, dtype=float)
            if value.ndim == 1:
                value = value[:, None]
            if value.ndim != 2:
                raise ValueError("Dictionary outputs must be 1-D or 2-D")
            parts.append(value)
        if len(parts) != n_horizons:
            raise ValueError("Number of outputs does not match horizons")
        return np.stack(parts, axis=1)

    if isinstance(output, (list, tuple)) and len(output) == n_horizons:
        parts = []
        for value in output:
            value = np.asarray(value, dtype=float)
            if value.ndim == 1:
                value = value[:, None]
            if value.ndim != 2:
                raise ValueError("List outputs must be 1-D or 2-D")
            parts.append(value)
        return np.stack(parts, axis=1)

    array = np.asarray(output, dtype=float)
    if array.ndim == 1:
        if batch_size != 1:
            raise ValueError("1-D output is valid only for one sample")
        array = array[None, :]
    if array.ndim == 2:
        if n_horizons == 1:
            array = array[:, None, :]
        elif array.shape[1] == n_horizons:
            array = array[:, :, None]
        else:
            raise ValueError("Ambiguous 2-D model output")
    elif array.ndim == 3:
        if array.shape[1] == n_horizons:
            pass
        elif array.shape[2] == n_horizons:
            array = np.transpose(array, (0, 2, 1))
        else:
            raise ValueError("Could not identify horizon dimension")
    else:
        raise ValueError("Model output must be 1-D, 2-D, or 3-D")
    if array.shape[0] != batch_size:
        raise ValueError("Model output batch size does not match input")
    return array


class TemporalPredictor(ABC):
    """Abstract interface for all DT2 temporal prediction models."""

    def __init__(
        self,
        config: Optional[TemporalPredictorConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.config = config or TemporalPredictorConfig()
        self.model_name = model_name or self.__class__.__name__

    @abstractmethod
    def _predict_model(self, sequences: np.ndarray) -> Any:
        """Return raw predictions for a sequence batch."""
        raise NotImplementedError

    def _validate_input(self, sequences: np.ndarray) -> None:
        if self.config.require_finite_input and not np.isfinite(sequences).all():
            raise ValueError("Input sequence contains non-finite values")
        if self.config.feature_names is not None:
            expected = len(self.config.feature_names)
            if sequences.shape[-1] != expected:
                raise ValueError(f"Expected {expected} features, got {sequences.shape[-1]}")

    def predict_batch(
        self,
        sequences: ArrayLike,
        *,
        timestamps: Optional[Sequence[Any]] = None,
    ) -> TemporalPredictionBatch:
        """Predict future states for a batch of sequences."""
        batch = _as_batch(sequences, self.config.history_length)
        self._validate_input(batch)
        raw = self._predict_model(batch)
        values = _normalize_output(raw, batch.shape[0], len(self.config.horizons))
        lower = -np.inf if self.config.clip_min is None else self.config.clip_min
        upper = np.inf if self.config.clip_max is None else self.config.clip_max
        values = np.clip(values, lower, upper)
        return TemporalPredictionBatch(
            horizons=self.config.horizons,
            values=values,
            target_names=self.config.target_names,
            timestamps=np.asarray(timestamps, dtype=object) if timestamps is not None else None,
            model_name=self.model_name,
        )

    def predict(self, sequence: ArrayLike, *, timestamp: Optional[Any] = None) -> TemporalPrediction:
        """Predict future states for one sequence."""
        result = self.predict_batch(
            np.asarray(sequence, dtype=float)[None, ...],
            timestamps=[timestamp] if timestamp is not None else None,
        )
        return TemporalPrediction(
            horizons=result.horizons,
            values=result.values[0],
            target_names=result.target_names,
            timestamp=timestamp,
            model_name=result.model_name,
            metadata=result.metadata,
        )

    def __call__(self, sequence: ArrayLike, *, timestamp: Optional[Any] = None) -> TemporalPrediction:
        return self.predict(sequence, timestamp=timestamp)


class CallableTemporalPredictor(TemporalPredictor):
    """Adapter that wraps a plain Python prediction function."""

    def __init__(
        self,
        predictor: Callable[[np.ndarray], Any],
        config: Optional[TemporalPredictorConfig] = None,
        model_name: str = "callable_temporal_predictor",
    ) -> None:
        super().__init__(config=config, model_name=model_name)
        if not callable(predictor):
            raise TypeError("predictor must be callable")
        self.predictor = predictor

    def _predict_model(self, sequences: np.ndarray) -> Any:
        return self.predictor(sequences)


class LastValueTemporalPredictor(TemporalPredictor):
    """Persistence baseline that repeats the latest observed state."""

    def __init__(
        self,
        config: Optional[TemporalPredictorConfig] = None,
        target_feature_indices: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(config=config, model_name="last_value_baseline")
        self.target_feature_indices = (
            tuple(int(i) for i in target_feature_indices)
            if target_feature_indices is not None
            else None
        )

    def _predict_model(self, sequences: np.ndarray) -> np.ndarray:
        last_state = sequences[:, -1, :]
        if self.target_feature_indices is not None:
            last_state = last_state[:, self.target_feature_indices]
        return np.repeat(last_state[:, None, :], len(self.config.horizons), axis=1)


def create_temporal_predictor(
    predictor: Union[TemporalPredictor, Callable[[np.ndarray], Any], None],
    config: Optional[TemporalPredictorConfig] = None,
    *,
    model_name: Optional[str] = None,
) -> TemporalPredictor:
    """Create a predictor from an existing model, callable, or baseline."""
    if isinstance(predictor, TemporalPredictor):
        return predictor
    if predictor is None:
        return LastValueTemporalPredictor(config=config)
    if callable(predictor):
        return CallableTemporalPredictor(
            predictor, config=config,
            model_name=model_name or "callable_temporal_predictor",
        )
    raise TypeError("predictor must be a TemporalPredictor, callable, or None")


def prediction_to_horizon_dict(prediction: TemporalPrediction) -> Dict[int, np.ndarray]:
    """Convert a prediction object to a horizon-keyed dictionary."""
    return {h: prediction.at_horizon(h) for h in prediction.horizons}


__all__ = [
    "TemporalPredictorConfig",
    "TemporalPrediction",
    "TemporalPredictionBatch",
    "TemporalPredictor",
    "CallableTemporalPredictor",
    "LastValueTemporalPredictor",
    "create_temporal_predictor",
    "prediction_to_horizon_dict",
]
