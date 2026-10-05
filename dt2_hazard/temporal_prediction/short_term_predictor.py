"""
Short-term temporal predictor for Digital Twin 2.

The short-term predictor is the near-future prediction component of DT2.
It is intended to provide relatively high-resolution predictions over a
short horizon so that the hazard, survival, reachability, and safety
layers can react before a predicted trajectory enters an unsafe region.

The design is inspired by the two-level temporal prediction idea used in
KnowSafe: a short-term model focuses on near-future accuracy while a
separate long-term predictor can provide earlier warning.  This module
only produces predictions; it does not make pump-control decisions.

Default timing assumes the DT2/SimGlucose data stream is sampled every
5 minutes:

    horizon 1 -> 5 minutes
    horizon 2 -> 10 minutes
    horizon 3 -> 15 minutes

The sampling interval is configurable, so the class can also be used
with other data frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from .temporal_predictor import (
        ArrayLike,
        CallableTemporalPredictor,
        LastValueTemporalPredictor,
        TemporalPrediction,
        TemporalPredictionBatch,
        TemporalPredictor,
        TemporalPredictorConfig,
    )
except ImportError:  # pragma: no cover - supports direct file execution
    from temporal_predictor import (
        ArrayLike,
        CallableTemporalPredictor,
        LastValueTemporalPredictor,
        TemporalPrediction,
        TemporalPredictionBatch,
        TemporalPredictor,
        TemporalPredictorConfig,
    )


@dataclass(frozen=True)
class ShortTermConfig:
    """Configuration for the DT2 short-term prediction layer."""

    history_length: int = 12
    sample_interval_minutes: float = 5.0
    horizons: Tuple[int, ...] = (1, 2, 3)
    target_names: Optional[Tuple[str, ...]] = None
    feature_names: Optional[Tuple[str, ...]] = None
    clip_min: Optional[float] = 20.0
    clip_max: Optional[float] = 600.0
    require_finite_input: bool = True
    use_persistence_baseline: bool = True

    def __post_init__(self) -> None:
        if self.history_length <= 0:
            raise ValueError("history_length must be positive.")
        if self.sample_interval_minutes <= 0:
            raise ValueError("sample_interval_minutes must be positive.")
        if not self.horizons:
            raise ValueError("At least one short-term horizon is required.")

        normalized = tuple(sorted(set(int(h) for h in self.horizons)))
        if any(h <= 0 for h in normalized):
            raise ValueError("All short-term horizons must be positive.")
        object.__setattr__(self, "horizons", normalized)

        if (
            self.clip_min is not None
            and self.clip_max is not None
            and self.clip_min >= self.clip_max
        ):
            raise ValueError("clip_min must be smaller than clip_max.")

    @property
    def horizon_minutes(self) -> Tuple[float, ...]:
        """Convert discrete horizons to minutes."""
        return tuple(
            float(h * self.sample_interval_minutes)
            for h in self.horizons
        )

    @property
    def max_horizon_minutes(self) -> float:
        """Largest short-term prediction horizon in minutes."""
        return max(self.horizon_minutes)

    def to_predictor_config(self) -> TemporalPredictorConfig:
        """Create the common temporal predictor configuration."""
        return TemporalPredictorConfig(
            history_length=self.history_length,
            horizons=self.horizons,
            target_names=self.target_names,
            feature_names=self.feature_names,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            require_finite_input=self.require_finite_input,
        )


@dataclass
class ShortTermPrediction:
    """Packaged short-term prediction with time-based horizon metadata."""

    prediction: TemporalPrediction
    horizon_minutes: Tuple[float, ...]
    layer: str = "short_term"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def at_minutes(self, minutes: float) -> np.ndarray:
        """Return the prediction corresponding to a requested horizon."""
        differences = np.abs(
            np.asarray(self.horizon_minutes, dtype=float) - float(minutes)
        )
        index = int(np.argmin(differences))
        if not np.isclose(
            self.horizon_minutes[index],
            float(minutes),
            atol=1e-9,
        ):
            raise KeyError(
                f"No short-term prediction exists at {minutes} minutes. "
                f"Available horizons: {self.horizon_minutes}."
            )
        return self.prediction.values[index]

    @property
    def values(self) -> np.ndarray:
        """Prediction values ordered by configured horizon."""
        return self.prediction.values

    @property
    def max_horizon_minutes(self) -> float:
        """Largest available prediction horizon in minutes."""
        return max(self.horizon_minutes)


@dataclass
class ShortTermPredictionBatch:
    """Batch version of :class:`ShortTermPrediction`."""

    prediction: TemporalPredictionBatch
    horizon_minutes: Tuple[float, ...]
    layer: str = "short_term"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def values(self) -> np.ndarray:
        """Prediction values with shape (batch, horizons, targets)."""
        return self.prediction.values

    @property
    def n_samples(self) -> int:
        """Number of predicted sequences."""
        return self.prediction.n_samples

    def at_minutes(self, minutes: float) -> np.ndarray:
        """Return batch predictions at a requested time horizon."""
        differences = np.abs(
            np.asarray(self.horizon_minutes, dtype=float) - float(minutes)
        )
        index = int(np.argmin(differences))
        if not np.isclose(
            self.horizon_minutes[index],
            float(minutes),
            atol=1e-9,
        ):
            raise KeyError(
                f"No short-term prediction exists at {minutes} minutes. "
                f"Available horizons: {self.horizon_minutes}."
            )
        return self.prediction.values[:, index, :]


class ShortTermPredictor:
    """
    Near-future DT2 prediction wrapper.

    Parameters
    ----------
    model:
        A ``TemporalPredictor`` instance, a callable accepting a NumPy
        batch of shape ``(batch, history, features)``, or ``None``.
        ``None`` uses a persistence baseline when enabled.
    config:
        Short-term prediction configuration.

    Notes
    -----
    A production implementation can pass a trained GRU/LSTM/Transformer
    through the ``model`` argument without changing the downstream DT2
    interface.  The model should predict state variables needed by the
    hazard and reachability layers, for example glucose, insulin, or
    other selected DT2 state features.
    """

    def __init__(
        self,
        model: Optional[
            Union[
                TemporalPredictor,
                Callable[[np.ndarray], Any],
            ]
        ] = None,
        config: Optional[ShortTermConfig] = None,
    ) -> None:
        self.config = config or ShortTermConfig()

        if model is None:
            if not self.config.use_persistence_baseline:
                raise ValueError(
                    "model is None and use_persistence_baseline is False."
                )
            self.model: TemporalPredictor = LastValueTemporalPredictor(
                config=self.config.to_predictor_config(),
            )
        elif isinstance(model, TemporalPredictor):
            self.model = model
        elif callable(model):
            self.model = CallableTemporalPredictor(
                predictor=model,
                config=self.config.to_predictor_config(),
                model_name="short_term_callable_model",
            )
        else:
            raise TypeError(
                "model must be a TemporalPredictor, callable, or None."
            )

        self._validate_model_config()

    def _validate_model_config(self) -> None:
        """Ensure the wrapped model agrees with short-term timing."""
        model_config = self.model.config

        if model_config.history_length != self.config.history_length:
            raise ValueError(
                "Short-term model history length does not match config: "
                f"{model_config.history_length} != "
                f"{self.config.history_length}."
            )

        if tuple(model_config.horizons) != tuple(self.config.horizons):
            raise ValueError(
                "Short-term model horizons do not match config: "
                f"{model_config.horizons} != {self.config.horizons}."
            )

    def predict(
        self,
        sequence: ArrayLike,
        *,
        timestamp: Optional[Any] = None,
    ) -> ShortTermPrediction:
        """Generate short-term predictions for one input sequence."""
        result = self.model.predict(sequence, timestamp=timestamp)

        return ShortTermPrediction(
            prediction=result,
            horizon_minutes=self.config.horizon_minutes,
            metadata={
                "sample_interval_minutes": self.config.sample_interval_minutes,
                "horizon_steps": self.config.horizons,
                "model_name": result.model_name,
            },
        )

    def predict_batch(
        self,
        sequences: ArrayLike,
        *,
        timestamps: Optional[Sequence[Any]] = None,
    ) -> ShortTermPredictionBatch:
        """Generate short-term predictions for a sequence batch."""
        result = self.model.predict_batch(
            sequences,
            timestamps=timestamps,
        )

        return ShortTermPredictionBatch(
            prediction=result,
            horizon_minutes=self.config.horizon_minutes,
            metadata={
                "sample_interval_minutes": self.config.sample_interval_minutes,
                "horizon_steps": self.config.horizons,
                "model_name": result.model_name,
            },
        )

    def predict_glucose(
        self,
        sequence: ArrayLike,
        *,
        glucose_target_index: int = 0,
        timestamp: Optional[Any] = None,
    ) -> ShortTermPrediction:
        """
        Generate short-term predictions and retain only glucose.

        This helper is useful when the underlying model predicts multiple
        state variables but the immediate consumer needs the glucose
        trajectory for hazard/reachability analysis.
        """
        result = self.predict(sequence, timestamp=timestamp)
        index = int(glucose_target_index)

        if index < 0 or index >= result.values.shape[1]:
            raise IndexError(
                f"glucose_target_index {index} is outside the target range."
            )

        glucose_values = result.values[:, index : index + 1]
        glucose_target_name = (
            ("glucose",)
            if result.prediction.target_names is None
            else (result.prediction.target_names[index],)
        )

        prediction = TemporalPrediction(
            horizons=result.prediction.horizons,
            values=glucose_values,
            target_names=glucose_target_name,
            timestamp=result.prediction.timestamp,
            model_name=result.prediction.model_name,
            metadata=dict(result.prediction.metadata),
        )

        return ShortTermPrediction(
            prediction=prediction,
            horizon_minutes=result.horizon_minutes,
            metadata=dict(result.metadata),
        )

    def __call__(
        self,
        sequence: ArrayLike,
        *,
        timestamp: Optional[Any] = None,
    ) -> ShortTermPrediction:
        """Alias for :meth:`predict`."""
        return self.predict(sequence, timestamp=timestamp)


def create_short_term_predictor(
    model: Optional[
        Union[
            TemporalPredictor,
            Callable[[np.ndarray], Any],
        ]
    ] = None,
    config: Optional[ShortTermConfig] = None,
) -> ShortTermPredictor:
    """Factory for constructing the DT2 short-term predictor."""
    return ShortTermPredictor(model=model, config=config)


__all__ = [
    "ShortTermConfig",
    "ShortTermPrediction",
    "ShortTermPredictionBatch",
    "ShortTermPredictor",
    "create_short_term_predictor",
]
