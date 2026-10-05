"""Long-term temporal predictor for Digital Twin 2.

Provides farther-future predictions that complement the short-term predictor.
This layer produces evidence for reachability, hazard, survival and mitigation
logic; it does not directly control the insulin pump.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

try:
    from .temporal_predictor import TemporalPredictor, TemporalPredictorConfig
except ImportError:
    from temporal_predictor import TemporalPredictor, TemporalPredictorConfig


@dataclass(frozen=True)
class LongTermPredictorConfig(TemporalPredictorConfig):
    """Configuration for farther-future prediction."""

    horizons: Tuple[int, ...] = (6, 12)
    rollout_mode: str = "direct"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rollout_mode not in {"direct", "autoregressive", "persistence"}:
            raise ValueError(
                "rollout_mode must be 'direct', 'autoregressive', or 'persistence'."
            )


class LongTermPredictor(TemporalPredictor):
    """Model-agnostic long-term DT2 predictor.

    ``model`` may be callable or expose ``predict(sequences)``. With no model,
    a persistence baseline is used for development and benchmarking.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[LongTermPredictorConfig] = None,
        model_name: str = "dt2_long_term_predictor",
    ) -> None:
        self.model = model
        super().__init__(config or LongTermPredictorConfig(), model_name)

    def _predict_model(self, sequences: np.ndarray) -> Any:
        if self.model is None or self.config.rollout_mode == "persistence":
            last_state = sequences[:, -1, :]
            return np.repeat(last_state[:, None, :], len(self.config.horizons), axis=1)
        if callable(self.model):
            return self.model(sequences)
        predict = getattr(self.model, "predict", None)
        if callable(predict):
            return predict(sequences)
        raise TypeError("model must be callable or expose predict().")

    def predict_trajectory(self, sequence: np.ndarray, *, timestamp: Any = None):
        """Predict the farther-future trajectory for one sequence."""
        return self.predict(sequence, timestamp=timestamp)

    def predict_trajectory_batch(self, sequences: np.ndarray, *, timestamps=None):
        """Predict farther-future trajectories for a batch."""
        return self.predict_batch(sequences, timestamps=timestamps)


def create_long_term_predictor(
    model: Optional[Any] = None,
    *,
    history_length: int = 12,
    horizons: Sequence[int] = (6, 12),
    target_names: Optional[Sequence[str]] = None,
    feature_names: Optional[Sequence[str]] = None,
    clip_min: Optional[float] = 20.0,
    clip_max: Optional[float] = 600.0,
) -> LongTermPredictor:
    """Create a configured long-term predictor."""
    config = LongTermPredictorConfig(
        history_length=history_length,
        horizons=tuple(horizons),
        target_names=tuple(target_names) if target_names is not None else None,
        feature_names=tuple(feature_names) if feature_names is not None else None,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    return LongTermPredictor(model=model, config=config)


__all__ = ["LongTermPredictorConfig", "LongTermPredictor", "create_long_term_predictor"]
