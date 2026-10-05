"""
Safety-aware loss functions for DT2 temporal prediction.

These losses penalize predicted physiological states that violate
configured safety bounds. They are intended to complement ordinary
prediction losses such as MSE/MAE.

The loss is differentiable with respect to predictions and uses smooth
penalties where practical, making it suitable for neural-network training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class SafetyLossConfig:
    """Configuration for safety-constraint penalties."""

    glucose_min: float = 70.0
    glucose_max: float = 180.0
    hard_min: float = 20.0
    hard_max: float = 600.0
    weight: float = 1.0
    hard_weight: float = 2.0
    smoothness: float = 5.0

    def __post_init__(self) -> None:
        if self.glucose_min >= self.glucose_max:
            raise ValueError("glucose_min must be below glucose_max.")
        if self.hard_min >= self.hard_max:
            raise ValueError("hard_min must be below hard_max.")
        if self.weight < 0 or self.hard_weight < 0:
            raise ValueError("Loss weights must be non-negative.")
        if self.smoothness <= 0:
            raise ValueError("smoothness must be positive.")


def _softplus(x: np.ndarray, beta: float) -> np.ndarray:
    """Numerically stable softplus."""
    z = beta * x
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def safety_violation_penalty(
    predictions: Sequence[float],
    *,
    config: Optional[SafetyLossConfig] = None,
) -> float:
    """
    Compute a smooth penalty for predicted glucose outside safe bounds.

    The penalty is zero when all predictions lie within the target range.
    Hard physical-range violations receive an additional penalty.
    """
    config = config or SafetyLossConfig()
    g = np.asarray(predictions, dtype=float)

    if not np.isfinite(g).all():
        raise ValueError("predictions contain non-finite values.")

    low = _softplus(config.glucose_min - g, config.smoothness)
    high = _softplus(g - config.glucose_max, config.smoothness)

    hard_low = _softplus(config.hard_min - g, config.smoothness)
    hard_high = _softplus(g - config.hard_max, config.smoothness)

    target_penalty = np.mean(low + high)
    hard_penalty = np.mean(hard_low + hard_high)

    return float(
        config.weight * target_penalty
        + config.hard_weight * hard_penalty
    )


def safety_loss(
    predictions: Sequence[float],
    *,
    config: Optional[SafetyLossConfig] = None,
) -> float:
    """Alias for :func:`safety_violation_penalty`."""
    return safety_violation_penalty(predictions, config=config)


def safety_margin(
    predictions: Sequence[float],
    *,
    lower: float = 70.0,
    upper: float = 180.0,
) -> np.ndarray:
    """
    Return distance to the nearest safe-boundary.

    Positive values indicate distance inside the interval; negative values
    indicate violation magnitude.
    """
    g = np.asarray(predictions, dtype=float)
    return np.minimum(g - lower, upper - g)


__all__ = [
    "SafetyLossConfig",
    "safety_violation_penalty",
    "safety_loss",
    "safety_margin",
]
