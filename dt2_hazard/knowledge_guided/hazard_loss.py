"""
Hazard-aware prediction losses for DT2.

The hazard loss penalizes predictions that move toward or into configured
hazard regions. It is intentionally distinct from hazard probability:
this module supplies a training/evidence penalty, not a calibrated event
probability.

The default implementation is glucose-centric and can later be extended
to multiple physiological and cyber-physical hazards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class HazardLossConfig:
    """Configuration for hazard-region prediction penalties."""

    low_threshold: float = 70.0
    high_threshold: float = 180.0
    critical_low: float = 54.0
    critical_high: float = 250.0
    warning_weight: float = 1.0
    critical_weight: float = 2.0
    trend_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.low_threshold >= self.high_threshold:
            raise ValueError("low_threshold must be below high_threshold.")
        if self.critical_low >= self.low_threshold:
            raise ValueError("critical_low must be below low_threshold.")
        if self.critical_high <= self.high_threshold:
            raise ValueError("critical_high must exceed high_threshold.")
        if min(
            self.warning_weight,
            self.critical_weight,
            self.trend_weight,
        ) < 0:
            raise ValueError("Loss weights must be non-negative.")


def hazard_severity(
    glucose: Sequence[float],
    *,
    config: Optional[HazardLossConfig] = None,
) -> np.ndarray:
    """
    Compute a normalized hazard-severity score in [0, 1].

    This is an engineering severity indicator, not a probability.
    """
    config = config or HazardLossConfig()
    g = np.asarray(glucose, dtype=float)

    low_range = np.maximum(config.low_threshold - g, 0.0)
    high_range = np.maximum(g - config.high_threshold, 0.0)

    low_scale = max(config.low_threshold - config.critical_low, 1e-6)
    high_scale = max(config.critical_high - config.high_threshold, 1e-6)

    low_score = np.clip(low_range / low_scale, 0.0, 1.0)
    high_score = np.clip(high_range / high_scale, 0.0, 1.0)

    return np.maximum(low_score, high_score)


def hazard_loss(
    predictions: Sequence[float],
    *,
    previous_glucose: Optional[float] = None,
    config: Optional[HazardLossConfig] = None,
) -> float:
    """
    Penalize predictions entering increasingly hazardous glucose regions.

    A trend term optionally adds penalty when the prediction is moving
    farther toward a hazard region.
    """
    config = config or HazardLossConfig()
    g = np.asarray(predictions, dtype=float).reshape(-1)

    if len(g) == 0:
        return 0.0
    if not np.isfinite(g).all():
        raise ValueError("predictions contain non-finite values.")

    severity = hazard_severity(g, config=config)
    critical = (
        (g < config.critical_low) |
        (g > config.critical_high)
    ).astype(float)

    trend_penalty = 0.0
    if previous_glucose is not None and len(g):
        delta = g[0] - float(previous_glucose)
        toward_low = (g[0] < config.low_threshold) and (delta < 0)
        toward_high = (g[0] > config.high_threshold) and (delta > 0)
        if toward_low or toward_high:
            trend_penalty = abs(delta)

    return float(
        config.warning_weight * np.mean(severity)
        + config.critical_weight * np.mean(critical)
        + config.trend_weight * trend_penalty / 100.0
    )


def hazard_margin(
    glucose: Sequence[float],
    *,
    config: Optional[HazardLossConfig] = None,
) -> np.ndarray:
    """Return signed distance to the nearest warning hazard boundary."""
    config = config or HazardLossConfig()
    g = np.asarray(glucose, dtype=float)
    return np.minimum(
        g - config.low_threshold,
        config.high_threshold - g,
    )


__all__ = [
    "HazardLossConfig",
    "hazard_severity",
    "hazard_loss",
    "hazard_margin",
]
