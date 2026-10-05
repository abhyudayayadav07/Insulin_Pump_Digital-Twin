"""
threshold.py

Residual threshold estimation and anomaly classification for DT1.

The threshold module converts continuous residual evidence into an anomaly
signal. It does not make the final safety decision; that belongs to the
decision_engine layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ResidualThreshold:
    """A fitted residual threshold."""

    threshold: float
    method: str
    percentile: Optional[float] = None
    scale: Optional[float] = None

    def is_anomaly(self, residual: float) -> bool:
        return bool(abs(float(residual)) > self.threshold)

    def classify(
        self,
        residuals: np.ndarray,
    ) -> np.ndarray:
        residuals = np.asarray(residuals, dtype=float)
        return np.abs(residuals) > self.threshold


def percentile_threshold(
    residuals: np.ndarray,
    percentile: float = 99.0,
) -> ResidualThreshold:
    """Estimate a threshold from a reference residual distribution."""
    residuals = np.asarray(residuals, dtype=float)
    finite = np.abs(residuals[np.isfinite(residuals)])

    if finite.size == 0:
        raise ValueError("No finite residuals available.")

    if not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100.")

    threshold = float(np.percentile(finite, percentile))

    return ResidualThreshold(
        threshold=threshold,
        method="percentile",
        percentile=percentile,
    )


def standard_deviation_threshold(
    residuals: np.ndarray,
    multiplier: float = 3.0,
) -> ResidualThreshold:
    """Estimate threshold as multiplier times residual standard deviation."""
    residuals = np.asarray(residuals, dtype=float)
    finite = residuals[np.isfinite(residuals)]

    if finite.size == 0:
        raise ValueError("No finite residuals available.")
    if multiplier <= 0:
        raise ValueError("multiplier must be positive.")

    mean = float(np.mean(finite))
    std = float(np.std(finite))

    return ResidualThreshold(
        threshold=abs(mean) + multiplier * std,
        method="std",
        scale=std,
    )


def classify_residuals(
    residuals: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return a Boolean anomaly mask."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    return np.abs(
        np.asarray(residuals, dtype=float)
    ) > threshold


__all__ = [
    "ResidualThreshold",
    "percentile_threshold",
    "standard_deviation_threshold",
    "classify_residuals",
]
