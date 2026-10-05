"""
residual.py

Residual/error calculation for DT1.

Residual:
    r(t) = actual_glucose(t) - predicted_glucose(t)

The residual measures disagreement between the physiological trajectory
and the ML-based Digital Twin prediction.

This residual is an evidence signal for downstream anomaly/hazard analysis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def calculate_residual(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> np.ndarray:
    """Return signed prediction residuals."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: actual={actual.shape}, "
            f"predicted={predicted.shape}."
        )

    return actual - predicted


def absolute_residual(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> np.ndarray:
    """Return absolute prediction errors."""
    return np.abs(
        calculate_residual(actual, predicted)
    )


def squared_residual(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> np.ndarray:
    """Return squared prediction errors."""
    return np.square(
        calculate_residual(actual, predicted)
    )


def normalized_residual(
    residual: np.ndarray,
    *,
    scale: Optional[float] = None,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Normalize residual magnitude.

    If scale is supplied, divide by that fixed scale. Otherwise use the
    standard deviation of the supplied residual sequence.
    """
    residual = np.asarray(residual, dtype=float)

    if scale is None:
        scale = float(np.nanstd(residual))

    if not np.isfinite(scale) or abs(scale) < epsilon:
        scale = 1.0

    return residual / scale


def residual_dataframe(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    timestamps: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Create a convenient residual-analysis DataFrame."""
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)

    residual = calculate_residual(actual, predicted)

    result = pd.DataFrame({
        "actual_glucose": actual,
        "predicted_glucose": predicted,
        "residual": residual,
        "absolute_residual": np.abs(residual),
    })

    if timestamps is not None:
        if len(timestamps) != len(result):
            raise ValueError(
                "timestamps must have the same length as predictions."
            )
        result.insert(
            0,
            "time",
            pd.Series(timestamps).reset_index(drop=True),
        )

    return result


__all__ = [
    "calculate_residual",
    "absolute_residual",
    "squared_residual",
    "normalized_residual",
    "residual_dataframe",
]
