"""Prediction uncertainty utilities for DT2 temporal prediction.

Prediction uncertainty is distinct from hazard probability. These utilities
provide empirical residual, rolling, ensemble and conformal-style intervals.
They are research/engineering estimates until calibrated on appropriate data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class UncertaintyConfig:
    interval_level: float = 0.95
    min_scale: float = 1e-6
    calibration_window: int = 100
    ddof: int = 1

    def __post_init__(self):
        if not 0 < self.interval_level < 1:
            raise ValueError("interval_level must be between 0 and 1")
        if self.min_scale <= 0 or self.calibration_window <= 0 or self.ddof < 0:
            raise ValueError("invalid uncertainty configuration")


@dataclass
class PredictionInterval:
    lower: np.ndarray
    upper: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    level: float
    method: str = "empirical"

    @property
    def width(self):
        return self.upper - self.lower


@dataclass
class UncertaintyEstimate:
    scale: np.ndarray
    level: float
    method: str
    sample_count: int
    metadata: Dict[str, object]

    def interval(self, prediction: np.ndarray, z_value: Optional[float] = None):
        if z_value is None:
            z_value = normal_quantile(self.level)
        center = np.asarray(prediction, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        return PredictionInterval(
            lower=center - z_value * scale,
            upper=center + z_value * scale,
            center=center,
            scale=scale,
            level=self.level,
            method=self.method,
        )


def normal_quantile(level: float) -> float:
    """Return an approximate two-sided standard-normal quantile."""
    table = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.975: 2.2414, 0.99: 2.5758}
    if level in table:
        return table[level]
    return _inverse_normal_cdf(0.5 + level / 2)


def _inverse_normal_cdf(p: float) -> float:
    # Acklam rational approximation.
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1")
    a = (-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924)
    b = (-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857)
    c = (-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878)
    d = (0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742)
    plow, phigh = 0.02425, 0.97575
    if p < plow:
        q = np.sqrt(-2*np.log(p))
        return float((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
    if p > phigh:
        q = np.sqrt(-2*np.log(1-p))
        numerator = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        denominator = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return float(-(numerator / denominator))
    q, r = p-0.5, (p-0.5)**2
    numerator = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    denominator = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1.0
    return float(numerator / denominator)


def _matrix(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("values must be 1-D or 2-D")
    return values


def residual_scale(residuals, *, config: Optional[UncertaintyConfig] = None):
    """Estimate per-target standard deviation from residuals."""
    config = config or UncertaintyConfig()
    values = _matrix(residuals)
    scales = np.full(values.shape[1], config.min_scale)
    for j in range(values.shape[1]):
        col = values[np.isfinite(values[:, j]), j]
        if len(col) > config.ddof:
            s = np.std(col, ddof=config.ddof)
            if np.isfinite(s):
                scales[j] = max(float(s), config.min_scale)
    return scales


def rolling_residual_scale(residuals, *, window=100, min_scale=1e-6):
    """Estimate scale from the most recent residual window."""
    return residual_scale(residuals[-window:], config=UncertaintyConfig(min_scale=min_scale, calibration_window=window))


def conformal_nonconformity_scale(residuals, *, level=0.95, min_scale=1e-6):
    """Empirical quantile of absolute residuals; formal coverage needs calibration assumptions."""
    values = _matrix(residuals)
    scales = np.full(values.shape[1], min_scale)
    for j in range(values.shape[1]):
        col = np.abs(values[:, j])
        col = col[np.isfinite(col)]
        if len(col):
            scales[j] = max(float(np.quantile(col, level)), min_scale)
    return scales


def gaussian_prediction_interval(prediction, scale, *, level=0.95):
    """Construct a Gaussian-style prediction interval."""
    center, scale = np.asarray(prediction, dtype=float), np.asarray(scale, dtype=float)
    z = normal_quantile(level)
    return PredictionInterval(center-z*scale, center+z*scale, center, scale, level, "gaussian")


def conformal_prediction_interval(prediction, nonconformity_scale, *, level=0.95):
    """Construct an empirical absolute-residual interval."""
    center, scale = np.asarray(prediction, dtype=float), np.asarray(nonconformity_scale, dtype=float)
    return PredictionInterval(center-scale, center+scale, center, scale, level, "conformal_style")


def ensemble_uncertainty(predictions, *, min_scale=1e-6):
    """Estimate uncertainty from an ensemble dimension at axis 0."""
    values = np.asarray(predictions, dtype=float)
    if values.ndim < 2:
        raise ValueError("predictions must contain an ensemble dimension")
    scale = np.nanstd(values, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale), np.maximum(scale, min_scale), min_scale)
    return UncertaintyEstimate(scale, 0.95, "ensemble_spread", int(values.shape[0]), {})


def estimate_uncertainty(residuals, *, method="std", config=None):
    """Estimate uncertainty using standard deviation, rolling, or conformal scale."""
    config = config or UncertaintyConfig()
    method = method.lower()
    if method == "std":
        scale = residual_scale(residuals, config=config)
    elif method in {"rolling", "rolling_std"}:
        scale = rolling_residual_scale(residuals, window=config.calibration_window, min_scale=config.min_scale)
    elif method in {"conformal", "absolute_residual"}:
        scale = conformal_nonconformity_scale(residuals, level=config.interval_level, min_scale=config.min_scale)
    else:
        raise ValueError("method must be one of: std, rolling, conformal")
    return UncertaintyEstimate(scale, config.interval_level, method, int(np.asarray(residuals).shape[0]), {})


__all__ = [
    "UncertaintyConfig", "PredictionInterval", "UncertaintyEstimate",
    "normal_quantile", "residual_scale", "rolling_residual_scale",
    "conformal_nonconformity_scale", "gaussian_prediction_interval",
    "conformal_prediction_interval", "ensemble_uncertainty", "estimate_uncertainty",
]
