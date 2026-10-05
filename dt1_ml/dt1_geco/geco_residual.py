"""
GECO physiological residual calculation for DT1.

This module compares GECO-predicted glucose trajectories with observed
glucose/CGM measurements.

The residual is intended to provide a physiological-model consistency
signal for the hybrid DT1 pipeline:

    Observations
         +
    GECO state/parameters
         |
         v
    GECO prediction
         |
         v
    GECO residual
         |
         +--> anomaly monitoring / CUSUM
         +--> evidence fusion
         +--> security / hazard reasoning

Important:
    A residual is an evidence signal, not a final safety decision.
    Thresholds and normalization require validation on appropriate data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ResidualPoint:
    """Residual information for one prediction/observation pair."""

    step: int
    time_minutes: float
    predicted: float
    observed: float
    residual: float
    absolute_residual: float
    squared_residual: float
    normalized_residual: float
    valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "time_minutes": self.time_minutes,
            "predicted": self.predicted,
            "observed": self.observed,
            "residual": self.residual,
            "absolute_residual": self.absolute_residual,
            "squared_residual": self.squared_residual,
            "normalized_residual": self.normalized_residual,
            "valid": self.valid,
        }


@dataclass
class GECORResidual:
    """
    Compatibility result for a GECO residual series.

    The class name intentionally uses ``GECORResidual`` to remain
    compatible with the planned DT1 package API.
    """

    points: List[ResidualPoint]
    scale: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def residuals(self) -> np.ndarray:
        return np.asarray(
            [p.residual for p in self.points if p.valid],
            dtype=float,
        )

    @property
    def normalized_residuals(self) -> np.ndarray:
        return np.asarray(
            [p.normalized_residual for p in self.points if p.valid],
            dtype=float,
        )

    @property
    def predicted(self) -> np.ndarray:
        return np.asarray(
            [p.predicted for p in self.points if p.valid],
            dtype=float,
        )

    @property
    def observed(self) -> np.ndarray:
        return np.asarray(
            [p.observed for p in self.points if p.valid],
            dtype=float,
        )

    @property
    def absolute_residuals(self) -> np.ndarray:
        return np.abs(self.residuals)

    def mean_residual(self) -> float:
        values = self.residuals
        return float(np.mean(values)) if values.size else float("nan")

    def mean_absolute_residual(self) -> float:
        values = self.absolute_residuals
        return float(np.mean(values)) if values.size else float("nan")

    def rmse(self) -> float:
        values = self.residuals
        return float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan")

    def std_residual(self) -> float:
        values = self.residuals
        return float(np.std(values)) if values.size else float("nan")

    def max_absolute_residual(self) -> float:
        values = self.absolute_residuals
        return float(np.max(values)) if values.size else float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": [point.to_dict() for point in self.points],
            "scale": self.scale,
            "metadata": dict(self.metadata),
        }


# Alias with a more conventional spelling for users who prefer it.
GECORResidual = GECORResidual


class GECOResidualCalculator:
    """
    Calculate residuals between GECO predictions and observations.

    Convention
    ----------
    residual = observed - predicted

    Therefore:
        positive residual -> observed glucose is higher than GECO prediction
        negative residual -> observed glucose is lower than GECO prediction

    This convention is kept consistent throughout the module.
    """

    def __init__(
        self,
        scale: float = 1.0,
        min_scale: float = 1e-6,
        clip_normalized: Optional[float] = None,
    ):
        if scale <= 0:
            raise ValueError("scale must be > 0.")

        if min_scale <= 0:
            raise ValueError("min_scale must be > 0.")

        if clip_normalized is not None and clip_normalized <= 0:
            raise ValueError("clip_normalized must be > 0 when provided.")

        self.scale = max(float(scale), float(min_scale))
        self.min_scale = float(min_scale)
        self.clip_normalized = clip_normalized

    # ------------------------------------------------------------------
    # Core calculation
    # ------------------------------------------------------------------

    def calculate(
        self,
        predicted: Sequence[float],
        observed: Sequence[float],
        time_minutes: Optional[Sequence[float]] = None,
        skip_invalid: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GECORResidual:
        """
        Calculate residuals for aligned predicted and observed values.

        Parameters
        ----------
        predicted:
            GECO-predicted glucose values.

        observed:
            Measured glucose/CGM values.

        time_minutes:
            Optional timestamps relative to the prediction origin.

        skip_invalid:
            If True, NaN/inf pairs are omitted from the result.

        metadata:
            Optional metadata attached to the result.
        """
        pred = np.asarray(predicted, dtype=float).reshape(-1)
        obs = np.asarray(observed, dtype=float).reshape(-1)

        if pred.size != obs.size:
            raise ValueError(
                "predicted and observed must contain the same number of values."
            )

        if time_minutes is None:
            times = np.arange(pred.size, dtype=float)
        else:
            times = np.asarray(time_minutes, dtype=float).reshape(-1)
            if times.size != pred.size:
                raise ValueError(
                    "time_minutes must have the same length as predicted."
                )

        points: List[ResidualPoint] = []

        for i, (p, o, t) in enumerate(zip(pred, obs, times)):
            valid = bool(
                np.isfinite(p)
                and np.isfinite(o)
                and np.isfinite(t)
            )

            if not valid and skip_invalid:
                continue

            if valid:
                residual = float(o - p)
                absolute = abs(residual)
                squared = residual * residual
                normalized = residual / self.scale

                if self.clip_normalized is not None:
                    normalized = float(
                        np.clip(
                            normalized,
                            -self.clip_normalized,
                            self.clip_normalized,
                        )
                    )
            else:
                residual = float("nan")
                absolute = float("nan")
                squared = float("nan")
                normalized = float("nan")

            points.append(
                ResidualPoint(
                    step=i,
                    time_minutes=float(t),
                    predicted=float(p),
                    observed=float(o),
                    residual=residual,
                    absolute_residual=absolute,
                    squared_residual=squared,
                    normalized_residual=normalized,
                    valid=valid,
                )
            )

        result_metadata = {
            "residual_definition": "observed - predicted",
            "scale": self.scale,
            "n_input": int(pred.size),
            "n_valid": int(sum(point.valid for point in points)),
        }

        if metadata:
            result_metadata.update(metadata)

        return GECORResidual(
            points=points,
            scale=self.scale,
            metadata=result_metadata,
        )

    def calculate_from_trajectory(
        self,
        trajectory: Any,
        observed: Sequence[float],
        use_interstitial: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GECORResidual:
        """
        Calculate residuals directly from a GlucoseTrajectory-like object.

        The object is expected to expose:
            glucose
            interstitial_glucose
            time_minutes
        """
        if use_interstitial:
            predicted = np.asarray(
                trajectory.interstitial_glucose,
                dtype=float,
            )
        else:
            predicted = np.asarray(
                trajectory.glucose,
                dtype=float,
            )

        times = np.asarray(
            trajectory.time_minutes,
            dtype=float,
        )

        return self.calculate(
            predicted=predicted,
            observed=observed,
            time_minutes=times,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Scale estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_scale(
        residuals: Sequence[float],
        quantile: float = 0.95,
        divisor: float = 1.96,
        minimum: float = 1.0,
    ) -> float:
        """
        Estimate a robust residual scale.

        The default follows the GECO study's style of converting a high
        quantile of absolute residuals into a scale:

            scale = q_quantile(|residual|) / 1.96

        with a lower bound of 1 mg/dL.

        This is an engineering/research estimator and should be fitted
        causally on the appropriate training/fit data before deployment.
        """
        if not 0 < quantile < 1:
            raise ValueError("quantile must be between 0 and 1.")

        if divisor <= 0:
            raise ValueError("divisor must be > 0.")

        if minimum <= 0:
            raise ValueError("minimum must be > 0.")

        values = np.asarray(residuals, dtype=float).reshape(-1)
        values = np.abs(values[np.isfinite(values)])

        if values.size == 0:
            return float(minimum)

        q = float(np.quantile(values, quantile))
        return max(float(minimum), q / float(divisor))

    def fit_scale(
        self,
        residuals: Sequence[float],
        quantile: float = 0.95,
        divisor: float = 1.96,
        minimum: float = 1.0,
    ) -> float:
        """Estimate and update the calculator's residual scale."""
        self.scale = self.estimate_scale(
            residuals=residuals,
            quantile=quantile,
            divisor=divisor,
            minimum=minimum,
        )
        return self.scale

    # ------------------------------------------------------------------
    # Residual transformations
    # ------------------------------------------------------------------

    @staticmethod
    def absolute(residuals: Sequence[float]) -> np.ndarray:
        """Return absolute residual magnitude."""
        values = np.asarray(residuals, dtype=float)
        return np.abs(values)

    @staticmethod
    def squared(residuals: Sequence[float]) -> np.ndarray:
        """Return squared residuals."""
        values = np.asarray(residuals, dtype=float)
        return values ** 2

    def normalized(
        self,
        residuals: Sequence[float],
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """Normalize residuals by the supplied or configured scale."""
        values = np.asarray(residuals, dtype=float)
        denominator = self.scale if scale is None else float(scale)

        if denominator <= 0:
            raise ValueError("scale must be > 0.")

        return values / max(denominator, self.min_scale)

    # ------------------------------------------------------------------
    # Residual summaries
    # ------------------------------------------------------------------

    @staticmethod
    def summarize(residuals: Sequence[float]) -> Dict[str, float]:
        """Return descriptive statistics for a residual series."""
        values = np.asarray(residuals, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]

        if values.size == 0:
            return {
                "n": 0.0,
                "mean": float("nan"),
                "std": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
                "median": float("nan"),
                "q05": float("nan"),
                "q95": float("nan"),
                "max_abs": float("nan"),
            }

        return {
            "n": float(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "mae": float(np.mean(np.abs(values))),
            "rmse": float(np.sqrt(np.mean(values ** 2))),
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
            "max_abs": float(np.max(np.abs(values))),
        }

    # ------------------------------------------------------------------
    # Alignment helpers
    # ------------------------------------------------------------------

    @staticmethod
    def align_prediction_observation(
        predicted: Sequence[float],
        observed: Sequence[float],
        prediction_horizon: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align a prediction series with observations.

        ``prediction_horizon`` can be used when the predicted sequence
        corresponds to a future offset.

        Example:
            horizon=1 means prediction[t] is compared with observation[t+1].
        """
        pred = np.asarray(predicted, dtype=float).reshape(-1)
        obs = np.asarray(observed, dtype=float).reshape(-1)

        if prediction_horizon < 0:
            raise ValueError("prediction_horizon must be >= 0.")

        h = int(prediction_horizon)

        if h == 0:
            n = min(pred.size, obs.size)
            return pred[:n], obs[:n]

        if pred.size <= 0 or obs.size <= h:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        n = min(pred.size, obs.size - h)

        return pred[:n], obs[h : h + n]


# ----------------------------------------------------------------------
# Functional helpers
# ----------------------------------------------------------------------

def compute_geco_residual(
    predicted: Sequence[float],
    observed: Sequence[float],
    scale: float = 1.0,
    time_minutes: Optional[Sequence[float]] = None,
) -> GECORResidual:
    """Functional wrapper for residual calculation."""
    calculator = GECOResidualCalculator(scale=scale)

    return calculator.calculate(
        predicted=predicted,
        observed=observed,
        time_minutes=time_minutes,
    )


def estimate_geco_residual_scale(
    residuals: Sequence[float],
    quantile: float = 0.95,
    divisor: float = 1.96,
    minimum: float = 1.0,
) -> float:
    """Functional wrapper for residual-scale estimation."""
    return GECOResidualCalculator.estimate_scale(
        residuals=residuals,
        quantile=quantile,
        divisor=divisor,
        minimum=minimum,
    )


__all__ = [
    "ResidualPoint",
    "GECORResidual",
    "GECORResidual",
    "GECOResidualCalculator",
    "compute_geco_residual",
    "estimate_geco_residual_scale",
]
