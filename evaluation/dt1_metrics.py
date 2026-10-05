"""
DT1 Predictive Digital Twin Evaluation Metrics.

Purpose
-------
Provides regression and forecasting metrics for evaluating DT1 predictions
against observed glucose values.

Designed for:
    digital_twin/dt1_ml/
        predict.py
        validate.py
        inference.py
        residual.py

Primary metrics:
    - MAE
    - MSE
    - RMSE
    - MAPE
    - sMAPE
    - R2
    - mean error / bias
    - error standard deviation
    - median absolute error
    - maximum absolute error
    - prediction coverage
    - clinically-oriented glucose-zone summaries

Also supports:
    - horizon-wise metrics
    - error by glucose range
    - residual summaries
    - comparison of multiple DT1 models

Important
---------
These are research/engineering evaluation metrics. They do not establish
clinical safety or clinical effectiveness.

For glucose forecasting, MAPE can become unstable near zero, so the
implementation uses an explicit denominator floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


@dataclass
class DT1Metrics:
    """Complete metric report for a DT1 prediction set."""

    n: int
    mae: float
    mse: float
    rmse: float
    mean_error: float
    error_std: float
    median_absolute_error: float
    max_absolute_error: float
    mape: float
    smape: float
    r2: float

    # Optional prediction quality summaries.
    correlation: float = float("nan")
    within_5_mgdl: float = float("nan")
    within_10_mgdl: float = float("nan")
    within_15_mgdl: float = float("nan")
    within_20_mgdl: float = float("nan")

    # Glucose-zone metrics.
    hypo_mae: float = float("nan")
    target_mae: float = float("nan")
    hyper_mae: float = float("nan")

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        return {
            "n": self.n,
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
            "mean_error": self.mean_error,
            "error_std": self.error_std,
            "median_absolute_error": self.median_absolute_error,
            "max_absolute_error": self.max_absolute_error,
            "mape": self.mape,
            "smape": self.smape,
            "r2": self.r2,
            "correlation": self.correlation,
            "within_5_mgdl": self.within_5_mgdl,
            "within_10_mgdl": self.within_10_mgdl,
            "within_15_mgdl": self.within_15_mgdl,
            "within_20_mgdl": self.within_20_mgdl,
            "hypo_mae": self.hypo_mae,
            "target_mae": self.target_mae,
            "hyper_mae": self.hyper_mae,
            "metadata": dict(self.metadata),
        }


def _as_array(values: Sequence[float] | Iterable[float]):
    """Convert input to a finite floating-point array."""
    if np is None:
        values = list(values)
        return [float(v) for v in values]

    arr = np.asarray(list(values) if not hasattr(values, "shape") else values, dtype=float)
    return arr.reshape(-1)


def _validate_pair(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length: "
            f"{len(y_true)} != {len(y_pred)}"
        )

    if len(y_true) == 0:
        raise ValueError("At least one prediction is required.")

    if np is not None:
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.all(mask):
            y_true = y_true[mask]
            y_pred = y_pred[mask]

    if len(y_true) == 0:
        raise ValueError("No finite prediction/target pairs remain.")

    return y_true, y_pred


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    y_true, y_pred = _validate_pair(_as_array(y_true), _as_array(y_pred))
    if np is not None:
        return float(np.mean(np.abs(y_pred - y_true)))
    return sum(abs(p - t) for t, p in zip(y_true, y_pred)) / len(y_true)


def mse(y_true, y_pred) -> float:
    """Mean squared error."""
    y_true, y_pred = _validate_pair(_as_array(y_true), _as_array(y_pred))
    if np is not None:
        return float(np.mean((y_pred - y_true) ** 2))
    return sum((p - t) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(math.sqrt(mse(y_true, y_pred)))


def mean_error(y_true, y_pred) -> float:
    """Signed mean error: prediction - observation."""
    y_true, y_pred = _validate_pair(_as_array(y_true), _as_array(y_pred))
    if np is not None:
        return float(np.mean(y_pred - y_true))
    return sum(p - t for t, p in zip(y_true, y_pred)) / len(y_true)


def error_std(y_true, y_pred) -> float:
    """Standard deviation of signed prediction error."""
    y_true, y_pred = _validate_pair(_as_array(y_true), _as_array(y_pred))
    errors = y_pred - y_true
    if np is not None:
        return float(np.std(errors))
    mu = sum(errors) / len(errors)
    return float(math.sqrt(sum((e - mu) ** 2 for e in errors) / len(errors)))


def median_absolute_error(y_true, y_pred) -> float:
    """Median absolute prediction error."""
    y_true, y_pred = _validate_pair(_as_array(y_true), _as_array(y_pred))
    errors = [abs(p - t) for t, p in zip(y_true, y_pred)]
    if np is not None:
        return float(np.median(errors))
    errors.sort()
    n = len(errors)
    mid = n // 2
    if n % 2:
        return float(errors[mid])
    return float((errors[mid - 1] + errors[mid]) / 2)


def max_absolute_error(y_true, y_pred) -> float:
    """Maximum absolute prediction error."""
    y_true, y_pred = _validate_pair(_as_array(y_true), _as_array(y_pred))
    if np is not None:
        return float(np.max(np.abs(y_pred - y_true)))
    return float(max(abs(p - t) for t, p in zip(y_true, y_pred)))


def mape(
    y_true,
    y_pred,
    *,
    denominator_floor: float = 1.0,
    as_percentage: bool = True,
) -> float:
    """
    Mean absolute percentage error.

    A denominator floor avoids numerical instability for small glucose
    values. This should be reported with the chosen floor.
    """
    y_true, y_pred = _validate_pair(_as_array(y_true), _as_array(y_pred))

    if np is not None:
        denom = np.maximum(np.abs(y_true), denominator_floor)
        value = np.mean(np.abs((y_pred - y_true) / denom))
    else:
        value = sum(
            abs((p - t) / max(abs(t), denominator_floor))
            for t, p in zip(y_true, y_pred)
        ) / len(y_true)

    return float(value * 100.0 if as_percentage else value)


def smape(
    y_true,
    y_pred,
    *,
    denominator_floor: float = 1.0,
    as_percentage: bool = True,
) -> float:
    """Symmetric mean absolute percentage error."""
    y_true, y_pred = _validate_pair(_as_array(y_true), _as_array(y_pred))

    if np is not None:
        denom = np.maximum(np.abs(y_true) + np.abs(y_pred), denominator_floor)
        value = np.mean(2.0 * np.abs(y_pred - y_true) / denom)
    else:
        value = sum(
            2.0 * abs(p - t) /
            max(abs(t) + abs(p), denominator_floor)
            for t, p in zip(y_true, y_pred)
        ) / len(y_true)

    return float(value * 100.0 if as_percentage else value)


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination."""
    y_true, y_pred = _validate_pair(_as_array(y_true), _as_array(y_pred))

    if np is not None:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    else:
        mean_true = sum(y_true) / len(y_true)
        ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
        ss_tot = sum((t - mean_true) ** 2 for t in y_true)

    if ss_tot == 0:
        return 0.0

    return float(1.0 - ss_res / ss_tot)


def correlation(y_true, y_pred) -> float:
    """Pearson correlation coefficient."""
    y_true, y_pred = _validate_pair(_as_array(y_true), _as_array(y_pred))

    if len(y_true) < 2:
        return float("nan")

    if np is not None:
        a = float(np.std(y_true))
        b = float(np.std(y_pred))
        if a == 0 or b == 0:
            return float("nan")
        return float(np.corrcoef(y_true, y_pred)[0, 1])

    mean_true = sum(y_true) / len(y_true)
    mean_pred = sum(y_pred) / len(y_pred)

    num = sum(
        (t - mean_true) * (p - mean_pred)
        for t, p in zip(y_true, y_pred)
    )
    den_t = math.sqrt(sum((t - mean_true) ** 2 for t in y_true))
    den_p = math.sqrt(sum((p - mean_pred) ** 2 for p in y_pred))

    if den_t == 0 or den_p == 0:
        return float("nan")

    return float(num / (den_t * den_p))


def within_error(
    y_true,
    y_pred,
    tolerance: float,
) -> float:
    """Fraction of predictions whose absolute error is within tolerance."""
    y_true, y_pred = _validate_pair(_as_array(y_true), _as_array(y_pred))

    if np is not None:
        return float(np.mean(np.abs(y_pred - y_true) <= tolerance))

    return sum(
        abs(p - t) <= tolerance
        for t, p in zip(y_true, y_pred)
    ) / len(y_true)


def glucose_zone_masks(
    y_true,
    *,
    hypoglycemia_threshold: float = 70.0,
    hyperglycemia_threshold: float = 180.0,
):
    """
    Return masks for hypo, target, and hyperglycemic observations.

    Zones:
        hypo   : glucose < 70 mg/dL
        target : 70 <= glucose <= 180 mg/dL
        hyper  : glucose > 180 mg/dL
    """
    y_true = _as_array(y_true)

    if np is not None:
        return (
            y_true < hypoglycemia_threshold,
            (y_true >= hypoglycemia_threshold)
            & (y_true <= hyperglycemia_threshold),
            y_true > hyperglycemia_threshold,
        )

    return (
        [x < hypoglycemia_threshold for x in y_true],
        [
            hypoglycemia_threshold <= x <= hyperglycemia_threshold
            for x in y_true
        ],
        [x > hyperglycemia_threshold for x in y_true],
    )


def zone_mae(
    y_true,
    y_pred,
    *,
    hypoglycemia_threshold: float = 70.0,
    hyperglycemia_threshold: float = 180.0,
) -> Dict[str, float]:
    """Compute MAE separately for hypo/target/hyperglycemic observations."""
    y_true, y_pred = _validate_pair(_as_array(y_true), _as_array(y_pred))
    masks = glucose_zone_masks(
        y_true,
        hypoglycemia_threshold=hypoglycemia_threshold,
        hyperglycemia_threshold=hyperglycemia_threshold,
    )

    result = {}
    names = ("hypo_mae", "target_mae", "hyper_mae")

    for name, mask in zip(names, masks):
        if np is not None:
            if int(np.sum(mask)) == 0:
                result[name] = float("nan")
            else:
                result[name] = float(
                    np.mean(np.abs(y_pred[mask] - y_true[mask]))
                )
        else:
            selected = [
                (t, p)
                for t, p, keep in zip(y_true, y_pred, mask)
                if keep
            ]
            result[name] = (
                float("nan")
                if not selected
                else sum(abs(p - t) for t, p in selected) / len(selected)
            )

    return result


def compute_metrics(
    y_true,
    y_pred,
    *,
    denominator_floor: float = 1.0,
    hypoglycemia_threshold: float = 70.0,
    hyperglycemia_threshold: float = 180.0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> DT1Metrics:
    """Compute the complete DT1 metric report."""
    y_true, y_pred = _validate_pair(_as_array(y_true), _as_array(y_pred))

    zones = zone_mae(
        y_true,
        y_pred,
        hypoglycemia_threshold=hypoglycemia_threshold,
        hyperglycemia_threshold=hyperglycemia_threshold,
    )

    return DT1Metrics(
        n=len(y_true),
        mae=mae(y_true, y_pred),
        mse=mse(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mean_error=mean_error(y_true, y_pred),
        error_std=error_std(y_true, y_pred),
        median_absolute_error=median_absolute_error(y_true, y_pred),
        max_absolute_error=max_absolute_error(y_true, y_pred),
        mape=mape(
            y_true,
            y_pred,
            denominator_floor=denominator_floor,
        ),
        smape=smape(
            y_true,
            y_pred,
            denominator_floor=denominator_floor,
        ),
        r2=r2_score(y_true, y_pred),
        correlation=correlation(y_true, y_pred),
        within_5_mgdl=within_error(y_true, y_pred, 5.0),
        within_10_mgdl=within_error(y_true, y_pred, 10.0),
        within_15_mgdl=within_error(y_true, y_pred, 15.0),
        within_20_mgdl=within_error(y_true, y_pred, 20.0),
        hypo_mae=zones["hypo_mae"],
        target_mae=zones["target_mae"],
        hyper_mae=zones["hyper_mae"],
        metadata={
            "denominator_floor": denominator_floor,
            "hypoglycemia_threshold_mgdl": hypoglycemia_threshold,
            "hyperglycemia_threshold_mgdl": hyperglycemia_threshold,
            **dict(metadata or {}),
        },
    )


def compute_horizon_metrics(
    y_true,
    y_pred,
    *,
    horizon_axis: int = 1,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, DT1Metrics]:
    """
    Compute metrics independently for each forecast horizon.

    Expected shape:
        y_true: (samples, horizons)
        y_pred: (samples, horizons)
    """
    if np is None:
        raise ImportError("compute_horizon_metrics requires NumPy.")

    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)

    if true_arr.shape != pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true={true_arr.shape}, "
            f"y_pred={pred_arr.shape}"
        )

    if true_arr.ndim != 2:
        raise ValueError(
            "Horizon metrics require 2-D arrays: "
            "(samples, horizons)."
        )

    if horizon_axis not in (0, 1):
        raise ValueError("horizon_axis must be 0 or 1.")

    if horizon_axis == 0:
        true_arr = true_arr.T
        pred_arr = pred_arr.T

    reports = {}

    for h in range(true_arr.shape[1]):
        reports[f"horizon_{h + 1}"] = compute_metrics(
            true_arr[:, h],
            pred_arr[:, h],
            metadata={
                **dict(metadata or {}),
                "horizon_index": h,
                "horizon_number": h + 1,
            },
        )

    return reports


def residual_summary(
    y_true,
    y_pred,
) -> Dict[str, float]:
    """Return compact residual statistics useful for DT1 monitoring."""
    y_true, y_pred = _validate_pair(_as_array(y_true), _as_array(y_pred))

    if np is not None:
        residual = y_pred - y_true
        return {
            "mean": float(np.mean(residual)),
            "std": float(np.std(residual)),
            "median": float(np.median(residual)),
            "q05": float(np.quantile(residual, 0.05)),
            "q25": float(np.quantile(residual, 0.25)),
            "q75": float(np.quantile(residual, 0.75)),
            "q95": float(np.quantile(residual, 0.95)),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
        }

    residual = [p - t for t, p in zip(y_true, y_pred)]
    ordered = sorted(residual)

    def percentile(q):
        idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return float(ordered[idx])

    return {
        "mean": sum(residual) / len(residual),
        "std": error_std(y_true, y_pred),
        "median": median_absolute_error(
            [0.0] * len(residual), residual
        ),
        "q05": percentile(0.05),
        "q25": percentile(0.25),
        "q75": percentile(0.75),
        "q95": percentile(0.95),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }


def compare_models(
    y_true,
    predictions: Mapping[str, Sequence[float]],
) -> Dict[str, DT1Metrics]:
    """
    Evaluate multiple DT1 models against the same targets.

    Example:
        compare_models(
            y_test,
            {"lstm": lstm_pred, "gru": gru_pred}
        )
    """
    return {
        name: compute_metrics(y_true, pred, metadata={"model": name})
        for name, pred in predictions.items()
    }


__all__ = [
    "DT1Metrics",
    "mae",
    "mse",
    "rmse",
    "mean_error",
    "error_std",
    "median_absolute_error",
    "max_absolute_error",
    "mape",
    "smape",
    "r2_score",
    "correlation",
    "within_error",
    "glucose_zone_masks",
    "zone_mae",
    "compute_metrics",
    "compute_horizon_metrics",
    "residual_summary",
    "compare_models",
]
