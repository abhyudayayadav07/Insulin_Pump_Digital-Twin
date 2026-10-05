"""
DT2 Reactive-Safe Digital Twin Evaluation Metrics.

Purpose
-------
Evaluates the reactive-safe digital twin (DT2) as a safety-oriented
controller/reference model.

Unlike DT1, which is primarily evaluated as a prediction model, DT2 is
evaluated on:
    1. physiological/control tracking
    2. safe insulin recommendations
    3. constraint compliance
    4. hazard avoidance
    5. response latency
    6. stability and unnecessary intervention
    7. agreement with the safety reference

The module is designed to work with simulation outputs such as:
    - glucose / CGM
    - recommended insulin
    - actual insulin delivery
    - safe/reference insulin
    - hazard score/state
    - safety constraints
    - controller actions

Important
---------
These are research/engineering metrics. They do not establish clinical
safety or clinical effectiveness.

Recommended evaluation philosophy
----------------------------------
DT2 should not simply be judged by glucose prediction accuracy. A good DT2
should also:
    - avoid unsafe insulin recommendations
    - respect safety constraints
    - reduce hazardous excursions
    - react quickly to dangerous changes
    - avoid excessive unnecessary interventions
    - remain stable under attack/fault scenarios
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import math

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _as_array(values):
    """Convert a sequence to a 1-D float array."""
    if np is not None:
        arr = np.asarray(
            list(values) if not hasattr(values, "shape") else values,
            dtype=float,
        )
        return arr.reshape(-1)

    return [float(v) for v in values]


def _validate_pair(a, b):
    if len(a) != len(b):
        raise ValueError(
            f"Inputs must have equal length: {len(a)} != {len(b)}"
        )

    if len(a) == 0:
        raise ValueError("At least one sample is required.")

    if np is not None:
        mask = np.isfinite(a) & np.isfinite(b)
        a = a[mask]
        b = b[mask]
        if len(a) == 0:
            raise ValueError("No finite paired samples remain.")

    return a, b


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, value))


def _safe_mean(values) -> float:
    if np is not None:
        values = np.asarray(values, dtype=float)
        return float(np.mean(values)) if len(values) else float("nan")
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def _safe_std(values) -> float:
    if np is not None:
        values = np.asarray(values, dtype=float)
        return float(np.std(values)) if len(values) else float("nan")
    values = list(values)
    if not values:
        return float("nan")
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class DT2Metrics:
    """Complete DT2 evaluation report."""

    n: int

    # Physiological/control metrics.
    glucose_mae: float
    glucose_rmse: float
    glucose_bias: float
    glucose_time_in_range: float
    glucose_time_below_range: float
    glucose_time_above_range: float

    # Insulin safety/control metrics.
    insulin_mae: float
    insulin_rmse: float
    insulin_bias: float
    insulin_overdelivery_rate: float
    insulin_underdelivery_rate: float

    # Safety behavior.
    constraint_compliance_rate: float
    hazard_free_rate: float
    hazard_event_rate: float
    intervention_rate: float
    unnecessary_intervention_rate: float

    # Response/stability.
    mean_response_latency: float
    max_response_latency: float
    stability_score: float

    # DT2/reference agreement.
    safe_action_agreement: float

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics."""
        return {
            "n": self.n,
            "glucose_mae": self.glucose_mae,
            "glucose_rmse": self.glucose_rmse,
            "glucose_bias": self.glucose_bias,
            "glucose_time_in_range": self.glucose_time_in_range,
            "glucose_time_below_range": self.glucose_time_below_range,
            "glucose_time_above_range": self.glucose_time_above_range,
            "insulin_mae": self.insulin_mae,
            "insulin_rmse": self.insulin_rmse,
            "insulin_bias": self.insulin_bias,
            "insulin_overdelivery_rate": self.insulin_overdelivery_rate,
            "insulin_underdelivery_rate": self.insulin_underdelivery_rate,
            "constraint_compliance_rate": self.constraint_compliance_rate,
            "hazard_free_rate": self.hazard_free_rate,
            "hazard_event_rate": self.hazard_event_rate,
            "intervention_rate": self.intervention_rate,
            "unnecessary_intervention_rate": self.unnecessary_intervention_rate,
            "mean_response_latency": self.mean_response_latency,
            "max_response_latency": self.max_response_latency,
            "stability_score": self.stability_score,
            "safe_action_agreement": self.safe_action_agreement,
            "metadata": dict(self.metadata),
        }


@dataclass
class DT2Thresholds:
    """Safety/evaluation thresholds used by the metric functions."""

    low_glucose_mgdl: float = 70.0
    high_glucose_mgdl: float = 180.0

    # Severe zones used for hazard-event evaluation.
    severe_low_glucose_mgdl: float = 54.0
    severe_high_glucose_mgdl: float = 250.0

    insulin_tolerance: float = 0.05

    # Default response-time threshold in simulation samples.
    max_acceptable_latency_steps: int = 3

    # Oscillation/stability threshold.
    stability_error_threshold: float = 0.10


# ---------------------------------------------------------------------------
# Glucose metrics
# ---------------------------------------------------------------------------

def glucose_tracking_metrics(
    observed_glucose,
    reference_glucose,
) -> Dict[str, float]:
    """
    Evaluate DT2 glucose trajectory against a reference trajectory.

    Reference can be:
        - clean SimGlucose trajectory
        - expected/reference patient trajectory
        - another validated simulation baseline
    """
    observed_glucose, reference_glucose = _validate_pair(
        _as_array(observed_glucose),
        _as_array(reference_glucose),
    )

    error = observed_glucose - reference_glucose

    if np is not None:
        return {
            "mae": float(np.mean(np.abs(error))),
            "mse": float(np.mean(error ** 2)),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(np.mean(error)),
            "std_error": float(np.std(error)),
        }

    return {
        "mae": _safe_mean([abs(x) for x in error]),
        "mse": _safe_mean([x * x for x in error]),
        "rmse": math.sqrt(_safe_mean([x * x for x in error])),
        "bias": _safe_mean(error),
        "std_error": _safe_std(error),
    }


def time_in_glucose_range(
    glucose,
    *,
    low: float = 70.0,
    high: float = 180.0,
) -> Dict[str, float]:
    """Calculate fractions of observations in, below, and above target range."""
    glucose = _as_array(glucose)

    if np is not None:
        n = len(glucose)
        if n == 0:
            return {
                "time_in_range": float("nan"),
                "time_below_range": float("nan"),
                "time_above_range": float("nan"),
            }

        return {
            "time_in_range": float(
                np.mean((glucose >= low) & (glucose <= high))
            ),
            "time_below_range": float(np.mean(glucose < low)),
            "time_above_range": float(np.mean(glucose > high)),
        }

    n = len(glucose)
    return {
        "time_in_range": sum(low <= x <= high for x in glucose) / n,
        "time_below_range": sum(x < low for x in glucose) / n,
        "time_above_range": sum(x > high for x in glucose) / n,
    }


# ---------------------------------------------------------------------------
# Insulin metrics
# ---------------------------------------------------------------------------

def insulin_tracking_metrics(
    delivered_insulin,
    reference_insulin,
) -> Dict[str, float]:
    """Evaluate actual DT2 insulin delivery against a reference."""
    delivered_insulin, reference_insulin = _validate_pair(
        _as_array(delivered_insulin),
        _as_array(reference_insulin),
    )

    error = delivered_insulin - reference_insulin

    if np is not None:
        return {
            "mae": float(np.mean(np.abs(error))),
            "mse": float(np.mean(error ** 2)),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(np.mean(error)),
        }

    return {
        "mae": _safe_mean([abs(x) for x in error]),
        "mse": _safe_mean([x * x for x in error]),
        "rmse": math.sqrt(_safe_mean([x * x for x in error])),
        "bias": _safe_mean(error),
    }


def insulin_delivery_safety_metrics(
    commanded_insulin,
    actual_insulin,
    *,
    tolerance: float = 0.05,
) -> Dict[str, float]:
    """
    Measure delivery mismatch.

    overdelivery:
        actual > command + tolerance

    underdelivery:
        actual < command - tolerance
    """
    commanded_insulin, actual_insulin = _validate_pair(
        _as_array(commanded_insulin),
        _as_array(actual_insulin),
    )

    error = actual_insulin - commanded_insulin

    if np is not None:
        return {
            "overdelivery_rate": float(np.mean(error > tolerance)),
            "underdelivery_rate": float(np.mean(error < -tolerance)),
            "mean_delivery_residual": float(np.mean(error)),
            "delivery_residual_std": float(np.std(error)),
        }

    return {
        "overdelivery_rate": sum(x > tolerance for x in error) / len(error),
        "underdelivery_rate": sum(x < -tolerance for x in error) / len(error),
        "mean_delivery_residual": _safe_mean(error),
        "delivery_residual_std": _safe_std(error),
    }


# ---------------------------------------------------------------------------
# Safety constraint metrics
# ---------------------------------------------------------------------------

def constraint_compliance_rate(
    constraint_results: Sequence[Any],
) -> float:
    """
    Compute fraction of evaluated constraints that are satisfied.

    Accepted item formats:
        - bool
        - objects with `.satisfied`
        - dictionaries with `satisfied`
        - dictionaries with `status` equal to "satisfied"
    """
    if not constraint_results:
        return float("nan")

    satisfied = 0

    for item in constraint_results:
        if isinstance(item, bool):
            satisfied += int(item)
            continue

        if isinstance(item, Mapping):
            if "satisfied" in item:
                satisfied += int(bool(item["satisfied"]))
            elif str(item.get("status", "")).lower() == "satisfied":
                satisfied += 1
            continue

        if hasattr(item, "satisfied"):
            satisfied += int(bool(item.satisfied))
            continue

        if hasattr(item, "status"):
            satisfied += int(
                str(item.status).lower().endswith("satisfied")
            )

    return float(satisfied / len(constraint_results))


# ---------------------------------------------------------------------------
# Hazard metrics
# ---------------------------------------------------------------------------

def hazard_metrics(
    hazard_scores,
    *,
    warning_threshold: float = 0.30,
    active_threshold: float = 0.60,
    critical_threshold: float = 0.85,
) -> Dict[str, float]:
    """
    Evaluate hazard-score behavior.

    `hazard_free_rate` is the fraction below warning threshold.
    `hazard_event_rate` is the fraction at or above active threshold.
    """
    hazard_scores = _as_array(hazard_scores)

    if len(hazard_scores) == 0:
        return {
            "hazard_free_rate": float("nan"),
            "warning_rate": float("nan"),
            "active_rate": float("nan"),
            "critical_rate": float("nan"),
            "hazard_event_rate": float("nan"),
        }

    if np is not None:
        return {
            "hazard_free_rate": float(np.mean(hazard_scores < warning_threshold)),
            "warning_rate": float(
                np.mean(
                    (hazard_scores >= warning_threshold)
                    & (hazard_scores < active_threshold)
                )
            ),
            "active_rate": float(
                np.mean(
                    (hazard_scores >= active_threshold)
                    & (hazard_scores < critical_threshold)
                )
            ),
            "critical_rate": float(np.mean(hazard_scores >= critical_threshold)),
            "hazard_event_rate": float(np.mean(hazard_scores >= active_threshold)),
        }

    return {
        "hazard_free_rate": sum(x < warning_threshold for x in hazard_scores) / len(hazard_scores),
        "warning_rate": sum(
            warning_threshold <= x < active_threshold for x in hazard_scores
        ) / len(hazard_scores),
        "active_rate": sum(
            active_threshold <= x < critical_threshold for x in hazard_scores
        ) / len(hazard_scores),
        "critical_rate": sum(x >= critical_threshold for x in hazard_scores) / len(hazard_scores),
        "hazard_event_rate": sum(x >= active_threshold for x in hazard_scores) / len(hazard_scores),
    }


# ---------------------------------------------------------------------------
# Intervention metrics
# ---------------------------------------------------------------------------

def intervention_metrics(
    actions: Sequence[Any],
    *,
    intervention_values: Optional[Sequence[Any]] = None,
    safe_action_values: Optional[Sequence[Any]] = None,
) -> Dict[str, float]:
    """
    Measure intervention frequency.

    If `intervention_values` is supplied, actions equal to one of those
    values are counted as interventions.

    Otherwise, non-zero numeric actions are treated as interventions.
    """
    if not actions:
        return {
            "intervention_rate": float("nan"),
            "intervention_count": 0.0,
        }

    if intervention_values is not None:
        allowed = set(intervention_values)
        mask = [a in allowed for a in actions]
    else:
        mask = []
        for action in actions:
            try:
                mask.append(abs(float(action)) > 1e-12)
            except (TypeError, ValueError):
                text = str(action).lower()
                mask.append(
                    text not in {
                        "none",
                        "normal",
                        "allow",
                        "no_action",
                        "0",
                    }
                )

    count = sum(mask)

    result = {
        "intervention_rate": float(count / len(actions)),
        "intervention_count": float(count),
    }

    if safe_action_values is not None:
        safe = set(safe_action_values)
        unnecessary = sum(
            bool(is_intervention) and action not in safe
            for action, is_intervention in zip(actions, mask)
        )
        result["unnecessary_intervention_rate"] = float(
            unnecessary / max(count, 1)
        )
    else:
        result["unnecessary_intervention_rate"] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Response latency
# ---------------------------------------------------------------------------

def response_latency_metrics(
    hazard_active,
    intervention_active,
    *,
    max_latency_steps: int = 3,
) -> Dict[str, float]:
    """
    Measure response latency from hazard onset to intervention.

    Each hazard episode begins when `hazard_active` changes False -> True.
    The first subsequent intervention is used as the response point.

    Latency is measured in samples/steps.
    """
    hazard = [bool(x) for x in hazard_active]
    intervention = [bool(x) for x in intervention_active]

    if len(hazard) != len(intervention):
        raise ValueError("hazard_active and intervention_active must match.")

    latencies = []
    active = False
    onset = None

    for i, is_hazard in enumerate(hazard):
        if is_hazard and not active:
            active = True
            onset = i

        if active and intervention[i]:
            latencies.append(i - onset)
            active = False
            onset = None

        if active and not is_hazard:
            active = False
            onset = None

    if not latencies:
        return {
            "mean_response_latency": float("nan"),
            "max_response_latency": float("nan"),
            "response_within_threshold_rate": float("nan"),
            "response_count": 0.0,
        }

    if np is not None:
        mean_latency = float(np.mean(latencies))
        max_latency = float(np.max(latencies))
        within = float(np.mean(np.asarray(latencies) <= max_latency_steps))
    else:
        mean_latency = _safe_mean(latencies)
        max_latency = float(max(latencies))
        within = sum(x <= max_latency_steps for x in latencies) / len(latencies)

    return {
        "mean_response_latency": mean_latency,
        "max_response_latency": max_latency,
        "response_within_threshold_rate": within,
        "response_count": float(len(latencies)),
    }


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def stability_score(
    glucose,
    *,
    target: float = 110.0,
    error_scale: float = 50.0,
) -> float:
    """
    Simple bounded glucose-stability score.

    1.0 = glucose remains close to target.
    0.0 = large deviations.

    This is an engineering indicator, not a clinical stability metric.
    """
    glucose = _as_array(glucose)

    if len(glucose) == 0:
        return float("nan")

    errors = [abs(float(x) - target) for x in glucose]

    if np is not None:
        normalized = np.clip(np.asarray(errors) / max(error_scale, 1e-9), 0.0, 1.0)
        return float(1.0 - np.mean(normalized))

    normalized = [
        min(1.0, x / max(error_scale, 1e-9))
        for x in errors
    ]
    return float(1.0 - sum(normalized) / len(normalized))


# ---------------------------------------------------------------------------
# Safe action agreement
# ---------------------------------------------------------------------------

def safe_action_agreement(
    dt2_actions,
    reference_actions,
) -> float:
    """Fraction of DT2 actions matching the trusted reference actions."""
    if len(dt2_actions) != len(reference_actions):
        raise ValueError("Action sequences must have equal length.")

    if not dt2_actions:
        return float("nan")

    return float(
        sum(a == b for a, b in zip(dt2_actions, reference_actions))
        / len(dt2_actions)
    )


# ---------------------------------------------------------------------------
# Complete report
# ---------------------------------------------------------------------------

def compute_metrics(
    *,
    glucose=None,
    reference_glucose=None,
    delivered_insulin=None,
    reference_insulin=None,
    commanded_insulin=None,
    actual_insulin=None,
    constraint_results=None,
    hazard_scores=None,
    intervention_actions=None,
    hazard_active=None,
    intervention_active=None,
    dt2_actions=None,
    reference_actions=None,
    thresholds: Optional[DT2Thresholds] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> DT2Metrics:
    """
    Compute a complete DT2 evaluation report.

    Only available inputs are evaluated. Missing metric groups are returned
    as NaN rather than being fabricated.
    """
    thresholds = thresholds or DT2Thresholds()

    # Determine report length from the first available time series.
    candidates = [
        glucose,
        reference_glucose,
        delivered_insulin,
        reference_insulin,
        commanded_insulin,
        actual_insulin,
        hazard_scores,
    ]

    n = 0
    for values in candidates:
        if values is not None:
            n = len(values)
            break

    if n == 0:
        for values in (
            constraint_results,
            intervention_actions,
            hazard_active,
            intervention_active,
            dt2_actions,
            reference_actions,
        ):
            if values is not None:
                n = len(values)
                break

    glucose_report = {
        "mae": float("nan"),
        "rmse": float("nan"),
        "bias": float("nan"),
    }
    glucose_range = {
        "time_in_range": float("nan"),
        "time_below_range": float("nan"),
        "time_above_range": float("nan"),
    }

    if glucose is not None and reference_glucose is not None:
        glucose_report = glucose_tracking_metrics(
            glucose,
            reference_glucose,
        )

    if glucose is not None:
        glucose_range = time_in_glucose_range(
            glucose,
            low=thresholds.low_glucose_mgdl,
            high=thresholds.high_glucose_mgdl,
        )

    insulin_report = {
        "mae": float("nan"),
        "rmse": float("nan"),
        "bias": float("nan"),
    }

    if delivered_insulin is not None and reference_insulin is not None:
        insulin_report = insulin_tracking_metrics(
            delivered_insulin,
            reference_insulin,
        )

    delivery_report = {
        "overdelivery_rate": float("nan"),
        "underdelivery_rate": float("nan"),
    }

    if commanded_insulin is not None and actual_insulin is not None:
        delivery_report = insulin_delivery_safety_metrics(
            commanded_insulin,
            actual_insulin,
            tolerance=thresholds.insulin_tolerance,
        )

    compliance = float("nan")
    if constraint_results is not None:
        compliance = constraint_compliance_rate(constraint_results)

    hazard_report = {
        "hazard_free_rate": float("nan"),
        "hazard_event_rate": float("nan"),
    }

    if hazard_scores is not None:
        hazard_report = hazard_metrics(hazard_scores)

    interventions = {
        "intervention_rate": float("nan"),
        "unnecessary_intervention_rate": float("nan"),
    }

    if intervention_actions is not None:
        interventions = intervention_metrics(intervention_actions)

    latency = {
        "mean_response_latency": float("nan"),
        "max_response_latency": float("nan"),
    }

    if hazard_active is not None and intervention_active is not None:
        latency = response_latency_metrics(
            hazard_active,
            intervention_active,
            max_latency_steps=thresholds.max_acceptable_latency_steps,
        )

    stability = float("nan")
    if glucose is not None:
        stability = globals()["stability_score"](glucose)

    agreement = float("nan")
    if dt2_actions is not None and reference_actions is not None:
        agreement = safe_action_agreement(
            dt2_actions,
            reference_actions,
        )

    return DT2Metrics(
        n=n,
        glucose_mae=glucose_report["mae"],
        glucose_rmse=glucose_report["rmse"],
        glucose_bias=glucose_report["bias"],
        glucose_time_in_range=glucose_range["time_in_range"],
        glucose_time_below_range=glucose_range["time_below_range"],
        glucose_time_above_range=glucose_range["time_above_range"],
        insulin_mae=insulin_report["mae"],
        insulin_rmse=insulin_report["rmse"],
        insulin_bias=insulin_report["bias"],
        insulin_overdelivery_rate=delivery_report["overdelivery_rate"],
        insulin_underdelivery_rate=delivery_report["underdelivery_rate"],
        constraint_compliance_rate=compliance,
        hazard_free_rate=hazard_report["hazard_free_rate"],
        hazard_event_rate=hazard_report["hazard_event_rate"],
        intervention_rate=interventions["intervention_rate"],
        unnecessary_intervention_rate=interventions[
            "unnecessary_intervention_rate"
        ],
        mean_response_latency=latency["mean_response_latency"],
        max_response_latency=latency["max_response_latency"],
        stability_score=stability,
        safe_action_agreement=agreement,
        metadata={
            "low_glucose_mgdl": thresholds.low_glucose_mgdl,
            "high_glucose_mgdl": thresholds.high_glucose_mgdl,
            "severe_low_glucose_mgdl": thresholds.severe_low_glucose_mgdl,
            "severe_high_glucose_mgdl": thresholds.severe_high_glucose_mgdl,
            **dict(metadata or {}),
        },
    )


__all__ = [
    "DT2Metrics",
    "DT2Thresholds",
    "glucose_tracking_metrics",
    "time_in_glucose_range",
    "insulin_tracking_metrics",
    "insulin_delivery_safety_metrics",
    "constraint_compliance_rate",
    "hazard_metrics",
    "intervention_metrics",
    "response_latency_metrics",
    "stability_score",
    "safe_action_agreement",
    "compute_metrics",
]
