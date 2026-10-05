"""
Visualization utilities for the insulin digital twin evaluation pipeline.

Provides plots for:
- DT1 predictive performance and residuals
- Model comparison
- DT2 glucose/insulin tracking
- Hazard and safety state
- DT1 vs DT2 comparison
- Safety constraint compliance
- Residual/error analysis
- Metric summaries

All plotting functions return a matplotlib Figure and optionally save it.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FIGSIZE = (10, 5)
DEFAULT_DPI = 150


def _prepare_x(x=None, n=None, label="Index"):
    if x is None:
        if n is None:
            raise ValueError("Either x or n must be provided.")
        return np.arange(n), label
    x = np.asarray(x)
    return x, label


def _save_figure(fig, save_path=None, dpi=DEFAULT_DPI):
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def _validate_same_length(*arrays):
    lengths = [len(np.asarray(a)) for a in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"All input arrays must have the same length. Got {lengths}.")


def _add_glucose_zones(ax, low=70.0, high=180.0):
    ax.axhline(low, linestyle="--", linewidth=1)
    ax.axhline(high, linestyle="--", linewidth=1)


def plot_dt1_predictions(
    actual,
    predicted,
    x=None,
    title="DT1 Glucose Prediction",
    xlabel="Time",
    ylabel="Glucose (mg/dL)",
    save_path=None,
    figsize=DEFAULT_FIGSIZE,
):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    _validate_same_length(actual, predicted)
    x, _ = _prepare_x(x, len(actual), xlabel)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, actual, label="Actual")
    ax.plot(x, predicted, label="Predicted")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, save_path)


def plot_dt1_residuals(
    residuals,
    x=None,
    threshold=None,
    title="DT1 Prediction Residuals",
    xlabel="Time",
    ylabel="Residual (mg/dL)",
    save_path=None,
    figsize=DEFAULT_FIGSIZE,
):
    residuals = np.asarray(residuals)
    x, _ = _prepare_x(x, len(residuals), xlabel)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, residuals, label="Residual")
    ax.axhline(0.0, linestyle="--", linewidth=1)
    if threshold is not None:
        ax.axhline(threshold, linestyle=":", linewidth=1, label="Threshold")
        ax.axhline(-threshold, linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, save_path)


def plot_error_distribution(
    errors,
    bins=40,
    title="Prediction Error Distribution",
    xlabel="Error",
    save_path=None,
    figsize=DEFAULT_FIGSIZE,
):
    errors = np.asarray(errors)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(errors, bins=bins)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, save_path)


def plot_model_comparison(
    model_metrics: Mapping[str, Mapping[str, float]],
    metric="mae",
    title=None,
    save_path=None,
    figsize=DEFAULT_FIGSIZE,
):
    names = list(model_metrics.keys())
    values = [model_metrics[name].get(metric, np.nan) for name in names]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(names, values)
    ax.set_title(title or f"Model Comparison: {metric.upper()}")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, save_path)


def plot_dt2_glucose(
    actual,
    dt2_predicted=None,
    x=None,
    low=70.0,
    high=180.0,
    title="DT2 Reactive-Safe Twin Glucose Tracking",
    save_path=None,
    figsize=DEFAULT_FIGSIZE,
):
    actual = np.asarray(actual)
    series = [actual]
    labels = ["Actual"]
    if dt2_predicted is not None:
        dt2_predicted = np.asarray(dt2_predicted)
        _validate_same_length(actual, dt2_predicted)
        series.append(dt2_predicted)
        labels.append("DT2")

    x, _ = _prepare_x(x, len(actual), "Time")
    fig, ax = plt.subplots(figsize=figsize)

    for values, label in zip(series, labels):
        ax.plot(x, values, label=label)

    _add_glucose_zones(ax, low, high)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, save_path)


def plot_insulin_delivery(
    commanded,
    actual,
    x=None,
    title="Insulin Command vs Actual Delivery",
    save_path=None,
    figsize=DEFAULT_FIGSIZE,
):
    commanded = np.asarray(commanded)
    actual = np.asarray(actual)
    _validate_same_length(commanded, actual)
    x, _ = _prepare_x(x, len(commanded), "Time")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, commanded, label="Commanded")
    ax.plot(x, actual, label="Actual")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Insulin")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, save_path)


def plot_hazard_score(
    score,
    x=None,
    warning=0.30,
    active=0.60,
    critical=0.85,
    title="Hazard Score",
    save_path=None,
    figsize=DEFAULT_FIGSIZE,
):
    score = np.asarray(score)
    x, _ = _prepare_x(x, len(score), "Time")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, score, label="Hazard score")
    for value, name in (
        (warning, "Warning"),
        (active, "Active"),
        (critical, "Critical"),
    ):
        ax.axhline(value, linestyle="--", linewidth=1, label=name)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Hazard score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, save_path)


def plot_safety_state(
    states: Sequence[Any],
    x=None,
    title="Safety State Over Time",
    save_path=None,
    figsize=DEFAULT_FIGSIZE,
):
    states = np.asarray(states)
    x, _ = _prepare_x(x, len(states), "Time")

    unique = list(dict.fromkeys(states.tolist()))
    mapping = {state: i for i, state in enumerate(unique)}
    numeric = [mapping[state] for state in states]

    fig, ax = plt.subplots(figsize=figsize)
    ax.step(x, numeric, where="post")
    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels([str(s) for s in unique])
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Safety State")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, save_path)


def plot_dt1_dt2_comparison(
    actual,
    dt1_predicted,
    dt2_predicted,
    x=None,
    low=70.0,
    high=180.0,
    title="DT1 vs DT2 Glucose Comparison",
    save_path=None,
    figsize=DEFAULT_FIGSIZE,
):
    actual = np.asarray(actual)
    dt1_predicted = np.asarray(dt1_predicted)
    dt2_predicted = np.asarray(dt2_predicted)
    _validate_same_length(actual, dt1_predicted, dt2_predicted)
    x, _ = _prepare_x(x, len(actual), "Time")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, actual, label="Actual")
    ax.plot(x, dt1_predicted, label="DT1 Predictive")
    ax.plot(x, dt2_predicted, label="DT2 Reactive-Safe")
    _add_glucose_zones(ax, low, high)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, save_path)


def plot_constraint_compliance(
    constraint_names: Sequence[str],
    compliance: Sequence[float],
    title="Safety Constraint Compliance",
    save_path=None,
    figsize=(10, 6),
):
    names = list(constraint_names)
    values = np.asarray(compliance, dtype=float)
    if len(names) != len(values):
        raise ValueError("constraint_names and compliance must have the same length.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(names, values)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Compliance")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, save_path)


def plot_residual_scatter(
    actual,
    predicted,
    title="Prediction Residual Analysis",
    save_path=None,
    figsize=DEFAULT_FIGSIZE,
):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    _validate_same_length(actual, predicted)
    residual = predicted - actual

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(actual, residual, alpha=0.6)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Actual Glucose (mg/dL)")
    ax.set_ylabel("Residual (mg/dL)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, save_path)


def plot_metric_summary(
    metrics: Mapping[str, float],
    title="Evaluation Metric Summary",
    save_path=None,
    figsize=(10, 6),
):
    names = list(metrics.keys())
    values = [metrics[name] for name in names]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(names, values)
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, save_path)


def save_all_figures(figures: Mapping[str, Any], output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fig in figures.items():
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight")
        paths[name] = str(path)
    return paths


__all__ = [
    "plot_dt1_predictions",
    "plot_dt1_residuals",
    "plot_error_distribution",
    "plot_model_comparison",
    "plot_dt2_glucose",
    "plot_insulin_delivery",
    "plot_hazard_score",
    "plot_safety_state",
    "plot_dt1_dt2_comparison",
    "plot_constraint_compliance",
    "plot_residual_scatter",
    "plot_metric_summary",
    "save_all_figures",
]
