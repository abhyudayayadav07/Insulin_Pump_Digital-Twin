"""
validate.py

Validation and evaluation utilities for DT1 glucose prediction models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate MAE, RMSE, MSE, and error statistics."""
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)

    mask = np.isfinite(actual) & np.isfinite(predicted)

    if not mask.any():
        raise ValueError("No finite prediction/target pairs available.")

    actual = actual[mask]
    predicted = predicted[mask]

    error = predicted - actual

    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error ** 2)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mean_error": float(np.mean(error)),
        "std_error": float(np.std(error)),
        "n": int(len(actual)),
    }


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions and targets from a PyTorch DataLoader."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    predictions = []
    targets = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"]

        prediction = model(x)

        predictions.append(
            prediction.detach().cpu().numpy()
        )
        targets.append(
            y.detach().cpu().numpy()
        )

    if not predictions:
        return np.empty(0), np.empty(0)

    return (
        np.concatenate(predictions).reshape(-1),
        np.concatenate(targets).reshape(-1),
    )


def validate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Evaluate a trained model and return predictions plus metrics."""
    predicted, actual = predict_loader(
        model,
        loader,
        device=device,
    )

    metrics = regression_metrics(actual, predicted)

    return {
        "metrics": metrics,
        "predicted": predicted,
        "actual": actual,
    }


__all__ = [
    "regression_metrics",
    "predict_loader",
    "validate_model",
]
