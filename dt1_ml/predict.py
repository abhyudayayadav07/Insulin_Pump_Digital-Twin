"""
predict.py

Prediction utilities for DT1 models.

This module performs model prediction but does not decide whether a
prediction is anomalous. Residual calculation and thresholding are handled
by residual.py and threshold.py respectively.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def predict_tensor(
    model: nn.Module,
    x: torch.Tensor | np.ndarray,
    *,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Predict glucose from a tensor/NumPy sequence batch."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    else:
        x = x.float()

    x = x.to(device)

    output = model(x)

    return output.detach().cpu().numpy()


def predict_sequence(
    model: nn.Module,
    sequence: np.ndarray | torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> float | np.ndarray:
    """Predict from one temporal sequence."""
    if isinstance(sequence, np.ndarray):
        sequence = torch.from_numpy(sequence).float()

    sequence = sequence.float()

    if sequence.ndim == 2:
        sequence = sequence.unsqueeze(0)

    prediction = predict_tensor(
        model,
        sequence,
        device=device,
    )

    if prediction.size == 1:
        return float(prediction.reshape(-1)[0])

    return prediction


def predict_dataframe(
    model: nn.Module,
    X: np.ndarray,
    *,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Convenience wrapper for pre-built sequence arrays."""
    return predict_tensor(
        model,
        X,
        device=device,
    )


__all__ = [
    "predict_tensor",
    "predict_sequence",
    "predict_dataframe",
]
