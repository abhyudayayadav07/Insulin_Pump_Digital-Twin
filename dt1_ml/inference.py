"""
inference.py

Deployment-style DT1 inference wrapper.

Inference combines:
    preprocessing/feature preparation -> sequence input -> trained model
without performing training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from .predict import predict_sequence


@dataclass
class InferenceResult:
    predicted_glucose: float
    model_name: str
    device: str


class DT1Inference:
    """Reusable inference engine for a trained DT1 model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | str = "cpu",
        model_name: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.model_name = (
            model_name
            or model.__class__.__name__
        )

    def predict(
        self,
        sequence: np.ndarray | torch.Tensor,
    ) -> InferenceResult:
        """Predict future glucose for one historical sequence."""
        prediction = predict_sequence(
            self.model,
            sequence,
            device=self.device,
        )

        if isinstance(prediction, np.ndarray):
            value = float(prediction.reshape(-1)[0])
        else:
            value = float(prediction)

        return InferenceResult(
            predicted_glucose=value,
            model_name=self.model_name,
            device=str(self.device),
        )

    def predict_batch(
        self,
        X: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Predict future glucose for multiple sequences."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        X = X.float()

        with torch.no_grad():
            output = self.model(
                X.to(self.device)
            )

        return output.detach().cpu().numpy()


__all__ = [
    "InferenceResult",
    "DT1Inference",
]
