"""
train.py

Training pipeline for Digital Twin 1 (DT1).

Connects:
    dataset.py -> model_factory.py -> optimization -> checkpoint.py

This module supports LSTM, GRU, and Transformer regression models.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .models.model_factory import create_model


@dataclass
class TrainingConfig:
    model_name: str = "lstm"
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 64
    weight_decay: float = 0.0
    patience: int = 5
    device: str = "auto"
    gradient_clip: Optional[float] = 1.0


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    gradient_clip: Optional[float] = None,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_items = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        if y.ndim == 1:
            y = y.unsqueeze(-1)

        if training:
            optimizer.zero_grad(set_to_none=True)

        prediction = model(x)
        loss = criterion(prediction, y)

        if training:
            loss.backward()

            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip,
                )

            optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    config: Optional[TrainingConfig] = None,
    checkpoint_path: Optional[str | Path] = None,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Train a model and optionally save the best validation checkpoint."""
    config = config or TrainingConfig()
    device = resolve_device(config.device)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = {
        "train_loss": [],
        "validation_loss": [],
    }

    best_validation = float("inf")
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        train_loss = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config.gradient_clip,
        )

        validation_loss = _run_epoch(
            model,
            validation_loader,
            criterion,
            None,
            device,
        )

        history["train_loss"].append(float(train_loss))
        history["validation_loss"].append(float(validation_loss))

        if validation_loss < best_validation:
            best_validation = validation_loss
            epochs_without_improvement = 0

            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    metrics={"validation_loss": validation_loss},
                    config=asdict(config),
                )
        else:
            epochs_without_improvement += 1

        if (
            config.patience > 0
            and epochs_without_improvement >= config.patience
        ):
            break

    return model, history


def build_model(
    model_name: str,
    input_size: int,
    **model_kwargs: Any,
) -> nn.Module:
    """Construct an LSTM, GRU, or Transformer model."""
    return create_model(
        model_name,
        input_size=input_size,
        **model_kwargs,
    )


__all__ = [
    "TrainingConfig",
    "resolve_device",
    "train_model",
    "build_model",
]
