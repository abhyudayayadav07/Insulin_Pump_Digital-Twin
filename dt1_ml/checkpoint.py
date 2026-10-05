"""
checkpoint.py

Checkpoint management for DT1 models.

Stores:
    - model state
    - optimizer state when supplied
    - epoch
    - metrics
    - model/training configuration

This keeps trained DT1 weights separate from the model architecture code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[dict[str, Any]] = None,
    config: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Save a DT1 training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
        "extra": extra or {},
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, path)

    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load a checkpoint.

    If model is supplied, its weights are restored.
    If optimizer is supplied and optimizer state exists, it is restored.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if model is not None:
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=strict,
        )
        model.to(device)

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint


def checkpoint_info(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Read checkpoint metadata without requiring a model instance."""
    checkpoint = load_checkpoint(
        path,
        device=device,
    )

    return {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
        "extra": checkpoint.get("extra", {}),
    }


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "checkpoint_info",
]
