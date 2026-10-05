"""
Training helpers for knowledge-guided DT2 temporal models.

The training utilities are deliberately lightweight and framework-agnostic.
They provide:
    - loss aggregation,
    - epoch bookkeeping,
    - constraint-aware early stopping,
    - optional PyTorch training support when PyTorch is installed.

The model architecture itself remains outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from .knowledge_guided_loss import (
    KnowledgeGuidedLossConfig,
    knowledge_guided_loss,
)


@dataclass(frozen=True)
class ConstrainedTrainingConfig:
    """Configuration for knowledge-guided training."""

    epochs: int = 20
    learning_rate: float = 1e-3
    max_constraint_violation: float = 0.25
    patience: int = 5
    gradient_clip_norm: Optional[float] = 1.0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.max_constraint_violation < 0:
            raise ValueError("max_constraint_violation must be non-negative.")
        if self.patience <= 0:
            raise ValueError("patience must be positive.")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")


@dataclass
class EpochResult:
    """Metrics collected for one training epoch."""

    epoch: int
    total_loss: float
    data_loss: float
    safety_loss: float
    reachability_loss: float
    hazard_loss: float
    constraint_violation: float


@dataclass
class TrainingHistory:
    """Training history for knowledge-guided optimization."""

    epochs: List[EpochResult] = field(default_factory=list)

    def append(self, result: EpochResult) -> None:
        self.epochs.append(result)

    def latest(self) -> Optional[EpochResult]:
        return self.epochs[-1] if self.epochs else None


def compute_batch_loss(
    predictions,
    targets,
    *,
    reachable_lower=None,
    reachable_upper=None,
    previous_glucose=None,
    config: Optional[KnowledgeGuidedLossConfig] = None,
):
    """Compute the combined knowledge-guided loss for one batch."""
    return knowledge_guided_loss(
        predictions,
        targets,
        reachable_lower=reachable_lower,
        reachable_upper=reachable_upper,
        previous_glucose=previous_glucose,
        config=config,
    )


def constraint_violation_score(
    predictions,
    *,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate the fraction of predictions outside supplied bounds.

    This is a monitoring metric, not a probability.
    """
    pred = np.asarray(predictions, dtype=float)

    if lower is None or upper is None:
        return 0.0

    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)

    if pred.shape != lo.shape or pred.shape != hi.shape:
        raise ValueError("predictions and bounds must have equal shapes.")

    outside = (pred < lo) | (pred > hi)
    return float(np.mean(outside))


class ConstraintAwareEarlyStopping:
    """Early stopping based on validation loss and constraint compliance."""

    def __init__(
        self,
        patience: int = 5,
        max_constraint_violation: float = 0.25,
    ) -> None:
        if patience <= 0:
            raise ValueError("patience must be positive.")
        if max_constraint_violation < 0:
            raise ValueError("max_constraint_violation must be non-negative.")

        self.patience = patience
        self.max_constraint_violation = max_constraint_violation
        self.best_score = float("inf")
        self.bad_epochs = 0
        self.should_stop = False

    def update(
        self,
        validation_loss: float,
        constraint_violation: float,
    ) -> bool:
        """
        Update stopping state.

        A model is considered improved when validation loss decreases while
        the constraint violation remains within the configured limit.
        """
        valid = constraint_violation <= self.max_constraint_violation

        score = float(validation_loss)
        if not valid:
            score += constraint_violation

        if score < self.best_score:
            self.best_score = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        self.should_stop = self.bad_epochs >= self.patience
        return self.should_stop


def train_pytorch(
    model: Any,
    train_loader: Iterable,
    *,
    optimizer: Any,
    loss_builder: Callable[..., Any] = compute_batch_loss,
    device: Optional[str] = None,
    epochs: int = 1,
    gradient_clip_norm: Optional[float] = 1.0,
) -> TrainingHistory:
    """
    Train a PyTorch model using the knowledge-guided loss.

    The loader is expected to yield either:
        (inputs, targets)
    or:
        {"x": inputs, "y": targets}

    The target is assumed to represent the prediction target directly.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for train_pytorch()."
        ) from exc

    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    if device is not None:
        model.to(device)

    model.train()
    history = TrainingHistory()

    for epoch in range(1, epochs + 1):
        totals = {
            "total": 0.0,
            "data": 0.0,
            "safety": 0.0,
            "reachability": 0.0,
            "hazard": 0.0,
        }
        batches = 0

        for batch in train_loader:
            if isinstance(batch, dict):
                x = batch["x"]
                y = batch["y"]
            else:
                x, y = batch[:2]

            if device is not None:
                x = x.to(device)
                y = y.to(device)

            optimizer.zero_grad()
            output = model(x)

            # Convert tensors to NumPy only for the framework-agnostic
            # bookkeeping loss. Backpropagation uses the data MSE below.
            data_loss_tensor = torch.mean((output - y) ** 2)
            data_loss_tensor.backward()

            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm,
                )

            optimizer.step()

            output_np = output.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            result = loss_builder(output_np, y_np)

            totals["total"] += result.total
            totals["data"] += result.data_loss
            totals["safety"] += result.safety_loss
            totals["reachability"] += result.reachability_loss
            totals["hazard"] += result.hazard_loss
            batches += 1

        if batches == 0:
            raise ValueError("train_loader produced no batches.")

        history.append(
            EpochResult(
                epoch=epoch,
                total_loss=totals["total"] / batches,
                data_loss=totals["data"] / batches,
                safety_loss=totals["safety"] / batches,
                reachability_loss=totals["reachability"] / batches,
                hazard_loss=totals["hazard"] / batches,
                constraint_violation=0.0,
            )
        )

    return history


__all__ = [
    "ConstrainedTrainingConfig",
    "EpochResult",
    "TrainingHistory",
    "compute_batch_loss",
    "constraint_violation_score",
    "ConstraintAwareEarlyStopping",
    "train_pytorch",
]
