"""
Combined knowledge-guided loss for DT2.

This module combines:
    1. ordinary prediction error,
    2. safety-constraint loss,
    3. reachability-consistency loss,
    4. hazard-region loss.

Conceptually:

    L_total =
        eta * L_data
        + w_safety * L_safety
        + w_reachability * L_reachability
        + w_hazard * L_hazard

This is the central loss-composition layer for the
KnowSafe-inspired DT2 prediction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .hazard_loss import HazardLossConfig, hazard_loss
from .reachability_loss import ReachabilityLossConfig, reachability_loss
from .safety_loss import SafetyLossConfig, safety_violation_penalty


@dataclass(frozen=True)
class KnowledgeGuidedLossConfig:
    """Weights for the combined knowledge-guided objective."""

    data_weight: float = 0.50
    safety_weight: float = 0.20
    reachability_weight: float = 0.20
    hazard_weight: float = 0.10
    normalize_components: bool = True

    def __post_init__(self) -> None:
        weights = (
            self.data_weight,
            self.safety_weight,
            self.reachability_weight,
            self.hazard_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Loss weights must be non-negative.")
        if sum(weights) <= 0:
            raise ValueError("At least one loss weight must be positive.")


@dataclass
class KnowledgeGuidedLossResult:
    """Breakdown of the combined loss."""

    total: float
    data_loss: float
    safety_loss: float
    reachability_loss: float
    hazard_loss: float

    def as_dict(self):
        return {
            "total": self.total,
            "data_loss": self.data_loss,
            "safety_loss": self.safety_loss,
            "reachability_loss": self.reachability_loss,
            "hazard_loss": self.hazard_loss,
        }


def mse_loss(
    predictions: Sequence[float],
    targets: Sequence[float],
) -> float:
    """Compute mean squared prediction error."""
    pred = np.asarray(predictions, dtype=float)
    target = np.asarray(targets, dtype=float)

    if pred.shape != target.shape:
        raise ValueError("predictions and targets must have equal shapes.")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("predictions and targets must be finite.")

    return float(np.mean((pred - target) ** 2))


def knowledge_guided_loss(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    reachable_lower: Optional[Sequence[float]] = None,
    reachable_upper: Optional[Sequence[float]] = None,
    previous_glucose: Optional[float] = None,
    config: Optional[KnowledgeGuidedLossConfig] = None,
    safety_config: Optional[SafetyLossConfig] = None,
    reachability_config: Optional[ReachabilityLossConfig] = None,
    hazard_config: Optional[HazardLossConfig] = None,
) -> KnowledgeGuidedLossResult:
    """
    Calculate the complete knowledge-guided loss.

    Reachability loss is included only when both reachable bounds are
    supplied.
    """
    config = config or KnowledgeGuidedLossConfig()

    data = mse_loss(predictions, targets)
    safety = safety_violation_penalty(
        predictions,
        config=safety_config,
    )
    hazard = hazard_loss(
        predictions,
        previous_glucose=previous_glucose,
        config=hazard_config,
    )

    if reachable_lower is not None and reachable_upper is not None:
        reachability = reachability_loss(
            predictions,
            reachable_lower,
            reachable_upper,
            config=reachability_config,
        )
    else:
        reachability = 0.0

    if config.normalize_components:
        # Keep the knowledge terms numerically comparable to the data term.
        data_norm = data / max(1.0, data)
        safety_norm = safety / max(1.0, safety)
        reach_norm = reachability / max(1.0, reachability)
        hazard_norm = hazard / max(1.0, hazard)
    else:
        data_norm = data
        safety_norm = safety
        reach_norm = reachability
        hazard_norm = hazard

    total = (
        config.data_weight * data_norm
        + config.safety_weight * safety_norm
        + config.reachability_weight * reach_norm
        + config.hazard_weight * hazard_norm
    )

    return KnowledgeGuidedLossResult(
        total=float(total),
        data_loss=float(data),
        safety_loss=float(safety),
        reachability_loss=float(reachability),
        hazard_loss=float(hazard),
    )


__all__ = [
    "KnowledgeGuidedLossConfig",
    "KnowledgeGuidedLossResult",
    "mse_loss",
    "knowledge_guided_loss",
]
