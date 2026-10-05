"""
Reachability-consistency losses for DT2.

The reachability loss penalizes predictions that fall outside a supplied
reachable-state envelope. This provides a lightweight implementation of
the knowledge-guided idea:

    data fit + consistency with physically/safely reachable states

The reachable envelope may come from the DT2 reachability module, a
simulator, or a domain-specific uncertainty model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ReachabilityLossConfig:
    """Configuration for reachable-set penalties."""

    weight: float = 1.0
    boundary_margin: float = 0.0
    normalize_by_width: bool = True
    minimum_width: float = 1e-6

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("weight must be non-negative.")
        if self.boundary_margin < 0:
            raise ValueError("boundary_margin must be non-negative.")
        if self.minimum_width <= 0:
            raise ValueError("minimum_width must be positive.")


def reachable_violation(
    predictions: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    config: Optional[ReachabilityLossConfig] = None,
) -> np.ndarray:
    """Return per-element normalized distance outside the reachable set."""
    config = config or ReachabilityLossConfig()

    pred = np.asarray(predictions, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)

    if pred.shape != lo.shape or pred.shape != hi.shape:
        raise ValueError("predictions, lower, and upper must have equal shapes.")
    if not np.isfinite(pred).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("Reachability inputs must be finite.")
    if np.any(lo > hi):
        raise ValueError("lower cannot exceed upper.")

    width = np.maximum(hi - lo, config.minimum_width)
    effective_lo = lo - config.boundary_margin
    effective_hi = hi + config.boundary_margin

    below = np.maximum(effective_lo - pred, 0.0)
    above = np.maximum(pred - effective_hi, 0.0)
    violation = below + above

    if config.normalize_by_width:
        violation = violation / width

    return violation


def reachability_loss(
    predictions: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    config: Optional[ReachabilityLossConfig] = None,
) -> float:
    """Compute mean reachability violation penalty."""
    config = config or ReachabilityLossConfig()
    violation = reachable_violation(
        predictions,
        lower,
        upper,
        config=config,
    )
    return float(config.weight * np.mean(violation))


def reachable_fraction(
    predictions: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> float:
    """Return the fraction of predictions inside their reachable bounds."""
    pred = np.asarray(predictions, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)

    if pred.shape != lo.shape or pred.shape != hi.shape:
        raise ValueError("All arrays must have equal shapes.")

    inside = (pred >= lo) & (pred <= hi)
    return float(np.mean(inside))


__all__ = [
    "ReachabilityLossConfig",
    "reachable_violation",
    "reachability_loss",
    "reachable_fraction",
]
