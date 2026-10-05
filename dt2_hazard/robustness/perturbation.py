"""Generic perturbation primitives for robustness testing."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class PerturbationType(str, Enum):
    BIAS = "bias"
    NOISE = "noise"
    SCALE = "scale"
    DROPOUT = "dropout"
    DELAY = "delay"
    DRIFT = "drift"
    SPIKE = "spike"


@dataclass(frozen=True)
class Perturbation:
    perturbation_type: PerturbationType
    magnitude: float
    probability: float = 1.0
    seed: Optional[int] = None


def apply_perturbation(values, perturbation: Perturbation):
    x = np.asarray(values, dtype=float).copy()
    rng = np.random.default_rng(perturbation.seed)
    m = float(perturbation.magnitude)
    p = max(0.0, min(1.0, float(perturbation.probability)))

    if perturbation.perturbation_type == PerturbationType.BIAS:
        return x + m
    if perturbation.perturbation_type == PerturbationType.NOISE:
        return x + rng.normal(0.0, abs(m), size=x.shape)
    if perturbation.perturbation_type == PerturbationType.SCALE:
        return x * (1.0 + m)
    if perturbation.perturbation_type == PerturbationType.DROPOUT:
        mask = rng.random(x.shape) < p
        out = x.copy()
        out[mask] = np.nan
        return out
    if perturbation.perturbation_type == PerturbationType.DELAY:
        shift = max(0, int(round(m)))
        if shift == 0:
            return x
        out = np.empty_like(x)
        if shift >= len(x):
            out[:] = x[0] if len(x) else 0.0
        else:
            out[:shift] = x[0]
            out[shift:] = x[:-shift]
        return out
    if perturbation.perturbation_type == PerturbationType.DRIFT:
        return x + np.linspace(0.0, m, len(x))
    if perturbation.perturbation_type == PerturbationType.SPIKE:
        out = x.copy()
        if len(out):
            indices = rng.random(len(out)) < p
            out[indices] += m
        return out

    raise ValueError(f"Unsupported perturbation: {perturbation.perturbation_type}")
