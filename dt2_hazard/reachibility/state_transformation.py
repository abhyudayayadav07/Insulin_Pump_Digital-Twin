"""State transformations used by the DT2 reachability layer."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence
import numpy as np

@dataclass(frozen=True)
class StateTransformationConfig:
    include_raw: bool = True
    include_rate: bool = True
    include_acceleration: bool = True
    include_squared: bool = False
    include_insulin_effect: bool = True
    include_meal_effect: bool = True
    insulin_scale: float = 1.0
    meal_scale: float = 1.0
    def __post_init__(self):
        if self.insulin_scale <= 0 or self.meal_scale <= 0:
            raise ValueError("scales must be positive")

def _rate(x: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0: raise ValueError("dt must be positive")
    return np.gradient(x, dt) if len(x) > 1 else np.zeros_like(x)

def transform_state(state: Sequence[float], *, dt_minutes: float = 5.0,
                    config: Optional[StateTransformationConfig] = None) -> np.ndarray:
    c = config or StateTransformationConfig(); x = np.asarray(state, dtype=float).reshape(-1)
    if not np.isfinite(x).all(): raise ValueError("state contains non-finite values")
    out = []
    if c.include_raw: out.extend(x.tolist())
    if c.include_rate: out.append(0.0)
    if c.include_acceleration: out.append(0.0)
    if c.include_squared: out.extend((x*x).tolist())
    if c.include_insulin_effect: out.append((x[1] if len(x)>1 else 0.0)/c.insulin_scale)
    if c.include_meal_effect: out.append((x[2] if len(x)>2 else 0.0)/c.meal_scale)
    return np.asarray(out, dtype=float)

def transform_trajectory(trajectory: Sequence[Sequence[float]], *, dt_minutes: float = 5.0,
                         config: Optional[StateTransformationConfig] = None) -> np.ndarray:
    c = config or StateTransformationConfig(); x = np.asarray(trajectory, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all(): raise ValueError("trajectory must be finite 2-D")
    out = [x] if c.include_raw else []
    r = _rate(x[:,0], dt_minutes)
    if c.include_rate: out.append(r[:,None])
    if c.include_acceleration: out.append(_rate(r, dt_minutes)[:,None])
    if c.include_squared: out.append(x*x)
    if c.include_insulin_effect: out.append(((x[:,1] if x.shape[1]>1 else np.zeros(len(x)))/c.insulin_scale)[:,None])
    if c.include_meal_effect: out.append(((x[:,2] if x.shape[1]>2 else np.zeros(len(x)))/c.meal_scale)[:,None])
    return np.concatenate(out, axis=1)

def transform_named_state(state: Mapping[str,float], *, config=None):
    c = config or StateTransformationConfig(); out = dict(state)
    out["glucose_rate"] = float(state.get("glucose_rate",0.0)) if c.include_rate else out.get("glucose_rate",0.0)
    out["glucose_acceleration"] = float(state.get("glucose_acceleration",0.0)) if c.include_acceleration else out.get("glucose_acceleration",0.0)
    if c.include_insulin_effect: out["insulin_effect"] = float(state.get("insulin",0.0))/c.insulin_scale
    if c.include_meal_effect: out["meal_effect"] = float(state.get("meal",0.0))/c.meal_scale
    if c.include_squared: out["glucose_squared"] = float(state.get("glucose",0.0))**2
    return out

__all__=["StateTransformationConfig","transform_state","transform_trajectory","transform_named_state"]
