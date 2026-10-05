"""Axis-aligned reachable-set representations for DT2."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np

@dataclass
class ReachableSet:
    lower: np.ndarray
    upper: np.ndarray
    horizon: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    def __post_init__(self):
        self.lower=np.asarray(self.lower,dtype=float).reshape(-1); self.upper=np.asarray(self.upper,dtype=float).reshape(-1)
        if self.lower.shape!=self.upper.shape or np.any(self.lower>self.upper): raise ValueError("invalid reachable bounds")
        if not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all(): raise ValueError("bounds must be finite")
    @property
    def dimension(self): return self.lower.size
    @property
    def center(self): return (self.lower+self.upper)/2
    @property
    def radius(self): return (self.upper-self.lower)/2
    @property
    def volume_proxy(self): return float(np.prod(self.upper-self.lower))
    def contains(self, point: Sequence[float]):
        p=np.asarray(point,dtype=float).reshape(-1)
        if p.shape!=self.lower.shape: raise ValueError("dimension mismatch")
        return bool(np.all(p>=self.lower)&np.all(p<=self.upper))
    def distance(self, point: Sequence[float]):
        p=np.asarray(point,dtype=float).reshape(-1)
        if p.shape!=self.lower.shape: raise ValueError("dimension mismatch")
        d=np.maximum(self.lower-p,0)+np.minimum(self.upper-p,0)
        return float(np.linalg.norm(d))
    def intersects(self, other):
        if self.dimension!=other.dimension: raise ValueError("dimension mismatch")
        return bool(np.all(self.lower<=other.upper)&np.all(other.lower<=self.upper))
    def intersection(self, other):
        if not self.intersects(other): return None
        return ReachableSet(np.maximum(self.lower,other.lower),np.minimum(self.upper,other.upper),self.horizon)
    @classmethod
    def from_center_radius(cls, center, radius, *, horizon=None):
        c=np.asarray(center,dtype=float).reshape(-1); r=np.asarray(radius,dtype=float).reshape(-1)
        if c.shape!=r.shape or np.any(r<0): raise ValueError("invalid center/radius")
        return cls(c-r,c+r,horizon)

def build_reachable_set(center, uncertainty, *, horizon=None, scale=1.0):
    if scale<=0: raise ValueError("scale must be positive")
    return ReachableSet.from_center_radius(center,np.abs(np.asarray(uncertainty))*scale,horizon=horizon)

__all__=["ReachableSet","build_reachable_set"]
