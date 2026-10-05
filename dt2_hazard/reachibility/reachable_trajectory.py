"""Time-ordered reachable trajectory representation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List,Optional,Sequence
import numpy as np
from .reachable_set import ReachableSet
from .unsafe_region import UnsafeRegion

@dataclass
class ReachableTrajectoryPoint:
    horizon:int
    center:np.ndarray
    lower:np.ndarray
    upper:np.ndarray
    unsafe_regions:List[str]
    @property
    def is_unsafe(self): return bool(self.unsafe_regions)

@dataclass
class ReachableTrajectory:
    points:List[ReachableTrajectoryPoint]
    @property
    def horizons(self): return np.asarray([p.horizon for p in self.points],int)
    @property
    def first_unsafe_horizon(self)->Optional[int]:
        h=[p.horizon for p in self.points if p.is_unsafe]; return min(h) if h else None
    @property
    def enters_unsafe_region(self): return self.first_unsafe_horizon is not None
    def glucose_bounds(self):
        if not self.points: return tuple(np.empty(0) for _ in range(3))
        return (np.asarray([p.lower[0] for p in self.points]),np.asarray([p.center[0] for p in self.points]),np.asarray([p.upper[0] for p in self.points]))

def _intersects(rs,region):
    t=region.region_type.value
    if t=="box": return bool(np.all(rs.lower<=region.upper)&np.all(region.lower<=rs.upper))
    if t=="glucose_low": return bool(rs.lower[0]<region.lower[0])
    if t=="glucose_high": return bool(rs.upper[0]>region.upper[0])
    return any(region.contains(x) for x in [rs.center,rs.lower,rs.upper])

def build_reachable_trajectory(horizons:Sequence[int],reachable_sets:Sequence[ReachableSet],unsafe_regions:Sequence[UnsafeRegion]):
    if len(horizons)!=len(reachable_sets): raise ValueError("length mismatch")
    points=[]
    for h,rs in zip(horizons,reachable_sets):
        hits=[r.name for r in unsafe_regions if _intersects(rs,r)]
        points.append(ReachableTrajectoryPoint(int(h),rs.center.copy(),rs.lower.copy(),rs.upper.copy(),hits))
    return ReachableTrajectory(points)

__all__=["ReachableTrajectoryPoint","ReachableTrajectory","build_reachable_trajectory"]
