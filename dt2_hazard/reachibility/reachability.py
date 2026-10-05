"""Lightweight reachable-set analyzer for DT2."""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Optional,Sequence,Dict,List
import numpy as np
from .reachable_set import ReachableSet
from .unsafe_region import UnsafeRegion

@dataclass(frozen=True)
class ReachabilityConfig:
    uncertainty_scale: float=1.0
    minimum_radius: float=1.0
    growth_per_horizon: float=0.10
    clip_min: Optional[float]=20.0
    clip_max: Optional[float]=600.0

@dataclass
class ReachabilityResult:
    horizons: np.ndarray
    reachable_sets: List[ReachableSet]
    unsafe_intersections: Dict[int,List[str]]
    unsafe_reachable: bool
    first_unsafe_horizon: Optional[int]
    metadata: Dict[str,object]=field(default_factory=dict)
    @property
    def unsafe_horizons(self): return [h for h,v in self.unsafe_intersections.items() if v]

class ReachabilityAnalyzer:
    def __init__(self,config=None): self.config=config or ReachabilityConfig()
    def build_sets(self,predictions,uncertainty,horizons):
        p=np.asarray(predictions,float); u=np.asarray(uncertainty,float); h=np.asarray(horizons,int)
        if p.ndim==1:p=p[:,None]
        if u.ndim==1:u=u[:,None]
        if p.shape!=u.shape or len(h)!=len(p): raise ValueError("shape mismatch")
        out=[]
        for row,sigma,hh in zip(p,u,h):
            growth=1+self.config.growth_per_horizon*max(int(hh)-1,0)
            r=np.maximum(np.abs(sigma)*self.config.uncertainty_scale*growth,self.config.minimum_radius)
            lo=row-r; hi=row+r
            if self.config.clip_min is not None: lo[0]=max(lo[0],self.config.clip_min)
            if self.config.clip_max is not None: hi[0]=min(hi[0],self.config.clip_max)
            out.append(ReachableSet(lo,hi,int(hh)))
        return out
    def analyze(self,predictions,uncertainty,horizons,unsafe_regions):
        h=np.asarray(horizons,int); sets=self.build_sets(predictions,uncertainty,h); hits={}
        for hh,rs in zip(h,sets): hits[int(hh)]=[r.name for r in unsafe_regions if _intersects(rs,r)]
        unsafe=[int(x) for x in h if hits[int(x)]]
        return ReachabilityResult(h,sets,hits,bool(unsafe),min(unsafe) if unsafe else None)

def _intersects(rs,region):
    t=region.region_type.value
    if t=="box": return rs.intersects(ReachableSet(region.lower,region.upper))
    if t=="glucose_low": return bool(rs.lower[0]<region.lower[0])
    if t=="glucose_high": return bool(rs.upper[0]>region.upper[0])
    candidates=[rs.center]
    if rs.dimension<=10:
        for mask in range(1<<rs.dimension):
            x=rs.lower.copy()
            for j in range(rs.dimension):
                if mask&(1<<j): x[j]=rs.upper[j]
            candidates.append(x)
    return any(region.contains(x) for x in candidates)

def analyze_reachability(predictions,uncertainty,horizons,unsafe_regions,*,config=None):
    return ReachabilityAnalyzer(config).analyze(predictions,uncertainty,horizons,unsafe_regions)

__all__=["ReachabilityConfig","ReachabilityResult","ReachabilityAnalyzer","analyze_reachability"]
