"""Unsafe-region definitions used by DT2 reachability analysis."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence
import numpy as np

class RegionType(str,Enum):
    BOX="box"; PREDICATE="predicate"; GLUCOSE_LOW="glucose_low"; GLUCOSE_HIGH="glucose_high"

@dataclass(frozen=True)
class UnsafeRegion:
    name: str
    region_type: RegionType
    lower: Optional[np.ndarray]=None
    upper: Optional[np.ndarray]=None
    predicate: Optional[Callable[[np.ndarray],bool]]=None
    description: str=""
    def __post_init__(self):
        if self.region_type==RegionType.BOX:
            if self.lower is None or self.upper is None: raise ValueError("box requires bounds")
            lo=np.asarray(self.lower,dtype=float).reshape(-1); hi=np.asarray(self.upper,dtype=float).reshape(-1)
            if lo.shape!=hi.shape or np.any(lo>hi): raise ValueError("invalid box")
            object.__setattr__(self,"lower",lo); object.__setattr__(self,"upper",hi)
        if self.region_type==RegionType.PREDICATE and self.predicate is None: raise ValueError("predicate required")
    def contains(self,state: Sequence[float]):
        x=np.asarray(state,dtype=float).reshape(-1)
        if self.region_type==RegionType.BOX: return bool(np.all(x>=self.lower)&np.all(x<=self.upper))
        if self.region_type==RegionType.GLUCOSE_LOW: return bool(x.size and x[0]<self.lower[0])
        if self.region_type==RegionType.GLUCOSE_HIGH: return bool(x.size and x[0]>self.upper[0])
        return bool(self.predicate(x))

def glucose_low_region(threshold=70.0,*,name="low_glucose"):
    return UnsafeRegion(name,RegionType.GLUCOSE_LOW,lower=np.array([threshold],float),description=f"glucose below {threshold}")
def glucose_high_region(threshold=180.0,*,name="high_glucose"):
    return UnsafeRegion(name,RegionType.GLUCOSE_HIGH,upper=np.array([threshold],float),description=f"glucose above {threshold}")
def box_region(lower,upper,*,name="unsafe_box",description=""):
    return UnsafeRegion(name,RegionType.BOX,np.asarray(lower,float),np.asarray(upper,float),description=description)
def state_is_unsafe(state,regions): return any(r.contains(state) for r in regions)

__all__=["RegionType","UnsafeRegion","glucose_low_region","glucose_high_region","box_region","state_is_unsafe"]
