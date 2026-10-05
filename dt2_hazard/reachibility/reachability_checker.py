"""Safety and consistency checks over DT2 reachable trajectories."""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Optional,Sequence,Dict,List
import numpy as np
from .reachable_set import ReachableSet
from .reachable_trajectory import ReachableTrajectory
from .unsafe_region import UnsafeRegion

@dataclass(frozen=True)
class ReachabilityCheckConfig:
    maximum_radius: Optional[float]=None
    deadline_horizon: Optional[int]=None

@dataclass
class ReachabilityCheckResult:
    reachable:bool
    unsafe:bool
    deadline_violated:bool
    uncertainty_excessive:bool
    first_unsafe_horizon:Optional[int]
    messages:List[str]=field(default_factory=list)
    region_hits:Dict[int,List[str]]=field(default_factory=dict)

class ReachabilityChecker:
    def __init__(self,config=None): self.config=config or ReachabilityCheckConfig()
    def check_set(self,reachable_set,predicted_state): return reachable_set.contains(predicted_state)
    def check_trajectory(self,trajectory:ReachableTrajectory,unsafe_regions:Sequence[UnsafeRegion]):
        first=trajectory.first_unsafe_horizon; unsafe=trajectory.enters_unsafe_region; hits={}
        for p in trajectory.points:
            names=[r.name for r in unsafe_regions if r.name in p.unsafe_regions]
            if names: hits[p.horizon]=names
        deadline=bool(unsafe and self.config.deadline_horizon is not None and first is not None and first<=self.config.deadline_horizon)
        excessive=False
        if self.config.maximum_radius is not None:
            excessive=any(np.any((p.upper-p.lower)/2>self.config.maximum_radius) for p in trajectory.points)
        msgs=[]
        if unsafe: msgs.append("Reachable trajectory intersects an unsafe region.")
        else: msgs.append("No reachable set intersects the supplied unsafe regions.")
        if deadline: msgs.append("Unsafe reachable state occurs within the configured deadline.")
        if excessive: msgs.append("Reachable-set uncertainty exceeds the configured maximum radius.")
        return ReachabilityCheckResult(bool(trajectory.points),unsafe,deadline,excessive,first,msgs,hits)

def check_reachability(trajectory,unsafe_regions,*,config=None): return ReachabilityChecker(config).check_trajectory(trajectory,unsafe_regions)

__all__=["ReachabilityCheckConfig","ReachabilityCheckResult","ReachabilityChecker","check_reachability"]
