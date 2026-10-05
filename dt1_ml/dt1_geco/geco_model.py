from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

@dataclass(frozen=True)
class GECOParameters:
    """Parameters for the five-state GECO research model."""
    egp_over_p1: float
    p1: float
    kc: float
    p3: float
    p2: float
    ki: float
    ku: float
    ka: float
    tau_i: float
    basal_glucose: float = 120.0
    basal_insulin: float = 0.0
    substep_minutes: float = 1.0
    substeps_per_observation: int = 5
    glucose_min: float = 20.0
    glucose_max: float = 600.0
    x_min_offset: float = 0.5

    @property
    def egp(self) -> float:
        return self.egp_over_p1 * self.p1

    @classmethod
    def from_egp(cls, *, egp: float, p1: float, **kwargs) -> "GECOParameters":
        if p1 <= 0:
            raise ValueError("p1 must be positive")
        return cls(egp_over_p1=egp / p1, p1=p1, **kwargs)

@dataclass
class GECOState:
    glucose: float
    insulin_action: float
    insulin_state: float
    gut_carbohydrate: float
    interstitial_glucose: float

    def as_array(self) -> Tuple[float, float, float, float, float]:
        return (self.glucose, self.insulin_action, self.insulin_state,
                self.gut_carbohydrate, self.interstitial_glucose)
    def copy(self):
        return replace(self)

@dataclass(frozen=True)
class GECOOutput:
    state: GECOState
    glucose: float
    interstitial_glucose: float
    insulin_action: float
    insulin_state: float
    gut_carbohydrate: float
    insulin_input: float
    carbohydrate_input: float
    elapsed_minutes: float

class GECOModel:
    """Five-state GECO model used as the physiological component of DT1.

    Dynamics:
      G_dot  = egp - p1*G - X*max(G,30) + kc*ka*Q
      X_dot  = -p2*X + p3*(I-Ib)
      I_dot  = -ki*I + ku*u
      Q_dot  = -ka*Q + c
      Gi_dot = (G-Gi)/tau_i

    The implementation follows the research paper's five one-minute Euler
    substeps per observation interval and its numerical state clamps.
    This is a research/simulation implementation, not a clinically
    validated physiological model.
    """
    STATE_NAMES = ("glucose", "insulin_action", "insulin_state",
                   "gut_carbohydrate", "interstitial_glucose")

    def __init__(self, params: GECOParameters, initial_state: Optional[GECOState] = None):
        self.params = params
        self._validate_parameters()
        self.state = initial_state.copy() if initial_state else self.initial_state()
        self.state = self._clamp_state(self.state)
        self.elapsed_minutes = 0.0

    def _validate_parameters(self):
        p = self.params
        for name in ("p1", "p2", "p3", "ki", "ku", "ka", "tau_i", "substep_minutes"):
            if getattr(p, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if p.substeps_per_observation < 1:
            raise ValueError("substeps_per_observation must be >= 1")
        if p.glucose_min >= p.glucose_max:
            raise ValueError("glucose_min must be smaller than glucose_max")

    def initial_state(self, glucose=None, interstitial_glucose=None,
                      insulin_state=None, insulin_action=0.0,
                      gut_carbohydrate=0.0) -> GECOState:
        g = self.params.basal_glucose if glucose is None else float(glucose)
        gi = g if interstitial_glucose is None else float(interstitial_glucose)
        i = self.params.basal_insulin if insulin_state is None else float(insulin_state)
        return self._clamp_state(GECOState(g, float(insulin_action), i,
                                           float(gut_carbohydrate), gi))

    def reset(self, **kwargs) -> GECOState:
        self.state = self.initial_state(**kwargs)
        self.elapsed_minutes = 0.0
        return self.state.copy()

    def _derivatives(self, state: GECOState, insulin_u: float, carbohydrate_c: float) -> GECOState:
        p = self.params
        g, x, i, q, gi = state.as_array()
        return GECOState(
            p.egp - p.p1*g - x*max(g, 30.0) + p.kc*p.ka*q,
            -p.p2*x + p.p3*(i-p.basal_insulin),
            -p.ki*i + p.ku*max(float(insulin_u), 0.0),
            -p.ka*q + max(float(carbohydrate_c), 0.0),
            (g-gi)/p.tau_i,
        )

    def _clamp_state(self, s: GECOState) -> GECOState:
        p = self.params
        return GECOState(
            min(max(s.glucose, p.glucose_min), p.glucose_max),
            max(s.insulin_action, -p.p1*p.x_min_offset),
            max(s.insulin_state, 0.0),
            max(s.gut_carbohydrate, 0.0),
            min(max(s.interstitial_glucose, p.glucose_min), p.glucose_max),
        )

    def _euler_step(self, dt, insulin_u, carbohydrate_c):
        d = self._derivatives(self.state, insulin_u, carbohydrate_c)
        s = self.state
        self.state = self._clamp_state(GECOState(
            s.glucose + dt*d.glucose,
            s.insulin_action + dt*d.insulin_action,
            s.insulin_state + dt*d.insulin_state,
            s.gut_carbohydrate + dt*d.gut_carbohydrate,
            s.interstitial_glucose + dt*d.interstitial_glucose,
        ))

    def step(self, insulin_u=0.0, carbohydrate_c=0.0, observation_minutes=None) -> GECOOutput:
        p = self.params
        total = (p.substep_minutes*p.substeps_per_observation
                 if observation_minutes is None else float(observation_minutes))
        if total <= 0:
            raise ValueError("observation interval must be > 0")
        remaining = total
        while remaining > 1e-12:
            dt = min(p.substep_minutes, remaining)
            self._euler_step(dt, insulin_u, carbohydrate_c)
            self.elapsed_minutes += dt
            remaining -= dt
        s = self.state.copy()
        return GECOOutput(s, s.glucose, s.interstitial_glucose, s.insulin_action,
                          s.insulin_state, s.gut_carbohydrate, float(insulin_u),
                          float(carbohydrate_c), self.elapsed_minutes)

    def predict(self, insulin_inputs: Sequence[float], carbohydrate_inputs=None,
                observation_minutes=None) -> List[GECOOutput]:
        u = list(insulin_inputs)
        c = [0.0]*len(u) if carbohydrate_inputs is None else list(carbohydrate_inputs)
        if len(u) != len(c):
            raise ValueError("insulin_inputs and carbohydrate_inputs must have the same length")
        return [self.step(ui, ci, observation_minutes) for ui, ci in zip(u, c)]

    def predict_glucose(self, insulin_inputs, carbohydrate_inputs=None, observation_minutes=None):
        return [o.glucose for o in self.predict(insulin_inputs, carbohydrate_inputs, observation_minutes)]

    def get_state(self) -> GECOState:
        return self.state.copy()

    def set_state(self, state: GECOState):
        self.state = self._clamp_state(state.copy())

__all__ = ["GECOParameters", "GECOState", "GECOOutput", "GECOModel"]
