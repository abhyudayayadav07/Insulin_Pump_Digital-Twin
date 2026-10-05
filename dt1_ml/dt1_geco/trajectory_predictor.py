"""
GECO-based future glucose trajectory prediction for DT1.

This module takes an estimated GECO physiological state and simulates
future glucose under supplied insulin/carbohydrate scenarios.

Pipeline:

    GECO parameters
          +
    current estimated state
          +
    future insulin / meal inputs
          |
          v
    GECO trajectory predictor
          |
          v
    future glucose trajectory

This is a research/simulation component and is not clinically validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .geco_model import GECOModel, GECOParameters, GECOState
except ImportError:
    from geco_model import GECOModel, GECOParameters, GECOState


@dataclass(frozen=True)
class TrajectoryPoint:
    """One point in a predicted physiological trajectory."""

    step: int
    time_minutes: float
    glucose: float
    interstitial_glucose: float
    insulin_action: float
    insulin_state: float
    gut_carbohydrate: float
    insulin_input: float
    carbohydrate_input: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "time_minutes": self.time_minutes,
            "glucose": self.glucose,
            "interstitial_glucose": self.interstitial_glucose,
            "insulin_action": self.insulin_action,
            "insulin_state": self.insulin_state,
            "gut_carbohydrate": self.gut_carbohydrate,
            "insulin_input": self.insulin_input,
            "carbohydrate_input": self.carbohydrate_input,
        }


@dataclass
class GlucoseTrajectory:
    """Container for a complete future GECO trajectory."""

    points: List[TrajectoryPoint]
    initial_state: GECOState
    horizon_minutes: float
    step_minutes: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def glucose(self) -> np.ndarray:
        return np.asarray([p.glucose for p in self.points], dtype=float)

    @property
    def interstitial_glucose(self) -> np.ndarray:
        return np.asarray(
            [p.interstitial_glucose for p in self.points],
            dtype=float,
        )

    @property
    def time_minutes(self) -> np.ndarray:
        return np.asarray(
            [p.time_minutes for p in self.points],
            dtype=float,
        )

    @property
    def insulin(self) -> np.ndarray:
        return np.asarray(
            [p.insulin_input for p in self.points],
            dtype=float,
        )

    @property
    def carbohydrate(self) -> np.ndarray:
        return np.asarray(
            [p.carbohydrate_input for p in self.points],
            dtype=float,
        )

    def final_glucose(self) -> float:
        if not self.points:
            return float("nan")
        return float(self.points[-1].glucose)

    def min_glucose(self) -> float:
        if not self.points:
            return float("nan")
        return float(np.min(self.glucose))

    def max_glucose(self) -> float:
        if not self.points:
            return float("nan")
        return float(np.max(self.glucose))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "initial_state": {
                "glucose": self.initial_state.glucose,
                "insulin_action": self.initial_state.insulin_action,
                "insulin_state": self.initial_state.insulin_state,
                "gut_carbohydrate": self.initial_state.gut_carbohydrate,
                "interstitial_glucose": self.initial_state.interstitial_glucose,
            },
            "horizon_minutes": self.horizon_minutes,
            "step_minutes": self.step_minutes,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TrajectoryPredictorConfig:
    """Configuration for GECO trajectory prediction."""

    default_step_minutes: float = 5.0
    default_horizon_minutes: float = 30.0

    glucose_min: float = 20.0
    glucose_max: float = 600.0

    include_initial_point: bool = True

    def validate(self) -> None:
        if self.default_step_minutes <= 0:
            raise ValueError("default_step_minutes must be > 0.")

        if self.default_horizon_minutes <= 0:
            raise ValueError("default_horizon_minutes must be > 0.")

        if self.glucose_min >= self.glucose_max:
            raise ValueError("glucose_min must be smaller than glucose_max.")


class GECOTrajectoryPredictor:
    """
    Predict future glucose trajectories using the GECO model.

    The predictor never modifies the supplied initial state. It creates
    a separate GECOModel instance for each prediction.

    Future inputs can be supplied as:
        1. constant scalar values,
        2. sequences aligned to prediction steps,
        3. a callable(step, time_minutes) -> value.

    Examples
    --------
    Constant insulin:
        predictor.predict(
            state=state,
            horizon_minutes=30,
            insulin_input=0.02,
        )

    Step-wise future insulin:
        predictor.predict(
            state=state,
            horizon_minutes=30,
            step_minutes=5,
            insulin_input=[0.01, 0.01, 0.02, 0.02, 0.02, 0.01],
        )
    """

    def __init__(
        self,
        parameters: Optional[GECOParameters] = None,
        config: Optional[TrajectoryPredictorConfig] = None,
    ):
        self.parameters = parameters or GECOParameters()
        self.config = config or TrajectoryPredictorConfig()
        self.config.validate()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_horizon(horizon_minutes: float, step_minutes: float) -> int:
        if horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be > 0.")

        if step_minutes <= 0:
            raise ValueError("step_minutes must be > 0.")

        steps = int(round(horizon_minutes / step_minutes))

        if steps <= 0:
            raise ValueError("Prediction horizon contains no steps.")

        # Avoid silently changing a requested horizon by a large amount.
        actual_horizon = steps * step_minutes
        tolerance = max(1e-9, 0.5 * step_minutes)

        if abs(actual_horizon - horizon_minutes) > tolerance:
            raise ValueError(
                "horizon_minutes must be an integer multiple of "
                "step_minutes."
            )

        return steps

    @staticmethod
    def _input_value(
        source: Any,
        step_index: int,
        time_minutes: float,
        total_steps: int,
        name: str,
    ) -> float:
        if callable(source):
            value = source(step_index, time_minutes)
        elif np.isscalar(source):
            value = source
        else:
            values = list(source)

            if len(values) < total_steps:
                raise ValueError(
                    f"{name} sequence must contain at least "
                    f"{total_steps} values."
                )

            value = values[step_index]

        value = float(value)

        if not np.isfinite(value):
            raise ValueError(f"{name} contains a non-finite value.")

        return value

    @staticmethod
    def _state_copy(state: GECOState) -> GECOState:
        return GECOState(
            glucose=float(state.glucose),
            insulin_action=float(state.insulin_action),
            insulin_state=float(state.insulin_state),
            gut_carbohydrate=float(state.gut_carbohydrate),
            interstitial_glucose=float(state.interstitial_glucose),
        )

    def _clamp_state(self, state: GECOState) -> GECOState:
        p = self.parameters
        x_min = -p.p1 * p.x_min_offset

        return GECOState(
            glucose=float(
                np.clip(
                    state.glucose,
                    max(self.config.glucose_min, p.glucose_min),
                    min(self.config.glucose_max, p.glucose_max),
                )
            ),
            insulin_action=max(float(state.insulin_action), x_min),
            insulin_state=max(float(state.insulin_state), 0.0),
            gut_carbohydrate=max(float(state.gut_carbohydrate), 0.0),
            interstitial_glucose=float(
                np.clip(
                    state.interstitial_glucose,
                    max(self.config.glucose_min, p.glucose_min),
                    min(self.config.glucose_max, p.glucose_max),
                )
            ),
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        state: GECOState,
        horizon_minutes: Optional[float] = None,
        step_minutes: Optional[float] = None,
        insulin_input: Any = 0.0,
        carbohydrate_input: Any = 0.0,
        include_initial_point: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GlucoseTrajectory:
        """
        Predict a future GECO trajectory.

        Parameters
        ----------
        state:
            Current estimated GECO state.

        horizon_minutes:
            Future prediction horizon.

        step_minutes:
            Time between returned trajectory points.

        insulin_input:
            Scalar, sequence, or callable future insulin input.

        carbohydrate_input:
            Scalar, sequence, or callable future carbohydrate input.

        include_initial_point:
            If True, trajectory starts at t=0 with the supplied state.
        """
        horizon = (
            self.config.default_horizon_minutes
            if horizon_minutes is None
            else float(horizon_minutes)
        )

        step = (
            self.config.default_step_minutes
            if step_minutes is None
            else float(step_minutes)
        )

        total_steps = self._validate_horizon(horizon, step)

        include_initial = (
            self.config.include_initial_point
            if include_initial_point is None
            else bool(include_initial_point)
        )

        current_state = self._clamp_state(self._state_copy(state))
        initial_state = self._state_copy(current_state)

        # GECOModel.step internally uses one-minute Euler substeps. For
        # a requested prediction interval, the model receives the complete
        # interval and performs the appropriate number of substeps.
        model = GECOModel(
            params=self.parameters,
            initial_state=current_state,
        )

        points: List[TrajectoryPoint] = []

        if include_initial:
            points.append(
                TrajectoryPoint(
                    step=0,
                    time_minutes=0.0,
                    glucose=float(current_state.glucose),
                    interstitial_glucose=float(
                        current_state.interstitial_glucose
                    ),
                    insulin_action=float(current_state.insulin_action),
                    insulin_state=float(current_state.insulin_state),
                    gut_carbohydrate=float(current_state.gut_carbohydrate),
                    insulin_input=0.0,
                    carbohydrate_input=0.0,
                )
            )

        for i in range(total_steps):
            t = (i + 1) * step

            insulin = self._input_value(
                insulin_input,
                i,
                t,
                total_steps,
                "insulin_input",
            )

            carbohydrate = self._input_value(
                carbohydrate_input,
                i,
                t,
                total_steps,
                "carbohydrate_input",
            )

            output = model.step(
                insulin_u=insulin,
                carbohydrate_c=carbohydrate,
                observation_minutes=step,
            )

            current_state = self._clamp_state(output.state)

            # Keep the prediction model synchronized with any explicit
            # numerical clamping applied by the predictor.
            model.set_state(current_state)

            points.append(
                TrajectoryPoint(
                    step=i + 1,
                    time_minutes=float(t),
                    glucose=float(current_state.glucose),
                    interstitial_glucose=float(
                        current_state.interstitial_glucose
                    ),
                    insulin_action=float(current_state.insulin_action),
                    insulin_state=float(current_state.insulin_state),
                    gut_carbohydrate=float(current_state.gut_carbohydrate),
                    insulin_input=insulin,
                    carbohydrate_input=carbohydrate,
                )
            )

        result_metadata = {
            "model": "GECO",
            "prediction_type": "mechanistic_trajectory",
            "horizon_minutes": horizon,
            "step_minutes": step,
            "include_initial_point": include_initial,
        }

        if metadata:
            result_metadata.update(metadata)

        return GlucoseTrajectory(
            points=points,
            initial_state=initial_state,
            horizon_minutes=horizon,
            step_minutes=step,
            metadata=result_metadata,
        )

    # ------------------------------------------------------------------
    # Convenience prediction modes
    # ------------------------------------------------------------------

    def predict_glucose(
        self,
        state: GECOState,
        horizon_minutes: Optional[float] = None,
        step_minutes: Optional[float] = None,
        insulin_input: Any = 0.0,
        carbohydrate_input: Any = 0.0,
    ) -> np.ndarray:
        """Return only the predicted plasma glucose values."""
        trajectory = self.predict(
            state=state,
            horizon_minutes=horizon_minutes,
            step_minutes=step_minutes,
            insulin_input=insulin_input,
            carbohydrate_input=carbohydrate_input,
        )
        return trajectory.glucose.copy()

    def predict_interstitial_glucose(
        self,
        state: GECOState,
        horizon_minutes: Optional[float] = None,
        step_minutes: Optional[float] = None,
        insulin_input: Any = 0.0,
        carbohydrate_input: Any = 0.0,
    ) -> np.ndarray:
        """Return only the predicted interstitial glucose values."""
        trajectory = self.predict(
            state=state,
            horizon_minutes=horizon_minutes,
            step_minutes=step_minutes,
            insulin_input=insulin_input,
            carbohydrate_input=carbohydrate_input,
        )
        return trajectory.interstitial_glucose.copy()

    def predict_constant_inputs(
        self,
        state: GECOState,
        insulin_u: float = 0.0,
        carbohydrate_c: float = 0.0,
        horizon_minutes: Optional[float] = None,
        step_minutes: Optional[float] = None,
    ) -> GlucoseTrajectory:
        """Convenience wrapper for constant future inputs."""
        return self.predict(
            state=state,
            horizon_minutes=horizon_minutes,
            step_minutes=step_minutes,
            insulin_input=float(insulin_u),
            carbohydrate_input=float(carbohydrate_c),
        )

    # ------------------------------------------------------------------
    # Scenario comparison
    # ------------------------------------------------------------------

    def compare_scenarios(
        self,
        state: GECOState,
        scenarios: Dict[str, Dict[str, Any]],
        horizon_minutes: Optional[float] = None,
        step_minutes: Optional[float] = None,
    ) -> Dict[str, GlucoseTrajectory]:
        """
        Predict multiple future input scenarios from the same state.

        Each scenario dictionary can contain:
            insulin_input
            carbohydrate_input
            metadata
        """
        results: Dict[str, GlucoseTrajectory] = {}

        for name, scenario in scenarios.items():
            if not isinstance(scenario, dict):
                raise TypeError(
                    f"Scenario '{name}' must be a dictionary."
                )

            results[name] = self.predict(
                state=state,
                horizon_minutes=horizon_minutes,
                step_minutes=step_minutes,
                insulin_input=scenario.get("insulin_input", 0.0),
                carbohydrate_input=scenario.get(
                    "carbohydrate_input",
                    0.0,
                ),
                metadata={
                    "scenario": name,
                    **scenario.get("metadata", {}),
                },
            )

        return results

    # ------------------------------------------------------------------
    # DataFrame-friendly output
    # ------------------------------------------------------------------

    @staticmethod
    def to_records(
        trajectory: GlucoseTrajectory,
    ) -> List[Dict[str, Any]]:
        """Convert a trajectory into a list of dictionaries."""
        return [point.to_dict() for point in trajectory.points]

    @staticmethod
    def to_numpy(
        trajectory: GlucoseTrajectory,
    ) -> Dict[str, np.ndarray]:
        """Convert a trajectory to NumPy arrays."""
        return {
            "time_minutes": trajectory.time_minutes,
            "glucose": trajectory.glucose,
            "interstitial_glucose": trajectory.interstitial_glucose,
            "insulin": trajectory.insulin,
            "carbohydrate": trajectory.carbohydrate,
        }


def create_trajectory_predictor(
    parameters: Optional[GECOParameters] = None,
    config: Optional[TrajectoryPredictorConfig] = None,
) -> GECOTrajectoryPredictor:
    """Convenience factory."""
    return GECOTrajectoryPredictor(
        parameters=parameters,
        config=config,
    )


def predict_geco_trajectory(
    state: GECOState,
    parameters: Optional[GECOParameters] = None,
    horizon_minutes: float = 30.0,
    step_minutes: float = 5.0,
    insulin_input: Any = 0.0,
    carbohydrate_input: Any = 0.0,
) -> GlucoseTrajectory:
    """Functional convenience wrapper for one GECO trajectory."""
    predictor = GECOTrajectoryPredictor(parameters=parameters)

    return predictor.predict(
        state=state,
        horizon_minutes=horizon_minutes,
        step_minutes=step_minutes,
        insulin_input=insulin_input,
        carbohydrate_input=carbohydrate_input,
    )


__all__ = [
    "TrajectoryPoint",
    "GlucoseTrajectory",
    "TrajectoryPredictorConfig",
    "GECOTrajectoryPredictor",
    "create_trajectory_predictor",
    "predict_geco_trajectory",
]
