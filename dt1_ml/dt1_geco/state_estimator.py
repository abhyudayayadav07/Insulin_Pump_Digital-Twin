"""
State estimation for the GECO physiological component of DT1.

This module provides a lightweight Extended Kalman Filter (EKF) for the
five-state GECO model:

    x = [G, X, I, Q, Gi]

where:
    G  = plasma glucose
    X  = remote insulin action
    I  = insulin state
    Q  = gut carbohydrate state
    Gi = interstitial glucose

The observed CGM measurement is modeled as:

    y = Gi + measurement_noise

The estimator is intentionally separated from ``geco_model.py``:
    - geco_model.py       -> deterministic physiological dynamics
    - state_estimator.py  -> state estimation from noisy observations
    - parameter_estimation.py -> parameter fitting

This is a research/simulation implementation. It is not a clinically
validated state estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    from .geco_model import GECOModel, GECOParameters, GECOState
except ImportError:
    from geco_model import GECOModel, GECOParameters, GECOState


@dataclass(frozen=True)
class GECOFilterConfig:
    """Configuration for the GECO extended Kalman filter."""

    # CGM measurement noise variance, (mg/dL)^2.
    measurement_variance: float = 81.0

    # Process noise variances for:
    # [G, X, I, Q, Gi]
    process_variance: Tuple[float, float, float, float, float] = (
        1.0,
        1e-4,
        1e-3,
        1e-2,
        1.0,
    )

    # Initial covariance diagonal.
    initial_covariance: Tuple[float, float, float, float, float] = (
        25.0,
        1.0,
        1.0,
        10.0,
        25.0,
    )

    # Numerical safeguards.
    covariance_floor: float = 1e-10
    glucose_min: float = 20.0
    glucose_max: float = 600.0

    # Innovation validation / reset behavior.
    max_condition_number: float = 1e12
    reset_on_numerical_failure: bool = True

    def validate(self) -> None:
        if self.measurement_variance <= 0:
            raise ValueError("measurement_variance must be > 0.")

        if len(self.process_variance) != 5:
            raise ValueError("process_variance must contain five values.")

        if len(self.initial_covariance) != 5:
            raise ValueError("initial_covariance must contain five values.")

        if any(v < 0 for v in self.process_variance):
            raise ValueError("process_variance cannot contain negative values.")

        if any(v < 0 for v in self.initial_covariance):
            raise ValueError("initial_covariance cannot contain negative values.")

        if self.covariance_floor <= 0:
            raise ValueError("covariance_floor must be > 0.")

        if self.glucose_min >= self.glucose_max:
            raise ValueError("glucose_min must be smaller than glucose_max.")


@dataclass
class GECOStateEstimate:
    """One EKF estimate and associated measurement information."""

    state: GECOState
    covariance: np.ndarray

    measurement: Optional[float]
    predicted_measurement: Optional[float]

    innovation: Optional[float]
    innovation_variance: Optional[float]
    normalized_innovation: Optional[float]

    kalman_gain: Optional[np.ndarray]

    initialized: bool
    updated: bool
    numerical_reset: bool = False

    timestamp: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.state.as_array(), dtype=float)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": {
                "glucose": self.state.glucose,
                "insulin_action": self.state.insulin_action,
                "insulin_state": self.state.insulin_state,
                "gut_carbohydrate": self.state.gut_carbohydrate,
                "interstitial_glucose": self.state.interstitial_glucose,
            },
            "covariance": self.covariance.tolist(),
            "measurement": self.measurement,
            "predicted_measurement": self.predicted_measurement,
            "innovation": self.innovation,
            "innovation_variance": self.innovation_variance,
            "normalized_innovation": self.normalized_innovation,
            "kalman_gain": (
                None
                if self.kalman_gain is None
                else self.kalman_gain.tolist()
            ),
            "initialized": self.initialized,
            "updated": self.updated,
            "numerical_reset": self.numerical_reset,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


class GECOStateEstimator:
    """
    Extended Kalman Filter for the five-state GECO model.

    The filter uses:
        prediction -> CGM measurement update -> numerical validation

    The measurement model is:

        y = Gi

    because CGM is treated as an observation of interstitial glucose.

    The implementation uses finite-difference Jacobians rather than
    symbolic derivatives, which keeps the estimator synchronized with
    the actual GECOModel implementation.
    """

    STATE_DIM = 5

    def __init__(
        self,
        model: GECOModel,
        config: Optional[GECOFilterConfig] = None,
        initial_state: Optional[GECOState] = None,
    ):
        self.model = model
        self.config = config or GECOFilterConfig()
        self.config.validate()

        self.state = (
            initial_state.copy()
            if initial_state is not None
            else model.get_state()
        )

        self.covariance = np.diag(
            np.asarray(self.config.initial_covariance, dtype=float)
        )

        self.initialized = True
        self.last_estimate: Optional[GECOStateEstimate] = None
        self.update_count = 0
        self.reset_count = 0

        self._ensure_valid_covariance()

    # ------------------------------------------------------------------
    # State/vector conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _state_to_vector(state: GECOState) -> np.ndarray:
        return np.asarray(state.as_array(), dtype=float)

    @staticmethod
    def _vector_to_state(values: Sequence[float]) -> GECOState:
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) != 5:
            raise ValueError("GECO state vector must contain five values.")

        return GECOState(
            glucose=float(values[0]),
            insulin_action=float(values[1]),
            insulin_state=float(values[2]),
            gut_carbohydrate=float(values[3]),
            interstitial_glucose=float(values[4]),
        )

    def _clamp_state(self, state: GECOState) -> GECOState:
        p = self.model.params
        x_min = -p.p1 * p.x_min_offset

        return GECOState(
            glucose=float(np.clip(
                state.glucose,
                max(self.config.glucose_min, p.glucose_min),
                min(self.config.glucose_max, p.glucose_max),
            )),
            insulin_action=max(float(state.insulin_action), x_min),
            insulin_state=max(float(state.insulin_state), 0.0),
            gut_carbohydrate=max(float(state.gut_carbohydrate), 0.0),
            interstitial_glucose=float(np.clip(
                state.interstitial_glucose,
                max(self.config.glucose_min, p.glucose_min),
                min(self.config.glucose_max, p.glucose_max),
            )),
        )

    # ------------------------------------------------------------------
    # Dynamics and measurement models
    # ------------------------------------------------------------------

    def _transition(
        self,
        state_vector: np.ndarray,
        insulin_u: float,
        carbohydrate_c: float,
        dt_minutes: float,
    ) -> np.ndarray:
        """Evaluate one GECO transition without mutating the filter model."""
        temp_model = GECOModel(
            params=self.model.params,
            initial_state=self._vector_to_state(state_vector),
        )
        output = temp_model.step(
            insulin_u=insulin_u,
            carbohydrate_c=carbohydrate_c,
            observation_minutes=dt_minutes,
        )
        return self._state_to_vector(output.state)

    @staticmethod
    def _measurement_function(state_vector: np.ndarray) -> float:
        """CGM measurement model: y = interstitial glucose."""
        return float(state_vector[4])

    def _numerical_jacobian(
        self,
        state_vector: np.ndarray,
        insulin_u: float,
        carbohydrate_c: float,
        dt_minutes: float,
    ) -> np.ndarray:
        """
        Finite-difference state-transition Jacobian.

        F[i,j] = d f_i / d x_j
        """
        x = np.asarray(state_vector, dtype=float)
        base = self._transition(
            x, insulin_u, carbohydrate_c, dt_minutes
        )

        jacobian = np.zeros((self.STATE_DIM, self.STATE_DIM), dtype=float)

        for j in range(self.STATE_DIM):
            # Relative perturbation with a minimum absolute scale.
            delta = max(abs(x[j]) * 1e-5, 1e-6)

            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += delta
            x_minus[j] -= delta

            f_plus = self._transition(
                x_plus, insulin_u, carbohydrate_c, dt_minutes
            )
            f_minus = self._transition(
                x_minus, insulin_u, carbohydrate_c, dt_minutes
            )

            jacobian[:, j] = (f_plus - f_minus) / (2.0 * delta)

        return jacobian

    @staticmethod
    def _measurement_jacobian() -> np.ndarray:
        """Jacobian H for y = Gi."""
        h = np.zeros((1, 5), dtype=float)
        h[0, 4] = 1.0
        return h

    def _process_covariance(self, dt_minutes: float) -> np.ndarray:
        """
        Return process covariance.

        The configured variances are scaled by the duration ratio so
        that changing the observation interval does not silently keep
        exactly the same process-noise magnitude.
        """
        ratio = max(float(dt_minutes), 1e-9) / max(
            self.model.params.substep_minutes, 1e-9
        )
        q = np.asarray(self.config.process_variance, dtype=float) * ratio
        return np.diag(q)

    # ------------------------------------------------------------------
    # Numerical covariance handling
    # ------------------------------------------------------------------

    def _symmetrize_covariance(self) -> None:
        self.covariance = 0.5 * (
            self.covariance + self.covariance.T
        )

    def _ensure_valid_covariance(self) -> None:
        self._symmetrize_covariance()

        if not np.all(np.isfinite(self.covariance)):
            raise FloatingPointError("Covariance contains non-finite values.")

        eigenvalues = np.linalg.eigvalsh(self.covariance)

        if np.any(eigenvalues < -self.config.covariance_floor):
            raise FloatingPointError(
                "Covariance is not positive semi-definite."
            )

        # Project tiny negative eigenvalues / zero diagonal values away.
        diagonal = np.maximum(
            np.diag(self.covariance),
            self.config.covariance_floor,
        )
        self.covariance[np.diag_indices_from(self.covariance)] = diagonal
        self._symmetrize_covariance()

        condition = np.linalg.cond(self.covariance)
        if not np.isfinite(condition) or condition > self.config.max_condition_number:
            raise FloatingPointError("Covariance is numerically ill-conditioned.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        insulin_u: float = 0.0,
        carbohydrate_c: float = 0.0,
        observation_minutes: Optional[float] = None,
    ) -> GECOStateEstimate:
        """
        EKF prediction step.

        No measurement update is performed here.
        """
        if observation_minutes is None:
            observation_minutes = (
                self.model.params.substep_minutes
                * self.model.params.substeps_per_observation
            )

        dt = float(observation_minutes)
        if dt <= 0:
            raise ValueError("observation_minutes must be > 0.")

        x = self._state_to_vector(self.state)

        try:
            predicted_x = self._transition(
                x,
                insulin_u=float(insulin_u),
                carbohydrate_c=float(carbohydrate_c),
                dt_minutes=dt,
            )

            f = self._numerical_jacobian(
                x,
                insulin_u=float(insulin_u),
                carbohydrate_c=float(carbohydrate_c),
                dt_minutes=dt,
            )

            q = self._process_covariance(dt)

            predicted_p = (
                f @ self.covariance @ f.T
                + q
            )

            predicted_state = self._clamp_state(
                self._vector_to_state(predicted_x)
            )

            predicted_x = self._state_to_vector(predicted_state)

            predicted_p = 0.5 * (
                predicted_p + predicted_p.T
            )

            old_covariance = self.covariance
            self.covariance = predicted_p

            try:
                self._ensure_valid_covariance()
            except FloatingPointError:
                self.covariance = old_covariance
                if self.config.reset_on_numerical_failure:
                    self.reset(
                        state=predicted_state,
                        preserve_model_state=True,
                    )
                    return GECOStateEstimate(
                        state=self.state.copy(),
                        covariance=self.covariance.copy(),
                        measurement=None,
                        predicted_measurement=self._measurement_function(
                            predicted_x
                        ),
                        innovation=None,
                        innovation_variance=None,
                        normalized_innovation=None,
                        kalman_gain=None,
                        initialized=True,
                        updated=False,
                        numerical_reset=True,
                        metadata={"reason": "prediction_covariance_failure"},
                    )
                raise

            self.state = predicted_state

            predicted_measurement = self._measurement_function(predicted_x)

            return GECOStateEstimate(
                state=self.state.copy(),
                covariance=self.covariance.copy(),
                measurement=None,
                predicted_measurement=predicted_measurement,
                innovation=None,
                innovation_variance=None,
                normalized_innovation=None,
                kalman_gain=None,
                initialized=True,
                updated=False,
            )

        except Exception:
            if self.config.reset_on_numerical_failure:
                self.reset(
                    state=self.state,
                    preserve_model_state=True,
                )
            raise

    # ------------------------------------------------------------------
    # Measurement update
    # ------------------------------------------------------------------

    def update(
        self,
        measurement: Optional[float],
        timestamp: Optional[Any] = None,
    ) -> GECOStateEstimate:
        """
        Assimilate one CGM measurement.

        Invalid/missing measurements are treated as a skipped update,
        not as a zero glucose measurement.
        """
        predicted_x = self._state_to_vector(self.state)
        predicted_measurement = self._measurement_function(predicted_x)

        if measurement is None or not np.isfinite(float(measurement)):
            estimate = GECOStateEstimate(
                state=self.state.copy(),
                covariance=self.covariance.copy(),
                measurement=None,
                predicted_measurement=predicted_measurement,
                innovation=None,
                innovation_variance=None,
                normalized_innovation=None,
                kalman_gain=None,
                initialized=self.initialized,
                updated=False,
                timestamp=timestamp,
                metadata={"reason": "invalid_or_missing_measurement"},
            )
            self.last_estimate = estimate
            return estimate

        y = float(measurement)
        h = self._measurement_jacobian()

        innovation = y - predicted_measurement

        s = float(
            (h @ self.covariance @ h.T)[0, 0]
            + self.config.measurement_variance
        )

        if not np.isfinite(s) or s <= 0:
            if self.config.reset_on_numerical_failure:
                self.reset(
                    state=self.state,
                    preserve_model_state=True,
                )
                estimate = GECOStateEstimate(
                    state=self.state.copy(),
                    covariance=self.covariance.copy(),
                    measurement=y,
                    predicted_measurement=predicted_measurement,
                    innovation=innovation,
                    innovation_variance=None,
                    normalized_innovation=None,
                    kalman_gain=None,
                    initialized=True,
                    updated=False,
                    numerical_reset=True,
                    timestamp=timestamp,
                    metadata={"reason": "innovation_variance_failure"},
                )
                self.last_estimate = estimate
                return estimate
            raise FloatingPointError("Innovation variance is invalid.")

        k = (self.covariance @ h.T) / s

        updated_x = predicted_x + (
            k[:, 0] * innovation
        )

        # Joseph-form covariance update for numerical stability:
        # P = (I-KH)P(I-KH)' + KRK'
        identity = np.eye(self.STATE_DIM)
        kh = k @ h

        r = float(self.config.measurement_variance)

        updated_p = (
            (identity - kh)
            @ self.covariance
            @ (identity - kh).T
            + k * r @ k.T
        )

        updated_state = self._clamp_state(
            self._vector_to_state(updated_x)
        )

        old_state = self.state
        old_covariance = self.covariance

        self.state = updated_state
        self.covariance = 0.5 * (updated_p + updated_p.T)

        try:
            self._ensure_valid_covariance()
        except FloatingPointError:
            self.state = old_state
            self.covariance = old_covariance

            if self.config.reset_on_numerical_failure:
                self.reset(
                    state=self.state,
                    preserve_model_state=True,
                )
                estimate = GECOStateEstimate(
                    state=self.state.copy(),
                    covariance=self.covariance.copy(),
                    measurement=y,
                    predicted_measurement=predicted_measurement,
                    innovation=innovation,
                    innovation_variance=s,
                    normalized_innovation=innovation / np.sqrt(s),
                    kalman_gain=k[:, 0].copy(),
                    initialized=True,
                    updated=False,
                    numerical_reset=True,
                    timestamp=timestamp,
                    metadata={"reason": "updated_covariance_failure"},
                )
                self.last_estimate = estimate
                return estimate
            raise

        self.update_count += 1

        normalized_innovation = innovation / np.sqrt(s)

        estimate = GECOStateEstimate(
            state=self.state.copy(),
            covariance=self.covariance.copy(),
            measurement=y,
            predicted_measurement=predicted_measurement,
            innovation=innovation,
            innovation_variance=s,
            normalized_innovation=normalized_innovation,
            kalman_gain=k[:, 0].copy(),
            initialized=self.initialized,
            updated=True,
            timestamp=timestamp,
        )

        self.last_estimate = estimate
        return estimate

    def step(
        self,
        measurement: Optional[float],
        insulin_u: float = 0.0,
        carbohydrate_c: float = 0.0,
        observation_minutes: Optional[float] = None,
        timestamp: Optional[Any] = None,
    ) -> GECOStateEstimate:
        """
        Perform one complete EKF cycle:

            predict -> update

        The insulin and carbohydrate inputs are the inputs known at the
        current observation interval.
        """
        self.predict(
            insulin_u=insulin_u,
            carbohydrate_c=carbohydrate_c,
            observation_minutes=observation_minutes,
        )

        return self.update(
            measurement=measurement,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Sequence filtering
    # ------------------------------------------------------------------

    def filter_sequence(
        self,
        measurements: Sequence[Optional[float]],
        insulin_inputs: Optional[Sequence[float]] = None,
        carbohydrate_inputs: Optional[Sequence[float]] = None,
        timestamps: Optional[Sequence[Any]] = None,
        observation_minutes: Optional[float] = None,
    ) -> list[GECOStateEstimate]:
        """Run the EKF over a complete chronological sequence."""
        measurements = list(measurements)
        n = len(measurements)

        if insulin_inputs is None:
            insulin_inputs = [0.0] * n
        else:
            insulin_inputs = list(insulin_inputs)

        if carbohydrate_inputs is None:
            carbohydrate_inputs = [0.0] * n
        else:
            carbohydrate_inputs = list(carbohydrate_inputs)

        if len(insulin_inputs) != n:
            raise ValueError("insulin_inputs length must match measurements.")

        if len(carbohydrate_inputs) != n:
            raise ValueError(
                "carbohydrate_inputs length must match measurements."
            )

        if timestamps is not None:
            timestamps = list(timestamps)
            if len(timestamps) != n:
                raise ValueError(
                    "timestamps length must match measurements."
                )

        estimates = []

        for i in range(n):
            timestamp = None if timestamps is None else timestamps[i]

            estimates.append(
                self.step(
                    measurement=measurements[i],
                    insulin_u=insulin_inputs[i],
                    carbohydrate_c=carbohydrate_inputs[i],
                    observation_minutes=observation_minutes,
                    timestamp=timestamp,
                )
            )

        return estimates

    # ------------------------------------------------------------------
    # Reset and diagnostics
    # ------------------------------------------------------------------

    def reset(
        self,
        state: Optional[GECOState] = None,
        covariance: Optional[np.ndarray] = None,
        preserve_model_state: bool = False,
    ) -> GECOState:
        """
        Reset filter state/covariance.

        ``preserve_model_state`` is retained for explicitness when the
        estimator is recovering from a numerical issue. The estimator
        itself does not require the GECOModel object's state to be
        mutated by every filter update.
        """
        if state is None:
            state = self.model.get_state()

        self.state = self._clamp_state(state)

        if covariance is None:
            covariance = np.diag(
                np.asarray(
                    self.config.initial_covariance,
                    dtype=float,
                )
            )
        else:
            covariance = np.asarray(covariance, dtype=float)

            if covariance.shape != (5, 5):
                raise ValueError("covariance must have shape (5, 5).")

        self.covariance = covariance.copy()
        self._ensure_valid_covariance()

        self.initialized = True
        self.reset_count += 1
        self.last_estimate = None

        return self.state.copy()

    def get_state(self) -> GECOState:
        return self.state.copy()

    def get_covariance(self) -> np.ndarray:
        return self.covariance.copy()

    def innovation_statistics(self) -> Dict[str, Any]:
        """Return diagnostics from the most recent measurement update."""
        if self.last_estimate is None:
            return {
                "available": False,
                "update_count": self.update_count,
                "reset_count": self.reset_count,
            }

        return {
            "available": True,
            "innovation": self.last_estimate.innovation,
            "innovation_variance": self.last_estimate.innovation_variance,
            "normalized_innovation": self.last_estimate.normalized_innovation,
            "update_count": self.update_count,
            "reset_count": self.reset_count,
        }


def create_state_estimator(
    model: GECOModel,
    config: Optional[GECOFilterConfig] = None,
    initial_state: Optional[GECOState] = None,
) -> GECOStateEstimator:
    """Convenience factory for a GECO state estimator."""
    return GECOStateEstimator(
        model=model,
        config=config,
        initial_state=initial_state,
    )


__all__ = [
    "GECOFilterConfig",
    "GECOStateEstimate",
    "GECOStateEstimator",
    "create_state_estimator",
]
