"""
Parameter estimation utilities for the GECO physiological model.

This module estimates the GECO parameters used by the DT1 physiological
twin from chronological glucose / CGM, insulin, and carbohydrate data.

The GECO study describes fitting four quantities:
    - egp / p1
    - p1
    - kc
    - p3

using deterministic least-squares starts and causal 30-minute mechanistic
residuals. The exact bounds and dataset eligibility used in the paper are
kept configurable here rather than silently hard-coded as universal
physiological constants.

This module is an engineering/research implementation. It is not a
clinical parameter-identification method.

Expected input columns:
    glucose or cgm
    insulin
    meal / carbohydrate

The estimator uses only information available up to each fitting origin
when generating a causal forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover
    least_squares = None

try:
    from .geco_model import GECOModel, GECOParameters
except ImportError:  # direct script/module usage
    from geco_model import GECOModel, GECOParameters


@dataclass(frozen=True)
class GECOParameterBounds:
    """
    Bounds for the four fitted GECO quantities.

    The GECO paper specifies that egp/p1, p1, kc and p3 are fitted
    within prespecified log bounds. Because those exact numerical bounds
    should come from the selected experimental configuration, this class
    makes them explicit and configurable.

    Values are bounds in ordinary parameter space. The optimizer itself
    works in log-space.
    """

    egp_over_p1: Tuple[float, float] = (1e-4, 100.0)
    p1: Tuple[float, float] = (1e-5, 10.0)
    kc: Tuple[float, float] = (1e-6, 10.0)
    p3: Tuple[float, float] = (1e-7, 10.0)

    def validate(self) -> None:
        for name in ("egp_over_p1", "p1", "kc", "p3"):
            low, high = getattr(self, name)
            if low <= 0 or high <= 0 or low >= high:
                raise ValueError(
                    f"{name} bounds must satisfy 0 < low < high; "
                    f"got ({low}, {high})."
                )


@dataclass(frozen=True)
class GECOFixedParameters:
    """
    Parameters not estimated by this module.

    These values should be set from the chosen GECO configuration,
    literature implementation, or an explicitly documented experiment.
    """

    p2: float = 0.01
    ki: float = 0.01
    ku: float = 1.0
    ka: float = 0.01
    tau_i: float = 10.0
    basal_glucose: float = 120.0
    basal_insulin: float = 0.0
    substep_minutes: float = 1.0
    substeps_per_observation: int = 5
    glucose_min: float = 20.0
    glucose_max: float = 600.0
    x_min_offset: float = 0.5


@dataclass(frozen=True)
class GECOParameterVector:
    """The four parameters estimated by GECO fitting."""

    egp_over_p1: float
    p1: float
    kc: float
    p3: float

    @property
    def egp(self) -> float:
        return self.egp_over_p1 * self.p1

    def as_dict(self) -> Dict[str, float]:
        return {
            "egp_over_p1": float(self.egp_over_p1),
            "p1": float(self.p1),
            "kc": float(self.kc),
            "p3": float(self.p3),
            "egp": float(self.egp),
        }

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.egp_over_p1, self.p1, self.kc, self.p3],
            dtype=float,
        )


@dataclass(frozen=True)
class GECOFitOrigin:
    """One causal forecast origin used during parameter fitting."""

    index: int
    glucose: float
    insulin: float
    carbohydrate: float


@dataclass
class GECOFitResult:
    """Result of GECO parameter estimation."""

    parameters: GECOParameterVector
    fixed_parameters: GECOFixedParameters

    success: bool
    cost: float
    rmse: float
    mae: float
    n_residuals: int
    n_origins: int

    optimizer_status: int = 0
    optimizer_message: str = ""
    n_function_evaluations: int = 0

    residuals: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )

    initial_parameters: Optional[GECOParameterVector] = None
    bounds: Optional[GECOParameterBounds] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameters": self.parameters.as_dict(),
            "fixed_parameters": self.fixed_parameters.__dict__.copy(),
            "success": bool(self.success),
            "cost": float(self.cost),
            "rmse": float(self.rmse),
            "mae": float(self.mae),
            "n_residuals": int(self.n_residuals),
            "n_origins": int(self.n_origins),
            "optimizer_status": int(self.optimizer_status),
            "optimizer_message": self.optimizer_message,
            "n_function_evaluations": int(self.n_function_evaluations),
            "initial_parameters": (
                None
                if self.initial_parameters is None
                else self.initial_parameters.as_dict()
            ),
            "bounds": (
                None
                if self.bounds is None
                else {
                    name: getattr(self.bounds, name)
                    for name in (
                        "egp_over_p1",
                        "p1",
                        "kc",
                        "p3",
                    )
                }
            ),
            "metadata": dict(self.metadata),
        }


def _as_1d(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def _parameter_from_array(values: Sequence[float]) -> GECOParameterVector:
    values = np.asarray(values, dtype=float)
    if values.shape != (4,):
        raise ValueError("GECO parameter vector must contain four values.")
    return GECOParameterVector(
        egp_over_p1=float(values[0]),
        p1=float(values[1]),
        kc=float(values[2]),
        p3=float(values[3]),
    )


def _build_model(
    fitted: GECOParameterVector,
    fixed: GECOFixedParameters,
    initial_glucose: float,
    initial_interstitial: Optional[float] = None,
) -> GECOModel:
    params = GECOParameters(
        egp_over_p1=fitted.egp_over_p1,
        p1=fitted.p1,
        kc=fitted.kc,
        p3=fitted.p3,
        p2=fixed.p2,
        ki=fixed.ki,
        ku=fixed.ku,
        ka=fixed.ka,
        tau_i=fixed.tau_i,
        basal_glucose=fixed.basal_glucose,
        basal_insulin=fixed.basal_insulin,
        substep_minutes=fixed.substep_minutes,
        substeps_per_observation=fixed.substeps_per_observation,
        glucose_min=fixed.glucose_min,
        glucose_max=fixed.glucose_max,
        x_min_offset=fixed.x_min_offset,
    )

    state = None
    if initial_glucose is not None:
        state = model_state(
            glucose=float(initial_glucose),
            interstitial_glucose=(
                float(initial_interstitial)
                if initial_interstitial is not None
                else float(initial_glucose)
            ),
            insulin_state=fixed.basal_insulin,
        )

    return GECOModel(params=params, initial_state=state)


def model_state(
    glucose: float,
    interstitial_glucose: Optional[float] = None,
    insulin_state: float = 0.0,
):
    """Construct a GECOState without requiring callers to import it."""
    # Importing directly keeps this helper compatible with both package
    # and standalone execution.
    try:
        from .geco_model import GECOState
    except ImportError:
        from geco_model import GECOState

    return GECOState(
        glucose=float(glucose),
        insulin_action=0.0,
        insulin_state=float(insulin_state),
        gut_carbohydrate=0.0,
        interstitial_glucose=(
            float(glucose)
            if interstitial_glucose is None
            else float(interstitial_glucose)
        ),
    )


class GECOParameterEstimator:
    """
    Estimate GECO parameters using causal mechanistic forecasts.

    The estimator fits:
        [egp/p1, p1, kc, p3]

    while holding the remaining model parameters fixed.

    Parameters
    ----------
    fixed_parameters:
        Fixed GECO configuration.
    bounds:
        Positive parameter bounds.
    forecast_horizon_minutes:
        Forecast horizon. GECO's study used a 30-minute mechanistic
        forecast for fitting.
    forecast_step_minutes:
        Interval between forecast points. Default is five minutes,
        matching the common CGM/simulation interval.
    min_origins:
        Minimum number of valid forecast origins.
    """

    PARAMETER_NAMES = (
        "egp_over_p1",
        "p1",
        "kc",
        "p3",
    )

    def __init__(
        self,
        fixed_parameters: Optional[GECOFixedParameters] = None,
        bounds: Optional[GECOParameterBounds] = None,
        forecast_horizon_minutes: float = 30.0,
        forecast_step_minutes: float = 5.0,
        min_origins: int = 5,
    ):
        if least_squares is None:
            raise ImportError(
                "scipy is required for GECO parameter estimation. "
                "Install it with: pip install scipy"
            )

        self.fixed_parameters = fixed_parameters or GECOFixedParameters()
        self.bounds = bounds or GECOParameterBounds()
        self.bounds.validate()

        if forecast_horizon_minutes <= 0:
            raise ValueError("forecast_horizon_minutes must be > 0.")
        if forecast_step_minutes <= 0:
            raise ValueError("forecast_step_minutes must be > 0.")
        if min_origins < 1:
            raise ValueError("min_origins must be >= 1.")

        self.forecast_horizon_minutes = float(forecast_horizon_minutes)
        self.forecast_step_minutes = float(forecast_step_minutes)
        self.min_origins = int(min_origins)

    def _log_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lows = []
        highs = []
        for name in self.PARAMETER_NAMES:
            low, high = getattr(self.bounds, name)
            lows.append(np.log(low))
            highs.append(np.log(high))
        return np.asarray(lows), np.asarray(highs)

    def _validate_inputs(
        self,
        glucose: Sequence[float],
        insulin: Sequence[float],
        carbohydrate: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = _as_1d(glucose, "glucose")
        u = _as_1d(insulin, "insulin")
        c = _as_1d(carbohydrate, "carbohydrate")

        if not (len(g) == len(u) == len(c)):
            raise ValueError(
                "glucose, insulin and carbohydrate must have the same length."
            )

        return g, u, c

    def _forecast_offsets(self) -> int:
        ratio = self.forecast_horizon_minutes / self.forecast_step_minutes
        offsets = int(round(ratio))
        if offsets < 1:
            raise ValueError("Forecast horizon must contain at least one step.")
        return offsets

    def _build_origins(
        self,
        glucose: np.ndarray,
        insulin: np.ndarray,
        carbohydrate: np.ndarray,
    ) -> List[GECOFitOrigin]:
        """
        Select causal forecast origins.

        At origin k:
            - data up to k are available;
            - future observations k+1 ... k+H are used only as targets;
            - future insulin/meal values are NOT supplied to the model.

        This is a causal evaluation design.
        """
        horizon_steps = self._forecast_offsets()
        origins: List[GECOFitOrigin] = []

        for k in range(0, len(glucose) - horizon_steps):
            values = (glucose[k], insulin[k], carbohydrate[k])
            if not all(np.isfinite(v) for v in values):
                continue

            future = glucose[k + 1 : k + horizon_steps + 1]
            if len(future) < horizon_steps:
                continue
            if not np.all(np.isfinite(future)):
                continue

            origins.append(
                GECOFitOrigin(
                    index=k,
                    glucose=float(glucose[k]),
                    insulin=float(insulin[k]),
                    carbohydrate=float(carbohydrate[k]),
                )
            )

        return origins

    def _forecast_from_origin(
        self,
        fitted: GECOParameterVector,
        origin: GECOFitOrigin,
    ) -> float:
        """
        Produce one causal 30-minute mechanistic forecast.

        Future meal/insulin inputs are not available to the forecast, so
        the model holds the current inputs at their last known values.
        This can be changed through the estimator configuration in future
        versions if a different causal policy is required.
        """
        model = _build_model(
            fitted=fitted,
            fixed=self.fixed_parameters,
            initial_glucose=origin.glucose,
        )

        # Advance in forecast_step_minutes until the requested horizon.
        n_steps = int(round(
            self.forecast_horizon_minutes / self.forecast_step_minutes
        ))

        output = None
        for step_index in range(n_steps):
            # Causal hold: only information known at the origin is used.
            output = model.step(
                insulin_u=origin.insulin,
                carbohydrate_c=origin.carbohydrate,
                observation_minutes=self.forecast_step_minutes,
            )

        if output is None:
            raise RuntimeError("No GECO forecast was generated.")

        return float(output.glucose)

    def residual_vector(
        self,
        log_parameters: Sequence[float],
        glucose: Sequence[float],
        insulin: Sequence[float],
        carbohydrate: Sequence[float],
        origins: Optional[Sequence[GECOFitOrigin]] = None,
    ) -> np.ndarray:
        """
        Calculate causal forecast residuals for a log-parameter vector.

        Residual:
            predicted_glucose(k + H) - observed_glucose(k + H)
        """
        fitted = _parameter_from_array(np.exp(np.asarray(log_parameters)))

        g, u, c = self._validate_inputs(glucose, insulin, carbohydrate)

        if origins is None:
            origins = self._build_origins(g, u, c)

        horizon_steps = self._forecast_offsets()
        residuals: List[float] = []

        for origin in origins:
            target_index = origin.index + horizon_steps
            if target_index >= len(g):
                continue

            prediction = self._forecast_from_origin(fitted, origin)
            target = float(g[target_index])

            if np.isfinite(prediction) and np.isfinite(target):
                residuals.append(prediction - target)

        if not residuals:
            return np.empty(0, dtype=float)

        return np.asarray(residuals, dtype=float)

    def fit(
        self,
        glucose: Sequence[float],
        insulin: Sequence[float],
        carbohydrate: Sequence[float],
        initial_parameters: Optional[GECOParameterVector] = None,
        additional_starts: Optional[Sequence[GECOParameterVector]] = None,
        max_nfev: int = 100,
        loss: str = "linear",
        f_scale: float = 1.0,
    ) -> GECOFitResult:
        """
        Fit GECO parameters.

        Parameters
        ----------
        glucose, insulin, carbohydrate:
            Chronological physiological inputs.
        initial_parameters:
            First deterministic starting point.
        additional_starts:
            Optional extra deterministic starting points.
        max_nfev:
            Maximum function evaluations per optimization start.
        loss:
            scipy least_squares loss.
        f_scale:
            scipy robust-loss scale.

        Returns
        -------
        GECOFitResult
            Best result among the requested deterministic starts.
        """
        g, u, c = self._validate_inputs(glucose, insulin, carbohydrate)
        origins = self._build_origins(g, u, c)

        if len(origins) < self.min_origins:
            raise ValueError(
                f"Only {len(origins)} valid causal origins are available; "
                f"at least {self.min_origins} are required."
            )

        starts = []

        if initial_parameters is not None:
            starts.append(initial_parameters)
        else:
            starts.append(self.default_initial_parameters())

        if additional_starts is not None:
            starts.extend(additional_starts)

        lower_log, upper_log = self._log_bounds()

        best = None
        best_result = None

        for start in starts:
            start_vec = start.as_array()

            for i, value in enumerate(start_vec):
                low = np.exp(lower_log[i])
                high = np.exp(upper_log[i])
                if not (low <= value <= high):
                    raise ValueError(
                        f"Initial value for {self.PARAMETER_NAMES[i]}={value} "
                        f"is outside its configured bounds ({low}, {high})."
                    )

            x0 = np.log(start_vec)

            result = least_squares(
                fun=lambda theta: self.residual_vector(
                    theta,
                    g,
                    u,
                    c,
                    origins=origins,
                ),
                x0=x0,
                bounds=(lower_log, upper_log),
                max_nfev=max_nfev,
                loss=loss,
                f_scale=f_scale,
            )

            residuals = self.residual_vector(
                result.x,
                g,
                u,
                c,
                origins=origins,
            )

            cost = float(0.5 * np.sum(residuals ** 2))
            rmse = (
                float(np.sqrt(np.mean(residuals ** 2)))
                if len(residuals)
                else float("nan")
            )
            mae = (
                float(np.mean(np.abs(residuals)))
                if len(residuals)
                else float("nan")
            )

            if best is None or cost < best:
                best = cost
                best_result = (
                    result,
                    residuals,
                    rmse,
                    mae,
                    start,
                )

        if best_result is None:
            raise RuntimeError("GECO parameter optimization produced no result.")

        result, residuals, rmse, mae, winning_start = best_result
        fitted = _parameter_from_array(np.exp(result.x))

        return GECOFitResult(
            parameters=fitted,
            fixed_parameters=self.fixed_parameters,
            success=bool(result.success),
            cost=float(0.5 * np.sum(residuals ** 2)),
            rmse=rmse,
            mae=mae,
            n_residuals=int(len(residuals)),
            n_origins=int(len(origins)),
            optimizer_status=int(result.status),
            optimizer_message=str(result.message),
            n_function_evaluations=int(result.nfev),
            residuals=residuals,
            initial_parameters=winning_start,
            bounds=self.bounds,
            metadata={
                "forecast_horizon_minutes": self.forecast_horizon_minutes,
                "forecast_step_minutes": self.forecast_step_minutes,
                "parameterization": "log_space",
                "objective": "causal_mechanistic_forecast_residual",
                "optimizer": "scipy.optimize.least_squares",
            },
        )

    def default_initial_parameters(self) -> GECOParameterVector:
        """
        Return a safe interior starting point based on configured bounds.

        The geometric midpoint is used because the fitting problem is
        parameterized in log-space.
        """
        values = []
        for name in self.PARAMETER_NAMES:
            low, high = getattr(self.bounds, name)
            values.append(np.sqrt(low * high))

        return GECOParameterVector(*values)

    def fit_from_dataframe(
        self,
        dataframe,
        glucose_column: str = "glucose",
        cgm_column: Optional[str] = None,
        insulin_column: str = "insulin",
        carbohydrate_column: str = "meal",
        **fit_kwargs,
    ) -> GECOFitResult:
        """
        Fit from a pandas-like DataFrame.

        ``cgm_column`` is optional. The GECO physiological model uses a
        glucose observation for fitting; when glucose is unavailable,
        CGM can be supplied as the observed glucose channel.
        """
        if glucose_column in dataframe.columns:
            glucose = dataframe[glucose_column].to_numpy()
        elif cgm_column is not None and cgm_column in dataframe.columns:
            glucose = dataframe[cgm_column].to_numpy()
        elif "cgm" in dataframe.columns:
            glucose = dataframe["cgm"].to_numpy()
        else:
            raise KeyError(
                f"Neither '{glucose_column}' nor a usable CGM column was found."
            )

        if insulin_column not in dataframe.columns:
            raise KeyError(f"Missing insulin column: {insulin_column}")

        if carbohydrate_column not in dataframe.columns:
            raise KeyError(
                f"Missing carbohydrate/meal column: {carbohydrate_column}"
            )

        return self.fit(
            glucose=glucose,
            insulin=dataframe[insulin_column].to_numpy(),
            carbohydrate=dataframe[carbohydrate_column].to_numpy(),
            **fit_kwargs,
        )


def estimate_geco_parameters(
    glucose: Sequence[float],
    insulin: Sequence[float],
    carbohydrate: Sequence[float],
    *,
    fixed_parameters: Optional[GECOFixedParameters] = None,
    bounds: Optional[GECOParameterBounds] = None,
    initial_parameters: Optional[GECOParameterVector] = None,
    additional_starts: Optional[Sequence[GECOParameterVector]] = None,
    forecast_horizon_minutes: float = 30.0,
    forecast_step_minutes: float = 5.0,
    min_origins: int = 5,
    max_nfev: int = 100,
) -> GECOFitResult:
    """Convenience wrapper around ``GECOParameterEstimator.fit``."""
    estimator = GECOParameterEstimator(
        fixed_parameters=fixed_parameters,
        bounds=bounds,
        forecast_horizon_minutes=forecast_horizon_minutes,
        forecast_step_minutes=forecast_step_minutes,
        min_origins=min_origins,
    )
    return estimator.fit(
        glucose=glucose,
        insulin=insulin,
        carbohydrate=carbohydrate,
        initial_parameters=initial_parameters,
        additional_starts=additional_starts,
        max_nfev=max_nfev,
    )


__all__ = [
    "GECOParameterBounds",
    "GECOFixedParameters",
    "GECOParameterVector",
    "GECOFitOrigin",
    "GECOFitResult",
    "GECOParameterEstimator",
    "estimate_geco_parameters",
]
