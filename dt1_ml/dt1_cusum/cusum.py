"""
CUSUM sequential anomaly monitoring for DT1.

This module implements a research-oriented CUSUM monitor for residual or
innovation signals produced by the GECO physiological model.

Conceptual DT1 monitoring path:

    GECO prediction
          |
          v
       residual
          |
          v
    innovation / whitening
          |
          v
    CUSUM statistic
          |
          v
    anomaly evidence

The implementation supports:
    - one-sided positive CUSUM
    - one-sided negative CUSUM
    - two-sided CUSUM
    - residual normalization
    - optional AR(1) innovation whitening
    - reset after gaps/resets
    - configurable reference value and threshold
    - dynamic observation-noise variance
    - sequential stateful monitoring

This module generates anomaly evidence. It does not make a final
clinical, safety, or pump-control decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np


class CUSUMDirection(str, Enum):
    """Direction monitored by CUSUM."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    TWO_SIDED = "two_sided"


class CUSUMState(str, Enum):
    """Monitoring state derived from the CUSUM statistic."""

    NORMAL = "normal"
    WARNING = "warning"
    ALARM = "alarm"
    RESET = "reset"


@dataclass(frozen=True)
class CUSUMConfig:
    """
    Configuration for sequential CUSUM monitoring.

    Parameters
    ----------
    reference:
        Reference value k in normalized innovation units.

    threshold:
        Alarm threshold h in CUSUM units.

    warning_fraction:
        Fraction of threshold at which WARNING evidence is emitted.

    direction:
        Positive, negative, or two-sided monitoring.

    nominal_variance:
        Nominal observation variance. A value of 81 corresponds to a
        nominal standard deviation of 9 mg/dL.

    whitening_rho:
        AR(1) correlation coefficient used for innovation whitening.
        Set to zero to disable whitening.

    normalization_scale:
        Residual scale used when the input signal is in glucose units.
        If None, the raw signal is used.

    reset_on_alarm:
        Whether the cumulative statistic is reset immediately after
        an alarm.

    reset_after_gap:
        Whether the previous innovation is cleared when the caller
        indicates a gap/reset.
    """

    reference: float = 0.5
    threshold: float = 5.0
    warning_fraction: float = 0.6

    direction: CUSUMDirection = CUSUMDirection.TWO_SIDED

    nominal_variance: float = 81.0
    whitening_rho: float = 0.0

    normalization_scale: Optional[float] = None

    reset_on_alarm: bool = False
    reset_after_gap: bool = True

    min_variance: float = 1e-6
    max_abs_innovation: Optional[float] = None

    def __post_init__(self) -> None:
        if self.reference < 0:
            raise ValueError("reference must be >= 0.")

        if self.threshold <= 0:
            raise ValueError("threshold must be > 0.")

        if not 0 < self.warning_fraction < 1:
            raise ValueError("warning_fraction must be between 0 and 1.")

        if self.nominal_variance <= 0:
            raise ValueError("nominal_variance must be > 0.")

        if not -0.98 <= self.whitening_rho <= 0.98:
            raise ValueError("whitening_rho must be in [-0.98, 0.98].")

        if self.normalization_scale is not None:
            if self.normalization_scale <= 0:
                raise ValueError("normalization_scale must be > 0.")

        if self.min_variance <= 0:
            raise ValueError("min_variance must be > 0.")

        if self.max_abs_innovation is not None:
            if self.max_abs_innovation <= 0:
                raise ValueError("max_abs_innovation must be > 0.")


@dataclass(frozen=True)
class CUSUMResult:
    """Result of processing one innovation."""

    step: int
    raw_residual: float
    normalized_residual: float
    whitened_innovation: float

    positive_statistic: float
    negative_statistic: float
    statistic: float

    state: CUSUMState
    alarm: bool
    warning: bool
    reset: bool

    variance: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "raw_residual": self.raw_residual,
            "normalized_residual": self.normalized_residual,
            "whitened_innovation": self.whitened_innovation,
            "positive_statistic": self.positive_statistic,
            "negative_statistic": self.negative_statistic,
            "statistic": self.statistic,
            "state": self.state.value,
            "alarm": self.alarm,
            "warning": self.warning,
            "reset": self.reset,
            "variance": self.variance,
            "metadata": dict(self.metadata),
        }


@dataclass
class CUSUMSummary:
    """Summary of a complete CUSUM monitoring sequence."""

    results: List[CUSUMResult]
    total_samples: int
    warning_count: int
    alarm_count: int
    first_alarm_step: Optional[int]
    maximum_statistic: float
    final_statistic: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "warning_count": self.warning_count,
            "alarm_count": self.alarm_count,
            "first_alarm_step": self.first_alarm_step,
            "maximum_statistic": self.maximum_statistic,
            "final_statistic": self.final_statistic,
            "results": [r.to_dict() for r in self.results],
        }


class CUSUM:
    """
    Stateful sequential CUSUM monitor.

    The default convention expects a residual:

        residual = observed - predicted

    Positive values therefore indicate that observed glucose is above
    the model prediction, while negative values indicate that observed
    glucose is below prediction.

    For two-sided monitoring, both directions are accumulated separately.
    """

    def __init__(
        self,
        config: Optional[CUSUMConfig] = None,
    ) -> None:
        self.config = config or CUSUMConfig()
        self.reset()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all cumulative monitoring state."""
        self._positive = 0.0
        self._negative = 0.0
        self._previous_innovation: Optional[float] = None
        self._step = 0
        self._last_state = CUSUMState.RESET

    @property
    def positive_statistic(self) -> float:
        return float(self._positive)

    @property
    def negative_statistic(self) -> float:
        return float(self._negative)

    @property
    def statistic(self) -> float:
        if self.config.direction == CUSUMDirection.POSITIVE:
            return float(self._positive)

        if self.config.direction == CUSUMDirection.NEGATIVE:
            return float(self._negative)

        return float(max(self._positive, self._negative))

    # ------------------------------------------------------------------
    # Normalization / whitening
    # ------------------------------------------------------------------

    def _normalize_residual(
        self,
        residual: float,
        variance: float,
    ) -> float:
        """
        Normalize a residual.

        If normalization_scale is supplied, residuals are divided by
        that scale. Otherwise, they are standardized using sqrt(variance).
        """
        if self.config.normalization_scale is not None:
            return residual / self.config.normalization_scale

        std = np.sqrt(max(variance, self.config.min_variance))
        return residual / std

    def _whiten(
        self,
        innovation: float,
        reset: bool = False,
    ) -> float:
        """
        Apply optional AR(1) innovation whitening.

        z_k = (z_raw_k - rho * z_raw_(k-1)) / sqrt(1-rho^2)

        The first valid innovation after a reset/gap is left unwhitened.
        """
        rho = self.config.whitening_rho

        if reset or self._previous_innovation is None or abs(rho) < 1e-12:
            whitened = innovation
        else:
            denominator = np.sqrt(max(1.0 - rho * rho, 1e-12))
            whitened = (
                innovation - rho * self._previous_innovation
            ) / denominator

        self._previous_innovation = innovation

        if self.config.max_abs_innovation is not None:
            whitened = float(
                np.clip(
                    whitened,
                    -self.config.max_abs_innovation,
                    self.config.max_abs_innovation,
                )
            )

        return float(whitened)

    # ------------------------------------------------------------------
    # Sequential update
    # ------------------------------------------------------------------

    def update(
        self,
        residual: float,
        variance: Optional[float] = None,
        reset: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CUSUMResult:
        """
        Process one residual/innovation.

        Parameters
        ----------
        residual:
            Model residual, normally observed - predicted.

        variance:
            Observation/innovation variance. If omitted, nominal_variance
            is used.

        reset:
            Indicates that the sample follows a gap or monitoring reset.
        """
        raw = float(residual)

        if not np.isfinite(raw):
            raise ValueError("residual must be finite.")

        if variance is None:
            variance_value = self.config.nominal_variance
        else:
            variance_value = float(variance)

        if not np.isfinite(variance_value) or variance_value <= 0:
            raise ValueError("variance must be finite and > 0.")

        variance_value = max(
            variance_value,
            self.config.min_variance,
        )

        explicit_reset = bool(reset)

        if explicit_reset and self.config.reset_after_gap:
            self._positive = 0.0
            self._negative = 0.0
            self._previous_innovation = None

        normalized = self._normalize_residual(
            residual=raw,
            variance=variance_value,
        )

        whitened = self._whiten(
            innovation=normalized,
            reset=explicit_reset,
        )

        k = self.config.reference

        # Standard one-sided CUSUM recursions:
        #
        # S+ = max(0, S+ + z - k)
        # S- = max(0, S- - z - k)
        #
        self._positive = max(
            0.0,
            self._positive + whitened - k,
        )

        self._negative = max(
            0.0,
            self._negative - whitened - k,
        )

        current_statistic = self.statistic

        alarm = bool(current_statistic >= self.config.threshold)

        warning_threshold = (
            self.config.threshold * self.config.warning_fraction
        )
        warning = bool(
            current_statistic >= warning_threshold and not alarm
        )

        if alarm:
            state = CUSUMState.ALARM
        elif warning:
            state = CUSUMState.WARNING
        else:
            state = CUSUMState.NORMAL

        if explicit_reset:
            reset_flag = True
        else:
            reset_flag = False

        self._step += 1
        self._last_state = state

        result = CUSUMResult(
            step=self._step,
            raw_residual=raw,
            normalized_residual=float(normalized),
            whitened_innovation=float(whitened),
            positive_statistic=float(self._positive),
            negative_statistic=float(self._negative),
            statistic=float(current_statistic),
            state=state,
            alarm=alarm,
            warning=warning,
            reset=reset_flag,
            variance=float(variance_value),
            metadata=dict(metadata or {}),
        )

        if alarm and self.config.reset_on_alarm:
            self._positive = 0.0
            self._negative = 0.0

        return result

    # ------------------------------------------------------------------
    # Sequence processing
    # ------------------------------------------------------------------

    def update_many(
        self,
        residuals: Sequence[float],
        variances: Optional[Sequence[float]] = None,
        reset_indices: Optional[Iterable[int]] = None,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> CUSUMSummary:
        """Process a complete residual sequence."""
        values = np.asarray(residuals, dtype=float).reshape(-1)

        if variances is not None:
            variance_values = np.asarray(
                variances,
                dtype=float,
            ).reshape(-1)

            if variance_values.size != values.size:
                raise ValueError(
                    "variances must have the same length as residuals."
                )
        else:
            variance_values = None

        reset_set = set(int(i) for i in (reset_indices or []))

        if metadata is not None and len(metadata) != values.size:
            raise ValueError(
                "metadata must have the same length as residuals."
            )

        results: List[CUSUMResult] = []

        for i, residual in enumerate(values):
            variance = (
                None
                if variance_values is None
                else float(variance_values[i])
            )

            item_metadata = (
                None
                if metadata is None
                else dict(metadata[i])
            )

            result = self.update(
                residual=float(residual),
                variance=variance,
                reset=(i in reset_set),
                metadata=item_metadata,
            )

            results.append(result)

        alarm_steps = [
            result.step
            for result in results
            if result.alarm
        ]

        statistics = (
            [result.statistic for result in results]
            if results
            else []
        )

        return CUSUMSummary(
            results=results,
            total_samples=len(results),
            warning_count=sum(r.warning for r in results),
            alarm_count=sum(r.alarm for r in results),
            first_alarm_step=(
                alarm_steps[0]
                if alarm_steps
                else None
            ),
            maximum_statistic=(
                float(max(statistics))
                if statistics
                else 0.0
            ),
            final_statistic=(
                float(results[-1].statistic)
                if results
                else 0.0
            ),
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def positive_alarm(self) -> bool:
        """Return whether the positive CUSUM is above threshold."""
        return bool(
            self._positive >= self.config.threshold
        )

    def negative_alarm(self) -> bool:
        """Return whether the negative CUSUM is above threshold."""
        return bool(
            self._negative >= self.config.threshold
        )

    def is_alarm(self) -> bool:
        """Return whether the active CUSUM statistic is in alarm."""
        return bool(self.statistic >= self.config.threshold)

    def state(self) -> CUSUMState:
        """Return the latest monitoring state."""
        return self._last_state


# ----------------------------------------------------------------------
# Functional helpers
# ----------------------------------------------------------------------

def compute_cusum(
    residuals: Sequence[float],
    config: Optional[CUSUMConfig] = None,
    variances: Optional[Sequence[float]] = None,
    reset_indices: Optional[Iterable[int]] = None,
) -> CUSUMSummary:
    """Functional wrapper for processing a residual sequence."""
    monitor = CUSUM(config=config)

    return monitor.update_many(
        residuals=residuals,
        variances=variances,
        reset_indices=reset_indices,
    )


def estimate_ar1_correlation(
    innovations: Sequence[float],
    clip_min: float = 0.0,
    clip_max: float = 0.98,
) -> float:
    """
    Estimate lag-1 correlation from a valid innovation sequence.

    The result is clipped to the supplied range. For GECO-style
    monitoring, the default range prevents unstable whitening.

    This estimator should be fitted only on an appropriate baseline/fit
    period when used in a causal monitoring pipeline.
    """
    values = np.asarray(innovations, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]

    if values.size < 3:
        return 0.0

    x = values[:-1]
    y = values[1:]

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    denominator = np.sqrt(
        np.sum(x_centered ** 2)
        * np.sum(y_centered ** 2)
    )

    if denominator <= 0:
        return 0.0

    rho = float(
        np.sum(x_centered * y_centered)
        / denominator
    )

    return float(np.clip(rho, clip_min, clip_max))


__all__ = [
    "CUSUMDirection",
    "CUSUMState",
    "CUSUMConfig",
    "CUSUMResult",
    "CUSUMSummary",
    "CUSUM",
    "compute_cusum",
    "estimate_ar1_correlation",
]
