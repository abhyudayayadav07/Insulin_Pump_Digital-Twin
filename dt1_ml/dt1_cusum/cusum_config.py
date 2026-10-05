"""
Configuration helpers for the DT1 GECO/CUSUM monitoring layer.

This module keeps CUSUM monitoring parameters separate from the CUSUM
algorithm itself.

The configuration is intentionally explicit so that experiments can
record exactly which monitoring assumptions were used.

Important:
    These are research/simulation defaults. They are not clinically
    validated safety thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Mapping, Optional


class ResidualNormalization(str, Enum):
    """How GECO residuals are converted to monitoring units."""

    STANDARD_DEVIATION = "standard_deviation"
    FITTED_SCALE = "fitted_scale"
    NONE = "none"


class VarianceMode(str, Enum):
    """Source of observation/innovation variance."""

    FIXED = "fixed"
    PROVIDED = "provided"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class CUSUMMonitoringConfig:
    """
    Main CUSUM monitoring configuration.

    The defaults are conservative engineering defaults for a research
    implementation and should be calibrated/validated on the intended
    baseline data.

    Attributes
    ----------
    reference:
        CUSUM reference value k in normalized innovation units.

    threshold:
        CUSUM alarm threshold h.

    warning_fraction:
        Warning threshold as a fraction of h.

    whitening_rho:
        AR(1) coefficient used for innovation whitening.

    nominal_variance:
        Nominal residual/measurement variance. 81 corresponds to a
        standard deviation of 9 mg/dL.

    normalization:
        Residual normalization strategy.

    fitted_scale:
        Scale used when ``normalization`` is FITTED_SCALE.

    min_variance:
        Numerical lower bound on variance.

    reset_on_alarm:
        Reset accumulated CUSUM statistics after an alarm.

    reset_after_gap:
        Clear previous innovation after a data gap/reset.

    max_abs_innovation:
        Optional clipping limit for numerical robustness.
    """

    reference: float = 0.5
    threshold: float = 5.0
    warning_fraction: float = 0.6

    whitening_rho: float = 0.0

    nominal_variance: float = 81.0
    variance_mode: VarianceMode = VarianceMode.FIXED

    normalization: ResidualNormalization = (
        ResidualNormalization.STANDARD_DEVIATION
    )
    fitted_scale: Optional[float] = None

    min_variance: float = 1e-6
    max_abs_innovation: Optional[float] = None

    reset_on_alarm: bool = False
    reset_after_gap: bool = True

    sample_interval_minutes: float = 5.0

    def __post_init__(self) -> None:
        if self.reference < 0:
            raise ValueError("reference must be >= 0.")

        if self.threshold <= 0:
            raise ValueError("threshold must be > 0.")

        if not 0 < self.warning_fraction < 1:
            raise ValueError(
                "warning_fraction must be between 0 and 1."
            )

        if not -0.98 <= self.whitening_rho <= 0.98:
            raise ValueError(
                "whitening_rho must be in [-0.98, 0.98]."
            )

        if self.nominal_variance <= 0:
            raise ValueError(
                "nominal_variance must be > 0."
            )

        if self.fitted_scale is not None and self.fitted_scale <= 0:
            raise ValueError(
                "fitted_scale must be > 0 when provided."
            )

        if self.min_variance <= 0:
            raise ValueError(
                "min_variance must be > 0."
            )

        if (
            self.max_abs_innovation is not None
            and self.max_abs_innovation <= 0
        ):
            raise ValueError(
                "max_abs_innovation must be > 0 when provided."
            )

        if self.sample_interval_minutes <= 0:
            raise ValueError(
                "sample_interval_minutes must be > 0."
            )

    @property
    def warning_threshold(self) -> float:
        """Absolute CUSUM value at which warning evidence begins."""
        return self.threshold * self.warning_fraction

    @property
    def nominal_standard_deviation(self) -> float:
        """Nominal standard deviation implied by nominal_variance."""
        return self.nominal_variance ** 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a plain dictionary."""
        return {
            "reference": self.reference,
            "threshold": self.threshold,
            "warning_fraction": self.warning_fraction,
            "warning_threshold": self.warning_threshold,
            "whitening_rho": self.whitening_rho,
            "nominal_variance": self.nominal_variance,
            "nominal_standard_deviation": self.nominal_standard_deviation,
            "variance_mode": self.variance_mode.value,
            "normalization": self.normalization.value,
            "fitted_scale": self.fitted_scale,
            "min_variance": self.min_variance,
            "max_abs_innovation": self.max_abs_innovation,
            "reset_on_alarm": self.reset_on_alarm,
            "reset_after_gap": self.reset_after_gap,
            "sample_interval_minutes": self.sample_interval_minutes,
        }

    def with_updates(
        self,
        **updates: Any,
    ) -> "CUSUMMonitoringConfig":
        """Return a validated copy with selected fields changed."""
        return replace(self, **updates)


@dataclass(frozen=True)
class GECOCUSUMPreset:
    """
    Named GECO/CUSUM monitoring preset.

    A preset documents the intended experimental regime without mixing
    the preset definition into the CUSUM implementation.
    """

    name: str
    description: str
    config: CUSUMMonitoringConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config.to_dict(),
        }


def default_cusum_config() -> CUSUMMonitoringConfig:
    """Return the default research configuration."""
    return CUSUMMonitoringConfig()


def geco_baseline_config() -> CUSUMMonitoringConfig:
    """
    Return a baseline configuration for GECO residual monitoring.

    This is intentionally close to the project's initial CUSUM defaults:
    nominal variance 81 and no whitening until a baseline rho estimate
    is available.
    """
    return CUSUMMonitoringConfig(
        reference=0.5,
        threshold=5.0,
        warning_fraction=0.6,
        whitening_rho=0.0,
        nominal_variance=81.0,
        variance_mode=VarianceMode.FIXED,
        normalization=ResidualNormalization.STANDARD_DEVIATION,
        fitted_scale=None,
        reset_on_alarm=False,
        reset_after_gap=True,
        sample_interval_minutes=5.0,
    )


def geco_fitted_scale_config(
    fitted_scale: float,
    whitening_rho: float = 0.0,
) -> CUSUMMonitoringConfig:
    """
    Create a configuration using a GECO-fit residual scale.

    ``fitted_scale`` should be estimated causally from the appropriate
    model-fit/baseline period.
    """
    if fitted_scale <= 0:
        raise ValueError("fitted_scale must be > 0.")

    return CUSUMMonitoringConfig(
        reference=0.5,
        threshold=5.0,
        warning_fraction=0.6,
        whitening_rho=whitening_rho,
        nominal_variance=81.0,
        variance_mode=VarianceMode.FIXED,
        normalization=ResidualNormalization.FITTED_SCALE,
        fitted_scale=fitted_scale,
        reset_on_alarm=False,
        reset_after_gap=True,
        sample_interval_minutes=5.0,
    )


def geco_whitened_config(
    whitening_rho: float,
    threshold: float = 5.0,
) -> CUSUMMonitoringConfig:
    """
    Create a configuration with AR(1) innovation whitening enabled.

    The rho estimate should come from an appropriate baseline/fit
    innovation sequence and should be clipped to a stable range.
    """
    if not -0.98 <= whitening_rho <= 0.98:
        raise ValueError(
            "whitening_rho must be in [-0.98, 0.98]."
        )

    return CUSUMMonitoringConfig(
        reference=0.5,
        threshold=threshold,
        warning_fraction=0.6,
        whitening_rho=whitening_rho,
        nominal_variance=81.0,
        variance_mode=VarianceMode.FIXED,
        normalization=ResidualNormalization.STANDARD_DEVIATION,
        reset_on_alarm=False,
        reset_after_gap=True,
        sample_interval_minutes=5.0,
    )


def available_presets() -> Dict[str, GECOCUSUMPreset]:
    """Return named presets available to experiments."""
    baseline = geco_baseline_config()

    return {
        "baseline": GECOCUSUMPreset(
            name="baseline",
            description=(
                "Baseline GECO residual monitoring with fixed nominal "
                "variance and no AR(1) whitening."
            ),
            config=baseline,
        ),
    }


def config_from_mapping(
    values: Mapping[str, Any],
) -> CUSUMMonitoringConfig:
    """
    Build a configuration from a dictionary/YAML-like mapping.

    Enum-valued strings are converted automatically.
    """
    data = dict(values)

    if "variance_mode" in data:
        data["variance_mode"] = VarianceMode(
            data["variance_mode"]
        )

    if "normalization" in data:
        data["normalization"] = ResidualNormalization(
            data["normalization"]
        )

    return CUSUMMonitoringConfig(**data)


def validate_cusum_config(
    config: CUSUMMonitoringConfig,
) -> Dict[str, Any]:
    """
    Validate and summarize a CUSUM configuration.

    Dataclass construction already validates field ranges; this helper
    provides a structured validation result useful to experiment setup.
    """
    return {
        "valid": True,
        "warning_threshold": config.warning_threshold,
        "nominal_standard_deviation": (
            config.nominal_standard_deviation
        ),
        "config": config.to_dict(),
    }


def apply_to_cusum_config(
    config: CUSUMMonitoringConfig,
) -> Dict[str, Any]:
    """
    Convert this module's configuration to arguments expected by
    ``dt1_cusum.cusum.CUSUMConfig``.

    This keeps the algorithm module independent from higher-level
    configuration management.
    """
    normalization_scale = None

    if config.normalization == ResidualNormalization.FITTED_SCALE:
        normalization_scale = config.fitted_scale

    return {
        "reference": config.reference,
        "threshold": config.threshold,
        "warning_fraction": config.warning_fraction,
        "whitening_rho": config.whitening_rho,
        "nominal_variance": config.nominal_variance,
        "normalization_scale": normalization_scale,
        "reset_on_alarm": config.reset_on_alarm,
        "reset_after_gap": config.reset_after_gap,
        "min_variance": config.min_variance,
        "max_abs_innovation": config.max_abs_innovation,
    }


__all__ = [
    "ResidualNormalization",
    "VarianceMode",
    "CUSUMMonitoringConfig",
    "GECOCUSUMPreset",
    "default_cusum_config",
    "geco_baseline_config",
    "geco_fitted_scale_config",
    "geco_whitened_config",
    "available_presets",
    "config_from_mapping",
    "validate_cusum_config",
    "apply_to_cusum_config",
]
