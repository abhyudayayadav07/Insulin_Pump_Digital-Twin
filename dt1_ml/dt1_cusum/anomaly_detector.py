"""
Higher-level CUSUM anomaly detector for the DT1 GECO pipeline.

This module sits above the low-level CUSUM implementation.

Pipeline:

    GECO prediction
          |
          v
       residual
          |
          v
      CUSUM monitor
          |
          v
    CUSUM statistics
          |
          v
    AnomalyDetector
          |
          +--> NORMAL
          +--> WARNING
          +--> ANOMALY
          |
          v
    structured evidence for
    security / hazard / decision layers

The detector does not directly control the insulin pump and does not
make a clinical decision. It produces structured anomaly evidence that
can be consumed by the downstream decision engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

try:
    from .cusum import (
        CUSUM,
        CUSUMConfig,
        CUSUMResult,
        CUSUMState,
    )
except ImportError:
    from cusum import (
        CUSUM,
        CUSUMConfig,
        CUSUMResult,
        CUSUMState,
    )

try:
    from .cusum_config import (
        CUSUMMonitoringConfig,
        apply_to_cusum_config,
        default_cusum_config,
    )
except ImportError:
    from cusum_config import (
        CUSUMMonitoringConfig,
        apply_to_cusum_config,
        default_cusum_config,
    )


class AnomalyLevel(str, Enum):
    """High-level anomaly evidence state."""

    NORMAL = "normal"
    WARNING = "warning"
    ANOMALY = "anomaly"
    CRITICAL = "critical"


class AnomalyDirection(str, Enum):
    """Direction of persistent model-observation disagreement."""

    NONE = "none"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    BOTH = "both"


@dataclass(frozen=True)
class AnomalyDetectorConfig:
    """
    Configuration for the higher-level anomaly detector.

    Parameters
    ----------
    warning_score:
        Minimum normalized CUSUM statistic for warning evidence.

    anomaly_score:
        Minimum normalized CUSUM statistic for anomaly evidence.

    critical_score:
        Minimum normalized CUSUM statistic for critical evidence.

    persistence_samples:
        Number of consecutive warning/anomaly samples required before
        persistence evidence is emitted.

    use_direction:
        Whether the detector reports the direction of the CUSUM signal.

    confidence_scale:
        Scale controlling the conversion of CUSUM magnitude into a
        bounded evidence confidence.

    reset_persistence_on_normal:
        Reset consecutive persistence count after a normal sample.
    """

    warning_score: float = 0.60
    anomaly_score: float = 1.00
    critical_score: float = 1.50

    persistence_samples: int = 2

    use_direction: bool = True

    confidence_scale: float = 1.0

    reset_persistence_on_normal: bool = True

    def __post_init__(self) -> None:
        if self.warning_score < 0:
            raise ValueError("warning_score must be >= 0.")

        if self.anomaly_score <= self.warning_score:
            raise ValueError(
                "anomaly_score must be greater than warning_score."
            )

        if self.critical_score <= self.anomaly_score:
            raise ValueError(
                "critical_score must be greater than anomaly_score."
            )

        if self.persistence_samples <= 0:
            raise ValueError("persistence_samples must be > 0.")

        if self.confidence_scale <= 0:
            raise ValueError("confidence_scale must be > 0.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "warning_score": self.warning_score,
            "anomaly_score": self.anomaly_score,
            "critical_score": self.critical_score,
            "persistence_samples": self.persistence_samples,
            "use_direction": self.use_direction,
            "confidence_scale": self.confidence_scale,
            "reset_persistence_on_normal": (
                self.reset_persistence_on_normal
            ),
        }


@dataclass(frozen=True)
class AnomalyEvidence:
    """Structured anomaly evidence generated from one residual sample."""

    step: int
    residual: float
    normalized_residual: float
    whitened_innovation: float

    cusum_statistic: float
    positive_cusum: float
    negative_cusum: float

    level: AnomalyLevel
    direction: AnomalyDirection

    confidence: float
    persistent: bool

    warning: bool
    anomaly: bool
    critical: bool

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "residual": self.residual,
            "normalized_residual": self.normalized_residual,
            "whitened_innovation": self.whitened_innovation,
            "cusum_statistic": self.cusum_statistic,
            "positive_cusum": self.positive_cusum,
            "negative_cusum": self.negative_cusum,
            "level": self.level.value,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "persistent": self.persistent,
            "warning": self.warning,
            "anomaly": self.anomaly,
            "critical": self.critical,
            "metadata": dict(self.metadata),
        }


@dataclass
class AnomalyDetectionSummary:
    """Summary of anomaly detection over a residual sequence."""

    evidence: List[AnomalyEvidence]

    total_samples: int
    warning_count: int
    anomaly_count: int
    critical_count: int

    first_anomaly_step: Optional[int]
    first_critical_step: Optional[int]

    maximum_confidence: float
    maximum_cusum: float

    persistent_anomaly_detected: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "warning_count": self.warning_count,
            "anomaly_count": self.anomaly_count,
            "critical_count": self.critical_count,
            "first_anomaly_step": self.first_anomaly_step,
            "first_critical_step": self.first_critical_step,
            "maximum_confidence": self.maximum_confidence,
            "maximum_cusum": self.maximum_cusum,
            "persistent_anomaly_detected": (
                self.persistent_anomaly_detected
            ),
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ],
        }


class CUSUMAnomalyDetector:
    """
    Convert CUSUM outputs into structured anomaly evidence.

    The detector uses the CUSUM threshold as the primary alarm boundary
    and adds higher-level persistence/severity categories.

    This separation is deliberate:

        CUSUM
            = sequential statistical monitor

        AnomalyDetector
            = evidence interpretation

        DecisionEngine
            = final multi-source safety/security decision
    """

    def __init__(
        self,
        cusum: Optional[CUSUM] = None,
        cusum_config: Optional[CUSUMMonitoringConfig] = None,
        config: Optional[AnomalyDetectorConfig] = None,
    ) -> None:
        if cusum is not None and cusum_config is not None:
            raise ValueError(
                "Provide either cusum or cusum_config, not both."
            )

        if cusum is not None:
            self.cusum = cusum
            self.monitoring_config = None
        else:
            monitoring_config = (
                cusum_config
                if cusum_config is not None
                else default_cusum_config()
            )

            low_level_config = CUSUMConfig(
                **apply_to_cusum_config(monitoring_config)
            )

            self.cusum = CUSUM(low_level_config)
            self.monitoring_config = monitoring_config

        self.config = config or AnomalyDetectorConfig()

        self._consecutive_warning = 0
        self._consecutive_anomaly = 0
        self._last_direction = AnomalyDirection.NONE

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset CUSUM and higher-level persistence state."""
        self.cusum.reset()
        self._consecutive_warning = 0
        self._consecutive_anomaly = 0
        self._last_direction = AnomalyDirection.NONE

    @property
    def consecutive_anomaly_count(self) -> int:
        return self._consecutive_anomaly

    @property
    def consecutive_warning_count(self) -> int:
        return self._consecutive_warning

    # ------------------------------------------------------------------
    # Evidence interpretation
    # ------------------------------------------------------------------

    def _direction_from_cusum(
        self,
        result: CUSUMResult,
    ) -> AnomalyDirection:
        if not self.config.use_direction:
            return AnomalyDirection.NONE

        positive = result.positive_statistic
        negative = result.negative_statistic

        if positive <= 0 and negative <= 0:
            return AnomalyDirection.NONE

        if positive > 0 and negative <= 0:
            return AnomalyDirection.POSITIVE

        if negative > 0 and positive <= 0:
            return AnomalyDirection.NEGATIVE

        if positive > negative:
            return AnomalyDirection.POSITIVE

        if negative > positive:
            return AnomalyDirection.NEGATIVE

        return AnomalyDirection.BOTH

    def _confidence(
        self,
        statistic: float,
    ) -> float:
        """
        Convert CUSUM magnitude to bounded evidence confidence.

        This is a monotonic evidence mapping, not a calibrated
        probability.
        """
        score = max(0.0, float(statistic))
        scale = self.config.confidence_scale

        confidence = 1.0 - np.exp(-score / scale)

        return float(np.clip(confidence, 0.0, 1.0))

    def _level(
        self,
        statistic: float,
        persistent: bool,
    ) -> AnomalyLevel:
        if statistic >= self.config.critical_score:
            return AnomalyLevel.CRITICAL

        if statistic >= self.config.anomaly_score:
            return AnomalyLevel.ANOMALY

        if statistic >= self.config.warning_score:
            return AnomalyLevel.WARNING

        # Persistence is useful evidence, but does not independently
        # elevate a normal sample to an anomaly.
        if persistent:
            return AnomalyLevel.WARNING

        return AnomalyLevel.NORMAL

    def _update_persistence(
        self,
        result: CUSUMResult,
    ) -> bool:
        """
        Update persistence counters.

        Persistence is based on the low-level CUSUM warning/alarm state,
        not merely on a single large residual.
        """
        if result.alarm:
            self._consecutive_anomaly += 1
            self._consecutive_warning += 1
        elif result.warning:
            self._consecutive_warning += 1
            self._consecutive_anomaly = 0
        else:
            if self.config.reset_persistence_on_normal:
                self._consecutive_warning = 0
                self._consecutive_anomaly = 0

        persistent = (
            self._consecutive_anomaly
            >= self.config.persistence_samples
        )

        return persistent

    def _from_cusum_result(
        self,
        result: CUSUMResult,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AnomalyEvidence:
        persistent = self._update_persistence(result)

        direction = self._direction_from_cusum(result)

        level = self._level(
            statistic=result.statistic,
            persistent=persistent,
        )

        confidence = self._confidence(result.statistic)

        evidence_metadata = dict(result.metadata)

        if metadata:
            evidence_metadata.update(dict(metadata))

        evidence_metadata.update(
            {
                "cusum_state": result.state.value,
                "consecutive_warning": (
                    self._consecutive_warning
                ),
                "consecutive_anomaly": (
                    self._consecutive_anomaly
                ),
            }
        )

        return AnomalyEvidence(
            step=result.step,
            residual=result.raw_residual,
            normalized_residual=result.normalized_residual,
            whitened_innovation=result.whitened_innovation,
            cusum_statistic=result.statistic,
            positive_cusum=result.positive_statistic,
            negative_cusum=result.negative_statistic,
            level=level,
            direction=direction,
            confidence=confidence,
            persistent=persistent,
            warning=(
                level in (
                    AnomalyLevel.WARNING,
                    AnomalyLevel.ANOMALY,
                    AnomalyLevel.CRITICAL,
                )
            ),
            anomaly=(
                level in (
                    AnomalyLevel.ANOMALY,
                    AnomalyLevel.CRITICAL,
                )
            ),
            critical=(level == AnomalyLevel.CRITICAL),
            metadata=evidence_metadata,
        )

    # ------------------------------------------------------------------
    # Public update methods
    # ------------------------------------------------------------------

    def update(
        self,
        residual: float,
        variance: Optional[float] = None,
        reset: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AnomalyEvidence:
        """Process one GECO residual and generate anomaly evidence."""
        if reset:
            self._consecutive_warning = 0
            self._consecutive_anomaly = 0

        result = self.cusum.update(
            residual=residual,
            variance=variance,
            reset=reset,
            metadata=dict(metadata or {}),
        )

        return self._from_cusum_result(
            result=result,
            metadata=metadata,
        )

    def update_many(
        self,
        residuals: Sequence[float],
        variances: Optional[Sequence[float]] = None,
        reset_indices: Optional[Iterable[int]] = None,
        metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> AnomalyDetectionSummary:
        """Process an entire residual sequence."""
        values = np.asarray(
            residuals,
            dtype=float,
        ).reshape(-1)

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

        reset_set = set(
            int(index)
            for index in (reset_indices or [])
        )

        if metadata is not None and len(metadata) != values.size:
            raise ValueError(
                "metadata must have the same length as residuals."
            )

        evidence: List[AnomalyEvidence] = []

        for index, residual in enumerate(values):
            variance = (
                None
                if variance_values is None
                else float(variance_values[index])
            )

            item_metadata = (
                None
                if metadata is None
                else dict(metadata[index])
            )

            evidence.append(
                self.update(
                    residual=float(residual),
                    variance=variance,
                    reset=(index in reset_set),
                    metadata=item_metadata,
                )
            )

        anomaly_steps = [
            item.step
            for item in evidence
            if item.anomaly
        ]

        critical_steps = [
            item.step
            for item in evidence
            if item.critical
        ]

        return AnomalyDetectionSummary(
            evidence=evidence,
            total_samples=len(evidence),
            warning_count=sum(item.warning for item in evidence),
            anomaly_count=sum(item.anomaly for item in evidence),
            critical_count=sum(item.critical for item in evidence),
            first_anomaly_step=(
                anomaly_steps[0]
                if anomaly_steps
                else None
            ),
            first_critical_step=(
                critical_steps[0]
                if critical_steps
                else None
            ),
            maximum_confidence=(
                float(max(item.confidence for item in evidence))
                if evidence
                else 0.0
            ),
            maximum_cusum=(
                float(max(item.cusum_statistic for item in evidence))
                if evidence
                else 0.0
            ),
            persistent_anomaly_detected=any(
                item.persistent
                for item in evidence
            ),
        )

    # ------------------------------------------------------------------
    # GECO residual integration
    # ------------------------------------------------------------------

    def detect_from_residual_result(
        self,
        residual_result: Any,
        variance: Optional[Sequence[float]] = None,
        reset_indices: Optional[Iterable[int]] = None,
    ) -> AnomalyDetectionSummary:
        """
        Run detection directly on a GECORResidual-like object.

        The object is expected to expose ``residuals``.
        """
        if not hasattr(residual_result, "residuals"):
            raise TypeError(
                "residual_result must expose a 'residuals' attribute."
            )

        residuals = np.asarray(
            residual_result.residuals,
            dtype=float,
        )

        return self.update_many(
            residuals=residuals,
            variances=variance,
            reset_indices=reset_indices,
        )

    # ------------------------------------------------------------------
    # Convenience status methods
    # ------------------------------------------------------------------

    def is_anomaly(self) -> bool:
        """Return whether the current CUSUM state is anomalous."""
        return self.cusum.is_alarm()

    def current_confidence(self) -> float:
        """Return bounded current anomaly evidence confidence."""
        return self._confidence(self.cusum.statistic)

    def current_level(self) -> AnomalyLevel:
        """Return the current high-level anomaly state."""
        persistent = (
            self._consecutive_anomaly
            >= self.config.persistence_samples
        )

        return self._level(
            statistic=self.cusum.statistic,
            persistent=persistent,
        )


# ----------------------------------------------------------------------
# Functional helper
# ----------------------------------------------------------------------

def detect_anomalies(
    residuals: Sequence[float],
    cusum_config: Optional[CUSUMMonitoringConfig] = None,
    detector_config: Optional[AnomalyDetectorConfig] = None,
    variances: Optional[Sequence[float]] = None,
    reset_indices: Optional[Iterable[int]] = None,
) -> AnomalyDetectionSummary:
    """Functional wrapper for GECO residual anomaly detection."""
    detector = CUSUMAnomalyDetector(
        cusum_config=cusum_config,
        config=detector_config,
    )

    return detector.update_many(
        residuals=residuals,
        variances=variances,
        reset_indices=reset_indices,
    )


__all__ = [
    "AnomalyLevel",
    "AnomalyDirection",
    "AnomalyDetectorConfig",
    "AnomalyEvidence",
    "AnomalyDetectionSummary",
    "CUSUMAnomalyDetector",
    "detect_anomalies",
]
