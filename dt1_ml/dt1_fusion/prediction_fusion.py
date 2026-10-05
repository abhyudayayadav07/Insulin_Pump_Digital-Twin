"""
Prediction fusion for the DT1 hybrid predictive digital twin.

This module combines predictions from:

    1. ML/DL predictive model
    2. GECO physiological model

The fusion layer produces a unified predictive trajectory while
preserving model disagreement as explicit evidence.

Conceptual path:

    ML/DL prediction ─────┐
                          │
                          ├──> prediction fusion ──> DT1 prediction
                          │
    GECO prediction ──────┘             │
                                        └──> disagreement metadata

Important:
    Prediction fusion is not the final safety decision. The resulting
    prediction should remain subject to the DT1 residual/CUSUM layer,
    DT2 reactive-safe twin, hazard/risk analysis, and decision engine.

The default fusion is confidence-weighted interpolation:

    fused = w_ml * ML + w_geco * GECO

Weights are normalized to sum to one.

When disagreement becomes large, the configuration can reduce the
influence of the ML model and increase the GECO contribution. This is
an engineering mechanism for model-consistency handling, not a claim
that GECO is clinically superior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


class FusionMode(str, Enum):
    """Strategy used to combine ML/DL and GECO predictions."""

    FIXED_WEIGHT = "fixed_weight"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    DISAGREEMENT_ADAPTIVE = "disagreement_adaptive"


class FusionStatus(str, Enum):
    """Status of a fused prediction point."""

    NORMAL = "normal"
    DISAGREEMENT = "disagreement"
    HIGH_DISAGREEMENT = "high_disagreement"
    CRITICAL_DISAGREEMENT = "critical_disagreement"


@dataclass(frozen=True)
class PredictionFusionConfig:
    """
    Configuration for ML/GECO prediction fusion.

    Parameters
    ----------
    ml_weight:
        Base weight assigned to the ML/DL prediction.

    geco_weight:
        Base weight assigned to the GECO prediction.

    mode:
        Fusion strategy.

    ml_confidence:
        Baseline ML confidence used by confidence-weighted fusion.

    geco_confidence:
        Baseline GECO confidence used by confidence-weighted fusion.

    low_disagreement:
        Absolute disagreement at which adaptive fusion begins.

    high_disagreement:
        Absolute disagreement at which adaptive fusion strongly reduces
        ML influence.

    critical_disagreement:
        Absolute disagreement at which ML influence reaches the configured
        minimum.

    minimum_ml_weight:
        Lower bound on ML contribution during adaptive fusion.

    maximum_geco_weight:
        Upper bound on GECO contribution during adaptive fusion.

    clamp_min / clamp_max:
        Optional bounds on fused glucose.

    disagreement_confidence_scale:
        Scale for converting disagreement magnitude to bounded evidence.
    """

    ml_weight: float = 0.5
    geco_weight: float = 0.5

    mode: FusionMode = FusionMode.DISAGREEMENT_ADAPTIVE

    ml_confidence: float = 1.0
    geco_confidence: float = 1.0

    low_disagreement: float = 10.0
    high_disagreement: float = 40.0
    critical_disagreement: float = 70.0

    minimum_ml_weight: float = 0.10
    maximum_geco_weight: float = 0.90

    clamp_min: float = 20.0
    clamp_max: float = 600.0

    disagreement_confidence_scale: float = 20.0

    def __post_init__(self) -> None:
        if self.ml_weight < 0 or self.geco_weight < 0:
            raise ValueError("Model weights must be >= 0.")

        if self.ml_weight + self.geco_weight <= 0:
            raise ValueError(
                "At least one model weight must be > 0."
            )

        if not 0 <= self.ml_confidence <= 1:
            raise ValueError(
                "ml_confidence must be between 0 and 1."
            )

        if not 0 <= self.geco_confidence <= 1:
            raise ValueError(
                "geco_confidence must be between 0 and 1."
            )

        if not (
            0 <= self.low_disagreement
            <= self.high_disagreement
            <= self.critical_disagreement
        ):
            raise ValueError(
                "Disagreement thresholds must be monotonically increasing."
            )

        if not 0 <= self.minimum_ml_weight <= 1:
            raise ValueError(
                "minimum_ml_weight must be between 0 and 1."
            )

        if not 0 <= self.maximum_geco_weight <= 1:
            raise ValueError(
                "maximum_geco_weight must be between 0 and 1."
            )

        if self.clamp_min >= self.clamp_max:
            raise ValueError(
                "clamp_min must be smaller than clamp_max."
            )

        if self.disagreement_confidence_scale <= 0:
            raise ValueError(
                "disagreement_confidence_scale must be > 0."
            )

    def normalized_base_weights(self) -> tuple[float, float]:
        """Return base ML and GECO weights normalized to sum to one."""
        total = self.ml_weight + self.geco_weight

        return (
            self.ml_weight / total,
            self.geco_weight / total,
        )

    def to_dict(self) -> Dict[str, Any]:
        ml_base, geco_base = self.normalized_base_weights()

        return {
            "ml_weight": self.ml_weight,
            "geco_weight": self.geco_weight,
            "normalized_ml_weight": ml_base,
            "normalized_geco_weight": geco_base,
            "mode": self.mode.value,
            "ml_confidence": self.ml_confidence,
            "geco_confidence": self.geco_confidence,
            "low_disagreement": self.low_disagreement,
            "high_disagreement": self.high_disagreement,
            "critical_disagreement": self.critical_disagreement,
            "minimum_ml_weight": self.minimum_ml_weight,
            "maximum_geco_weight": self.maximum_geco_weight,
            "clamp_min": self.clamp_min,
            "clamp_max": self.clamp_max,
            "disagreement_confidence_scale": (
                self.disagreement_confidence_scale
            ),
        }


@dataclass(frozen=True)
class PredictionFusionPoint:
    """One fused prediction point."""

    step: int
    time_minutes: float

    ml_prediction: float
    geco_prediction: float
    fused_prediction: float

    disagreement: float
    absolute_disagreement: float

    ml_weight: float
    geco_weight: float

    disagreement_confidence: float
    status: FusionStatus

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "time_minutes": self.time_minutes,
            "ml_prediction": self.ml_prediction,
            "geco_prediction": self.geco_prediction,
            "fused_prediction": self.fused_prediction,
            "disagreement": self.disagreement,
            "absolute_disagreement": self.absolute_disagreement,
            "ml_weight": self.ml_weight,
            "geco_weight": self.geco_weight,
            "disagreement_confidence": (
                self.disagreement_confidence
            ),
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }


@dataclass
class PredictionFusionResult:
    """Complete fused DT1 prediction result."""

    points: List[PredictionFusionPoint]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def fused_predictions(self) -> np.ndarray:
        return np.asarray(
            [
                point.fused_prediction
                for point in self.points
            ],
            dtype=float,
        )

    @property
    def ml_predictions(self) -> np.ndarray:
        return np.asarray(
            [
                point.ml_prediction
                for point in self.points
            ],
            dtype=float,
        )

    @property
    def geco_predictions(self) -> np.ndarray:
        return np.asarray(
            [
                point.geco_prediction
                for point in self.points
            ],
            dtype=float,
        )

    @property
    def disagreements(self) -> np.ndarray:
        return np.asarray(
            [
                point.disagreement
                for point in self.points
            ],
            dtype=float,
        )

    @property
    def ml_weights(self) -> np.ndarray:
        return np.asarray(
            [
                point.ml_weight
                for point in self.points
            ],
            dtype=float,
        )

    @property
    def geco_weights(self) -> np.ndarray:
        return np.asarray(
            [
                point.geco_weight
                for point in self.points
            ],
            dtype=float,
        )

    def mean_absolute_disagreement(self) -> float:
        values = np.abs(self.disagreements)

        return (
            float(np.mean(values))
            if values.size
            else float("nan")
        )

    def maximum_absolute_disagreement(self) -> float:
        values = np.abs(self.disagreements)

        return (
            float(np.max(values))
            if values.size
            else float("nan")
        )

    def high_disagreement_count(self) -> int:
        return sum(
            point.status
            in (
                FusionStatus.HIGH_DISAGREEMENT,
                FusionStatus.CRITICAL_DISAGREEMENT,
            )
            for point in self.points
        )

    def critical_disagreement_count(self) -> int:
        return sum(
            point.status == FusionStatus.CRITICAL_DISAGREEMENT
            for point in self.points
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": [
                point.to_dict()
                for point in self.points
            ],
            "metadata": dict(self.metadata),
            "summary": {
                "mean_absolute_disagreement": (
                    self.mean_absolute_disagreement()
                ),
                "maximum_absolute_disagreement": (
                    self.maximum_absolute_disagreement()
                ),
                "high_disagreement_count": (
                    self.high_disagreement_count()
                ),
                "critical_disagreement_count": (
                    self.critical_disagreement_count()
                ),
            },
        }


class PredictionFusion:
    """
    Fuse ML/DL and GECO glucose predictions.

    The fusion layer deliberately keeps the individual predictions and
    their disagreement visible. It does not overwrite model outputs.
    """

    def __init__(
        self,
        config: Optional[PredictionFusionConfig] = None,
    ) -> None:
        self.config = config or PredictionFusionConfig()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _status(
        self,
        absolute_disagreement: float,
    ) -> FusionStatus:
        if (
            absolute_disagreement
            >= self.config.critical_disagreement
        ):
            return FusionStatus.CRITICAL_DISAGREEMENT

        if (
            absolute_disagreement
            >= self.config.high_disagreement
        ):
            return FusionStatus.HIGH_DISAGREEMENT

        if (
            absolute_disagreement
            >= self.config.low_disagreement
        ):
            return FusionStatus.DISAGREEMENT

        return FusionStatus.NORMAL

    def _disagreement_confidence(
        self,
        absolute_disagreement: float,
    ) -> float:
        """
        Convert disagreement magnitude into bounded evidence confidence.

        This is not a probability.
        """
        confidence = 1.0 - np.exp(
            -max(0.0, absolute_disagreement)
            / self.config.disagreement_confidence_scale
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def _fixed_weights(self) -> tuple[float, float]:
        return self.config.normalized_base_weights()

    def _confidence_weights(self) -> tuple[float, float]:
        ml_score = (
            self.config.ml_weight
            * self.config.ml_confidence
        )

        geco_score = (
            self.config.geco_weight
            * self.config.geco_confidence
        )

        total = ml_score + geco_score

        if total <= 0:
            return self._fixed_weights()

        return (
            ml_score / total,
            geco_score / total,
        )

    def _adaptive_weights(
        self,
        absolute_disagreement: float,
    ) -> tuple[float, float]:
        """
        Reduce ML contribution as model disagreement increases.

        The adaptive rule is deliberately monotonic and transparent.
        It is not a learned policy.
        """
        base_ml, base_geco = self._confidence_weights()

        low = self.config.low_disagreement
        high = self.config.critical_disagreement

        if absolute_disagreement <= low:
            ml_weight = base_ml
        elif absolute_disagreement >= high:
            ml_weight = self.config.minimum_ml_weight
        else:
            fraction = (
                absolute_disagreement - low
            ) / (high - low)

            ml_weight = (
                base_ml
                + fraction
                * (
                    self.config.minimum_ml_weight
                    - base_ml
                )
            )

            ml_weight = max(
                self.config.minimum_ml_weight,
                ml_weight,
            )

        geco_weight = 1.0 - ml_weight

        if geco_weight > self.config.maximum_geco_weight:
            geco_weight = self.config.maximum_geco_weight
            ml_weight = 1.0 - geco_weight

        ml_weight = float(np.clip(ml_weight, 0.0, 1.0))
        geco_weight = float(np.clip(geco_weight, 0.0, 1.0))

        total = ml_weight + geco_weight

        return (
            ml_weight / total,
            geco_weight / total,
        )

    def _weights(
        self,
        absolute_disagreement: float,
    ) -> tuple[float, float]:
        if self.config.mode == FusionMode.FIXED_WEIGHT:
            return self._fixed_weights()

        if self.config.mode == FusionMode.CONFIDENCE_WEIGHTED:
            return self._confidence_weights()

        return self._adaptive_weights(
            absolute_disagreement
        )

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------

    def fuse_point(
        self,
        ml_prediction: float,
        geco_prediction: float,
        step: int = 0,
        time_minutes: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PredictionFusionPoint:
        """Fuse one ML prediction with one GECO prediction."""
        ml = float(ml_prediction)
        geco = float(geco_prediction)

        if not np.isfinite(ml):
            raise ValueError("ml_prediction must be finite.")

        if not np.isfinite(geco):
            raise ValueError("geco_prediction must be finite.")

        disagreement = ml - geco
        absolute = abs(disagreement)

        ml_weight, geco_weight = self._weights(
            absolute_disagreement=absolute
        )

        fused = (
            ml_weight * ml
            + geco_weight * geco
        )

        fused = float(
            np.clip(
                fused,
                self.config.clamp_min,
                self.config.clamp_max,
            )
        )

        point_metadata = dict(metadata or {})

        point_metadata.update(
            {
                "fusion_mode": self.config.mode.value,
            }
        )

        return PredictionFusionPoint(
            step=int(step),
            time_minutes=float(time_minutes),
            ml_prediction=ml,
            geco_prediction=geco,
            fused_prediction=fused,
            disagreement=float(disagreement),
            absolute_disagreement=float(absolute),
            ml_weight=float(ml_weight),
            geco_weight=float(geco_weight),
            disagreement_confidence=(
                self._disagreement_confidence(absolute)
            ),
            status=self._status(absolute),
            metadata=point_metadata,
        )

    # ------------------------------------------------------------------
    # Sequence fusion
    # ------------------------------------------------------------------

    def fuse(
        self,
        ml_predictions: Sequence[float],
        geco_predictions: Sequence[float],
        time_minutes: Optional[Sequence[float]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PredictionFusionResult:
        """
        Fuse aligned ML and GECO prediction sequences.
        """
        ml = np.asarray(
            ml_predictions,
            dtype=float,
        ).reshape(-1)

        geco = np.asarray(
            geco_predictions,
            dtype=float,
        ).reshape(-1)

        if ml.size != geco.size:
            raise ValueError(
                "ml_predictions and geco_predictions must have "
                "the same length."
            )

        if time_minutes is None:
            times = np.arange(
                ml.size,
                dtype=float,
            )
        else:
            times = np.asarray(
                time_minutes,
                dtype=float,
            ).reshape(-1)

            if times.size != ml.size:
                raise ValueError(
                    "time_minutes must have the same length as predictions."
                )

        points: List[PredictionFusionPoint] = []

        for index, (ml_value, geco_value, time) in enumerate(
            zip(ml, geco, times)
        ):
            if not (
                np.isfinite(ml_value)
                and np.isfinite(geco_value)
                and np.isfinite(time)
            ):
                raise ValueError(
                    f"Non-finite prediction at index {index}."
                )

            points.append(
                self.fuse_point(
                    ml_prediction=float(ml_value),
                    geco_prediction=float(geco_value),
                    step=index,
                    time_minutes=float(time),
                    metadata=metadata,
                )
            )

        result_metadata = {
            "fusion_mode": self.config.mode.value,
            "n_points": int(len(points)),
            "base_ml_weight": (
                self.config.normalized_base_weights()[0]
            ),
            "base_geco_weight": (
                self.config.normalized_base_weights()[1]
            ),
        }

        if metadata:
            result_metadata.update(dict(metadata))

        return PredictionFusionResult(
            points=points,
            metadata=result_metadata,
        )

    # ------------------------------------------------------------------
    # Trajectory integration
    # ------------------------------------------------------------------

    def fuse_trajectories(
        self,
        ml_trajectory: Any,
        geco_trajectory: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PredictionFusionResult:
        """
        Fuse ML and GECO trajectory-like objects.

        Expected attributes:
            glucose
            time_minutes (optional)
        """
        if not hasattr(ml_trajectory, "glucose"):
            raise TypeError(
                "ml_trajectory must expose a 'glucose' attribute."
            )

        if not hasattr(geco_trajectory, "glucose"):
            raise TypeError(
                "geco_trajectory must expose a 'glucose' attribute."
            )

        ml_predictions = np.asarray(
            ml_trajectory.glucose,
            dtype=float,
        )

        geco_predictions = np.asarray(
            geco_trajectory.glucose,
            dtype=float,
        )

        if hasattr(ml_trajectory, "time_minutes"):
            times = np.asarray(
                ml_trajectory.time_minutes,
                dtype=float,
            )
        elif hasattr(geco_trajectory, "time_minutes"):
            times = np.asarray(
                geco_trajectory.time_minutes,
                dtype=float,
            )
        else:
            times = None

        return self.fuse(
            ml_predictions=ml_predictions,
            geco_predictions=geco_predictions,
            time_minutes=times,
            metadata=metadata,
        )


# ----------------------------------------------------------------------
# Functional helper
# ----------------------------------------------------------------------

def fuse_predictions(
    ml_predictions: Sequence[float],
    geco_predictions: Sequence[float],
    time_minutes: Optional[Sequence[float]] = None,
    config: Optional[PredictionFusionConfig] = None,
) -> PredictionFusionResult:
    """Functional wrapper for prediction fusion."""
    fusion = PredictionFusion(config=config)

    return fusion.fuse(
        ml_predictions=ml_predictions,
        geco_predictions=geco_predictions,
        time_minutes=time_minutes,
    )


__all__ = [
    "FusionMode",
    "FusionStatus",
    "PredictionFusionConfig",
    "PredictionFusionPoint",
    "PredictionFusionResult",
    "PredictionFusion",
    "fuse_predictions",
]
