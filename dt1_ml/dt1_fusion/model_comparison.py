"""
ML/DL versus GECO model comparison for DT1 fusion.

This module quantifies disagreement between the predictive ML/DL model
and the GECO physiological model.

Conceptual path:

    ML/DL prediction ─────┐
                          ├──> model comparison ──> DT1 evidence
    GECO prediction ──────┘

The comparison produces descriptive disagreement evidence. It does not
declare either model correct/incorrect and does not make the final
safety decision.

Residual convention:
    disagreement = ML prediction - GECO prediction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


class DisagreementLevel(str, Enum):
    """Qualitative level of ML/GECO disagreement."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class DisagreementDirection(str, Enum):
    """Direction of ML prediction relative to GECO prediction."""

    NONE = "none"
    ML_HIGHER = "ml_higher"
    ML_LOWER = "ml_lower"
    EQUAL = "equal"


@dataclass(frozen=True)
class ModelComparisonConfig:
    """
    Thresholds and options for ML/GECO comparison.

    Absolute thresholds are expressed in glucose units, normally mg/dL.
    Relative disagreement is dimensionless.

    These thresholds are research defaults and require validation for the
    intended dataset and application.
    """

    low_absolute_threshold: float = 10.0
    moderate_absolute_threshold: float = 20.0
    high_absolute_threshold: float = 40.0
    critical_absolute_threshold: float = 70.0

    low_relative_threshold: float = 0.05
    moderate_relative_threshold: float = 0.10
    high_relative_threshold: float = 0.20
    critical_relative_threshold: float = 0.35

    relative_denominator_floor: float = 20.0

    confidence_scale: float = 20.0

    direction_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        absolute = (
            self.low_absolute_threshold,
            self.moderate_absolute_threshold,
            self.high_absolute_threshold,
            self.critical_absolute_threshold,
        )

        if any(value < 0 for value in absolute):
            raise ValueError(
                "Absolute disagreement thresholds must be >= 0."
            )

        if not (
            self.low_absolute_threshold
            <= self.moderate_absolute_threshold
            <= self.high_absolute_threshold
            <= self.critical_absolute_threshold
        ):
            raise ValueError(
                "Absolute thresholds must be monotonically increasing."
            )

        relative = (
            self.low_relative_threshold,
            self.moderate_relative_threshold,
            self.high_relative_threshold,
            self.critical_relative_threshold,
        )

        if any(value < 0 for value in relative):
            raise ValueError(
                "Relative disagreement thresholds must be >= 0."
            )

        if not (
            self.low_relative_threshold
            <= self.moderate_relative_threshold
            <= self.high_relative_threshold
            <= self.critical_relative_threshold
        ):
            raise ValueError(
                "Relative thresholds must be monotonically increasing."
            )

        if self.relative_denominator_floor <= 0:
            raise ValueError(
                "relative_denominator_floor must be > 0."
            )

        if self.confidence_scale <= 0:
            raise ValueError("confidence_scale must be > 0.")

        if self.direction_tolerance < 0:
            raise ValueError(
                "direction_tolerance must be >= 0."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "low_absolute_threshold": self.low_absolute_threshold,
            "moderate_absolute_threshold": (
                self.moderate_absolute_threshold
            ),
            "high_absolute_threshold": (
                self.high_absolute_threshold
            ),
            "critical_absolute_threshold": (
                self.critical_absolute_threshold
            ),
            "low_relative_threshold": self.low_relative_threshold,
            "moderate_relative_threshold": (
                self.moderate_relative_threshold
            ),
            "high_relative_threshold": (
                self.high_relative_threshold
            ),
            "critical_relative_threshold": (
                self.critical_relative_threshold
            ),
            "relative_denominator_floor": (
                self.relative_denominator_floor
            ),
            "confidence_scale": self.confidence_scale,
            "direction_tolerance": self.direction_tolerance,
        }


@dataclass(frozen=True)
class ModelComparisonPoint:
    """Comparison result for one prediction horizon/time point."""

    step: int
    time_minutes: float

    ml_prediction: float
    geco_prediction: float

    disagreement: float
    absolute_disagreement: float
    relative_disagreement: float

    level: DisagreementLevel
    direction: DisagreementDirection

    confidence: float
    valid: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "time_minutes": self.time_minutes,
            "ml_prediction": self.ml_prediction,
            "geco_prediction": self.geco_prediction,
            "disagreement": self.disagreement,
            "absolute_disagreement": self.absolute_disagreement,
            "relative_disagreement": self.relative_disagreement,
            "level": self.level.value,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "valid": self.valid,
            "metadata": dict(self.metadata),
        }


@dataclass
class ModelComparisonResult:
    """Complete ML/GECO comparison result."""

    points: List[ModelComparisonPoint]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def disagreements(self) -> np.ndarray:
        return np.asarray(
            [
                point.disagreement
                for point in self.points
                if point.valid
            ],
            dtype=float,
        )

    @property
    def absolute_disagreements(self) -> np.ndarray:
        return np.abs(self.disagreements)

    @property
    def relative_disagreements(self) -> np.ndarray:
        return np.asarray(
            [
                point.relative_disagreement
                for point in self.points
                if point.valid
            ],
            dtype=float,
        )

    @property
    def levels(self) -> List[DisagreementLevel]:
        return [
            point.level
            for point in self.points
            if point.valid
        ]

    def mean_absolute_disagreement(self) -> float:
        values = self.absolute_disagreements
        return (
            float(np.mean(values))
            if values.size
            else float("nan")
        )

    def rmse_disagreement(self) -> float:
        values = self.disagreements
        return (
            float(np.sqrt(np.mean(values ** 2)))
            if values.size
            else float("nan")
        )

    def max_absolute_disagreement(self) -> float:
        values = self.absolute_disagreements
        return (
            float(np.max(values))
            if values.size
            else float("nan")
        )

    def high_or_above_count(self) -> int:
        return sum(
            point.level
            in (
                DisagreementLevel.HIGH,
                DisagreementLevel.CRITICAL,
            )
            for point in self.points
            if point.valid
        )

    def critical_count(self) -> int:
        return sum(
            point.level == DisagreementLevel.CRITICAL
            for point in self.points
            if point.valid
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
                "rmse_disagreement": self.rmse_disagreement(),
                "max_absolute_disagreement": (
                    self.max_absolute_disagreement()
                ),
                "high_or_above_count": (
                    self.high_or_above_count()
                ),
                "critical_count": self.critical_count(),
            },
        }


class ModelComparator:
    """
    Compare aligned ML/DL and GECO predictions.

    The primary disagreement signal is:

        ML prediction - GECO prediction

    A positive value means ML predicts higher glucose than GECO.
    A negative value means ML predicts lower glucose than GECO.
    """

    def __init__(
        self,
        config: Optional[ModelComparisonConfig] = None,
    ) -> None:
        self.config = config or ModelComparisonConfig()

    # ------------------------------------------------------------------
    # Point-level helpers
    # ------------------------------------------------------------------

    def _relative_disagreement(
        self,
        ml_prediction: float,
        geco_prediction: float,
    ) -> float:
        denominator = max(
            abs(geco_prediction),
            self.config.relative_denominator_floor,
        )

        return abs(
            ml_prediction - geco_prediction
        ) / denominator

    def _direction(
        self,
        disagreement: float,
    ) -> DisagreementDirection:
        tolerance = self.config.direction_tolerance

        if disagreement > tolerance:
            return DisagreementDirection.ML_HIGHER

        if disagreement < -tolerance:
            return DisagreementDirection.ML_LOWER

        return DisagreementDirection.EQUAL

    def _level(
        self,
        absolute_disagreement: float,
        relative_disagreement: float,
    ) -> DisagreementLevel:
        """
        Determine disagreement level using the stronger of absolute and
        relative disagreement signals.
        """
        if (
            absolute_disagreement
            >= self.config.critical_absolute_threshold
            or relative_disagreement
            >= self.config.critical_relative_threshold
        ):
            return DisagreementLevel.CRITICAL

        if (
            absolute_disagreement
            >= self.config.high_absolute_threshold
            or relative_disagreement
            >= self.config.high_relative_threshold
        ):
            return DisagreementLevel.HIGH

        if (
            absolute_disagreement
            >= self.config.moderate_absolute_threshold
            or relative_disagreement
            >= self.config.moderate_relative_threshold
        ):
            return DisagreementLevel.MODERATE

        if (
            absolute_disagreement
            >= self.config.low_absolute_threshold
            or relative_disagreement
            >= self.config.low_relative_threshold
        ):
            return DisagreementLevel.LOW

        return DisagreementLevel.NONE

    def _confidence(
        self,
        absolute_disagreement: float,
    ) -> float:
        """
        Convert disagreement magnitude to bounded evidence confidence.

        This is an evidence mapping, not a probability calibration.
        """
        confidence = 1.0 - np.exp(
            -max(0.0, absolute_disagreement)
            / self.config.confidence_scale
        )

        return float(np.clip(confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Single comparison
    # ------------------------------------------------------------------

    def compare_point(
        self,
        ml_prediction: float,
        geco_prediction: float,
        step: int = 0,
        time_minutes: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ModelComparisonPoint:
        """Compare one ML prediction with one GECO prediction."""
        ml = float(ml_prediction)
        geco = float(geco_prediction)

        if not np.isfinite(ml):
            raise ValueError("ml_prediction must be finite.")

        if not np.isfinite(geco):
            raise ValueError("geco_prediction must be finite.")

        disagreement = ml - geco
        absolute = abs(disagreement)
        relative = self._relative_disagreement(
            ml_prediction=ml,
            geco_prediction=geco,
        )

        level = self._level(
            absolute_disagreement=absolute,
            relative_disagreement=relative,
        )

        point_metadata = dict(metadata or {})

        return ModelComparisonPoint(
            step=int(step),
            time_minutes=float(time_minutes),
            ml_prediction=ml,
            geco_prediction=geco,
            disagreement=float(disagreement),
            absolute_disagreement=float(absolute),
            relative_disagreement=float(relative),
            level=level,
            direction=self._direction(disagreement),
            confidence=self._confidence(absolute),
            valid=True,
            metadata=point_metadata,
        )

    # ------------------------------------------------------------------
    # Sequence comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        ml_predictions: Sequence[float],
        geco_predictions: Sequence[float],
        time_minutes: Optional[Sequence[float]] = None,
        skip_invalid: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ModelComparisonResult:
        """
        Compare aligned ML and GECO prediction sequences.
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

        points: List[ModelComparisonPoint] = []

        for index, (ml_value, geco_value, time) in enumerate(
            zip(ml, geco, times)
        ):
            valid = bool(
                np.isfinite(ml_value)
                and np.isfinite(geco_value)
                and np.isfinite(time)
            )

            if not valid:
                if skip_invalid:
                    continue

                points.append(
                    ModelComparisonPoint(
                        step=index,
                        time_minutes=float(time),
                        ml_prediction=float(ml_value),
                        geco_prediction=float(geco_value),
                        disagreement=float("nan"),
                        absolute_disagreement=float("nan"),
                        relative_disagreement=float("nan"),
                        level=DisagreementLevel.NONE,
                        direction=DisagreementDirection.NONE,
                        confidence=0.0,
                        valid=False,
                        metadata=dict(metadata or {}),
                    )
                )
                continue

            points.append(
                self.compare_point(
                    ml_prediction=float(ml_value),
                    geco_prediction=float(geco_value),
                    step=index,
                    time_minutes=float(time),
                    metadata=metadata,
                )
            )

        result_metadata = {
            "comparison_definition": (
                "ml_prediction - geco_prediction"
            ),
            "n_input": int(ml.size),
            "n_valid": int(
                sum(point.valid for point in points)
            ),
        }

        if metadata:
            result_metadata.update(dict(metadata))

        return ModelComparisonResult(
            points=points,
            metadata=result_metadata,
        )

    # ------------------------------------------------------------------
    # Trajectory helpers
    # ------------------------------------------------------------------

    def compare_trajectories(
        self,
        ml_trajectory: Any,
        geco_trajectory: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ModelComparisonResult:
        """
        Compare ML and GECO trajectory-like objects.

        The objects should expose:
            glucose
            time_minutes

        This matches the output conventions of the DT1 trajectory
        components.
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

        return self.compare(
            ml_predictions=ml_predictions,
            geco_predictions=geco_predictions,
            time_minutes=times,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Summary / utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def summarize(
        result: ModelComparisonResult,
    ) -> Dict[str, float]:
        """Return descriptive comparison statistics."""
        disagreement = result.disagreements
        absolute = result.absolute_disagreements
        relative = result.relative_disagreements

        if disagreement.size == 0:
            return {
                "n": 0.0,
                "mean_disagreement": float("nan"),
                "mean_absolute_disagreement": float("nan"),
                "rmse": float("nan"),
                "std": float("nan"),
                "max_absolute_disagreement": float("nan"),
                "mean_relative_disagreement": float("nan"),
            }

        return {
            "n": float(disagreement.size),
            "mean_disagreement": float(
                np.mean(disagreement)
            ),
            "mean_absolute_disagreement": float(
                np.mean(absolute)
            ),
            "rmse": float(
                np.sqrt(np.mean(disagreement ** 2))
            ),
            "std": float(np.std(disagreement)),
            "max_absolute_disagreement": float(
                np.max(absolute)
            ),
            "mean_relative_disagreement": float(
                np.mean(relative)
            ),
        }

    @staticmethod
    def count_by_level(
        result: ModelComparisonResult,
    ) -> Dict[str, int]:
        """Count valid comparison points by disagreement level."""
        counts = {
            level.value: 0
            for level in DisagreementLevel
        }

        for point in result.points:
            if point.valid:
                counts[point.level.value] += 1

        return counts

    @staticmethod
    def count_by_direction(
        result: ModelComparisonResult,
    ) -> Dict[str, int]:
        """Count valid comparison points by disagreement direction."""
        counts = {
            direction.value: 0
            for direction in DisagreementDirection
        }

        for point in result.points:
            if point.valid:
                counts[point.direction.value] += 1

        return counts


# ----------------------------------------------------------------------
# Functional helper
# ----------------------------------------------------------------------

def compare_ml_geco(
    ml_predictions: Sequence[float],
    geco_predictions: Sequence[float],
    time_minutes: Optional[Sequence[float]] = None,
    config: Optional[ModelComparisonConfig] = None,
) -> ModelComparisonResult:
    """Functional wrapper for ML/GECO comparison."""
    comparator = ModelComparator(config=config)

    return comparator.compare(
        ml_predictions=ml_predictions,
        geco_predictions=geco_predictions,
        time_minutes=time_minutes,
    )


__all__ = [
    "DisagreementLevel",
    "DisagreementDirection",
    "ModelComparisonConfig",
    "ModelComparisonPoint",
    "ModelComparisonResult",
    "ModelComparator",
    "compare_ml_geco",
]
