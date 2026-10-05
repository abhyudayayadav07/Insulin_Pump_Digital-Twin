"""
DT1 evidence generation for the hybrid predictive digital twin.

This module converts outputs from the DT1 predictive components into a
structured evidence object for downstream security, hazard, and decision
layers.

Inputs can include:

    - ML/DL glucose prediction
    - GECO glucose prediction
    - ML/GECO disagreement
    - GECO residual
    - CUSUM statistic
    - CUSUM anomaly state
    - prediction confidence

Output:

    DT1Evidence

The evidence object deliberately keeps the individual signals visible.
It does not collapse them into a clinical probability and does not make
the final insulin-pump safety decision.

Expected downstream path:

    DT1 evidence
         |
         v
    evidence fusion
         |
         v
    hazard / risk reasoning
         |
         v
    decision engine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


class DT1EvidenceLevel(str, Enum):
    """Overall level of predictive/model-consistency evidence."""

    NORMAL = "normal"
    MONITOR = "monitor"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class DT1EvidenceType(str, Enum):
    """Individual evidence sources produced by DT1."""

    ML_PREDICTION = "ml_prediction"
    GECO_PREDICTION = "geco_prediction"
    MODEL_DISAGREEMENT = "model_disagreement"
    GECO_RESIDUAL = "geco_residual"
    CUSUM = "cusum"
    PREDICTION_CONFIDENCE = "prediction_confidence"
    DT1_CONSISTENCY = "dt1_consistency"


@dataclass(frozen=True)
class EvidenceGeneratorConfig:
    """
    Thresholds for converting DT1 signals into structured evidence.

    Thresholds are research defaults and should be calibrated against the
    intended baseline and attack/scenario datasets.
    """

    residual_warning: float = 15.0
    residual_high: float = 30.0
    residual_critical: float = 60.0

    disagreement_warning: float = 20.0
    disagreement_high: float = 40.0
    disagreement_critical: float = 70.0

    cusum_warning: float = 3.0
    cusum_high: float = 5.0
    cusum_critical: float = 8.0

    confidence_warning_floor: float = 0.50
    confidence_high_floor: float = 0.25
    confidence_critical_floor: float = 0.10

    prediction_horizon_minutes: float = 30.0

    def __post_init__(self) -> None:
        for name, value in (
            ("residual_warning", self.residual_warning),
            ("residual_high", self.residual_high),
            ("residual_critical", self.residual_critical),
            ("disagreement_warning", self.disagreement_warning),
            ("disagreement_high", self.disagreement_high),
            ("disagreement_critical", self.disagreement_critical),
            ("cusum_warning", self.cusum_warning),
            ("cusum_high", self.cusum_high),
            ("cusum_critical", self.cusum_critical),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")

        if not (
            self.residual_warning
            <= self.residual_high
            <= self.residual_critical
        ):
            raise ValueError(
                "Residual thresholds must be monotonically increasing."
            )

        if not (
            self.disagreement_warning
            <= self.disagreement_high
            <= self.disagreement_critical
        ):
            raise ValueError(
                "Disagreement thresholds must be monotonically increasing."
            )

        if not (
            self.cusum_warning
            <= self.cusum_high
            <= self.cusum_critical
        ):
            raise ValueError(
                "CUSUM thresholds must be monotonically increasing."
            )

        for name, value in (
            ("confidence_warning_floor", self.confidence_warning_floor),
            ("confidence_high_floor", self.confidence_high_floor),
            ("confidence_critical_floor", self.confidence_critical_floor),
        ):
            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

        if not (
            self.confidence_critical_floor
            <= self.confidence_high_floor
            <= self.confidence_warning_floor
        ):
            raise ValueError(
                "Confidence floors must be ordered from critical "
                "to warning."
            )

        if self.prediction_horizon_minutes <= 0:
            raise ValueError(
                "prediction_horizon_minutes must be > 0."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "residual_warning": self.residual_warning,
            "residual_high": self.residual_high,
            "residual_critical": self.residual_critical,
            "disagreement_warning": self.disagreement_warning,
            "disagreement_high": self.disagreement_high,
            "disagreement_critical": self.disagreement_critical,
            "cusum_warning": self.cusum_warning,
            "cusum_high": self.cusum_high,
            "cusum_critical": self.cusum_critical,
            "confidence_warning_floor": self.confidence_warning_floor,
            "confidence_high_floor": self.confidence_high_floor,
            "confidence_critical_floor": self.confidence_critical_floor,
            "prediction_horizon_minutes": self.prediction_horizon_minutes,
        }


@dataclass(frozen=True)
class EvidenceItem:
    """One atomic DT1 evidence signal."""

    evidence_type: DT1EvidenceType
    value: float
    level: DT1EvidenceLevel
    confidence: float
    direction: Optional[str] = None
    source: str = "DT1"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_type": self.evidence_type.value,
            "value": self.value,
            "level": self.level.value,
            "confidence": self.confidence,
            "direction": self.direction,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass
class DT1Evidence:
    """
    Structured DT1 evidence bundle.

    The individual evidence items are preserved so the downstream
    evidence-fusion layer can assign its own reliability/weights.
    """

    level: DT1EvidenceLevel

    confidence: float

    ml_prediction: Optional[float] = None
    geco_prediction: Optional[float] = None
    fused_prediction: Optional[float] = None

    disagreement: Optional[float] = None
    absolute_disagreement: Optional[float] = None

    geco_residual: Optional[float] = None
    normalized_residual: Optional[float] = None

    cusum_statistic: Optional[float] = None
    cusum_warning: bool = False
    cusum_alarm: bool = False

    evidence_items: List[EvidenceItem] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_anomaly_evidence(self) -> bool:
        return self.level in (
            DT1EvidenceLevel.WARNING,
            DT1EvidenceLevel.HIGH,
            DT1EvidenceLevel.CRITICAL,
        )

    @property
    def critical(self) -> bool:
        return self.level == DT1EvidenceLevel.CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "confidence": self.confidence,
            "ml_prediction": self.ml_prediction,
            "geco_prediction": self.geco_prediction,
            "fused_prediction": self.fused_prediction,
            "disagreement": self.disagreement,
            "absolute_disagreement": self.absolute_disagreement,
            "geco_residual": self.geco_residual,
            "normalized_residual": self.normalized_residual,
            "cusum_statistic": self.cusum_statistic,
            "cusum_warning": self.cusum_warning,
            "cusum_alarm": self.cusum_alarm,
            "evidence_items": [
                item.to_dict()
                for item in self.evidence_items
            ],
            "metadata": dict(self.metadata),
        }


@dataclass
class DT1EvidenceSequence:
    """Sequence of DT1 evidence bundles."""

    evidence: List[DT1Evidence]

    total_samples: int
    warning_count: int
    high_count: int
    critical_count: int

    maximum_confidence: float
    maximum_cusum: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "warning_count": self.warning_count,
            "high_count": self.high_count,
            "critical_count": self.critical_count,
            "maximum_confidence": self.maximum_confidence,
            "maximum_cusum": self.maximum_cusum,
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ],
        }


class DT1EvidenceGenerator:
    """
    Generate structured evidence from DT1 predictive outputs.

    Inputs are intentionally flexible so the generator can consume
    outputs from model_comparison.py, prediction_fusion.py,
    geco_residual.py, and anomaly_detector.py without tightly coupling
    those modules together.
    """

    def __init__(
        self,
        config: Optional[EvidenceGeneratorConfig] = None,
    ) -> None:
        self.config = config or EvidenceGeneratorConfig()

    # ------------------------------------------------------------------
    # Level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_level(
        levels: Sequence[DT1EvidenceLevel],
    ) -> DT1EvidenceLevel:
        order = {
            DT1EvidenceLevel.NORMAL: 0,
            DT1EvidenceLevel.MONITOR: 1,
            DT1EvidenceLevel.WARNING: 2,
            DT1EvidenceLevel.HIGH: 3,
            DT1EvidenceLevel.CRITICAL: 4,
        }

        if not levels:
            return DT1EvidenceLevel.NORMAL

        return max(
            levels,
            key=lambda level: order[level],
        )

    @staticmethod
    def _confidence_from_magnitude(
        value: float,
        scale: float,
    ) -> float:
        if scale <= 0:
            raise ValueError("scale must be > 0.")

        confidence = 1.0 - np.exp(
            -max(0.0, abs(value)) / scale
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def _residual_level(
        self,
        residual: float,
    ) -> DT1EvidenceLevel:
        magnitude = abs(residual)

        if magnitude >= self.config.residual_critical:
            return DT1EvidenceLevel.CRITICAL

        if magnitude >= self.config.residual_high:
            return DT1EvidenceLevel.HIGH

        if magnitude >= self.config.residual_warning:
            return DT1EvidenceLevel.WARNING

        return DT1EvidenceLevel.NORMAL

    def _disagreement_level(
        self,
        disagreement: float,
    ) -> DT1EvidenceLevel:
        magnitude = abs(disagreement)

        if magnitude >= self.config.disagreement_critical:
            return DT1EvidenceLevel.CRITICAL

        if magnitude >= self.config.disagreement_high:
            return DT1EvidenceLevel.HIGH

        if magnitude >= self.config.disagreement_warning:
            return DT1EvidenceLevel.WARNING

        return DT1EvidenceLevel.NORMAL

    def _cusum_level(
        self,
        statistic: float,
        alarm: bool = False,
        warning: bool = False,
    ) -> DT1EvidenceLevel:
        if alarm or statistic >= self.config.cusum_critical:
            return DT1EvidenceLevel.CRITICAL

        if statistic >= self.config.cusum_high:
            return DT1EvidenceLevel.HIGH

        if warning or statistic >= self.config.cusum_warning:
            return DT1EvidenceLevel.WARNING

        return DT1EvidenceLevel.NORMAL

    def _confidence_level(
        self,
        confidence: float,
    ) -> DT1EvidenceLevel:
        """
        Classify low predictive confidence as evidence about model
        reliability, not as evidence of a physiological hazard.
        """
        if confidence <= self.config.confidence_critical_floor:
            return DT1EvidenceLevel.CRITICAL

        if confidence <= self.config.confidence_high_floor:
            return DT1EvidenceLevel.HIGH

        if confidence <= self.config.confidence_warning_floor:
            return DT1EvidenceLevel.WARNING

        return DT1EvidenceLevel.NORMAL

    # ------------------------------------------------------------------
    # Evidence item constructors
    # ------------------------------------------------------------------

    def _prediction_item(
        self,
        evidence_type: DT1EvidenceType,
        value: float,
        source: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> EvidenceItem:
        return EvidenceItem(
            evidence_type=evidence_type,
            value=float(value),
            level=DT1EvidenceLevel.NORMAL,
            confidence=1.0,
            source=source,
            metadata=dict(metadata or {}),
        )

    def _disagreement_item(
        self,
        disagreement: float,
        direction: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> EvidenceItem:
        level = self._disagreement_level(disagreement)

        confidence = self._confidence_from_magnitude(
            disagreement,
            scale=max(
                self.config.disagreement_warning,
                1.0,
            ),
        )

        if direction is None:
            if disagreement > 0:
                direction = "ml_higher"
            elif disagreement < 0:
                direction = "ml_lower"
            else:
                direction = "equal"

        return EvidenceItem(
            evidence_type=DT1EvidenceType.MODEL_DISAGREEMENT,
            value=float(disagreement),
            level=level,
            confidence=confidence,
            direction=direction,
            source="model_comparison",
            metadata=dict(metadata or {}),
        )

    def _residual_item(
        self,
        residual: float,
        normalized_residual: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> EvidenceItem:
        level = self._residual_level(residual)

        confidence = self._confidence_from_magnitude(
            residual,
            scale=max(
                self.config.residual_warning,
                1.0,
            ),
        )

        return EvidenceItem(
            evidence_type=DT1EvidenceType.GECO_RESIDUAL,
            value=float(residual),
            level=level,
            confidence=confidence,
            direction=(
                "positive"
                if residual > 0
                else "negative"
                if residual < 0
                else "none"
            ),
            source="geco_residual",
            metadata={
                "normalized_residual": normalized_residual,
                **dict(metadata or {}),
            },
        )

    def _cusum_item(
        self,
        statistic: float,
        warning: bool = False,
        alarm: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> EvidenceItem:
        level = self._cusum_level(
            statistic=statistic,
            alarm=alarm,
            warning=warning,
        )

        confidence = self._confidence_from_magnitude(
            statistic,
            scale=max(
                self.config.cusum_warning,
                1.0,
            ),
        )

        return EvidenceItem(
            evidence_type=DT1EvidenceType.CUSUM,
            value=float(statistic),
            level=level,
            confidence=confidence,
            source="cusum",
            metadata=dict(metadata or {}),
        )

    def _confidence_item(
        self,
        confidence: float,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> EvidenceItem:
        if not np.isfinite(confidence):
            raise ValueError(
                "prediction confidence must be finite."
            )

        confidence = float(np.clip(confidence, 0.0, 1.0))

        return EvidenceItem(
            evidence_type=DT1EvidenceType.PREDICTION_CONFIDENCE,
            value=confidence,
            level=self._confidence_level(confidence),
            confidence=1.0 - confidence,
            source="dt1_prediction",
            metadata=dict(metadata or {}),
        )

    # ------------------------------------------------------------------
    # Single DT1 evidence bundle
    # ------------------------------------------------------------------

    def generate(
        self,
        ml_prediction: Optional[float] = None,
        geco_prediction: Optional[float] = None,
        fused_prediction: Optional[float] = None,
        disagreement: Optional[float] = None,
        geco_residual: Optional[float] = None,
        normalized_residual: Optional[float] = None,
        cusum_statistic: Optional[float] = None,
        cusum_warning: bool = False,
        cusum_alarm: bool = False,
        prediction_confidence: Optional[float] = None,
        disagreement_direction: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> DT1Evidence:
        """
        Generate one DT1 evidence bundle.

        All supplied numeric signals must be finite. Missing signals are
        simply omitted from the evidence bundle.
        """
        values = {
            "ml_prediction": ml_prediction,
            "geco_prediction": geco_prediction,
            "fused_prediction": fused_prediction,
            "disagreement": disagreement,
            "geco_residual": geco_residual,
            "normalized_residual": normalized_residual,
            "cusum_statistic": cusum_statistic,
            "prediction_confidence": prediction_confidence,
        }

        for name, value in values.items():
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(
                    f"{name} must be finite when provided."
                )

        items: List[EvidenceItem] = []
        levels: List[DT1EvidenceLevel] = []
        confidence_values: List[float] = []

        if ml_prediction is not None:
            items.append(
                self._prediction_item(
                    evidence_type=DT1EvidenceType.ML_PREDICTION,
                    value=float(ml_prediction),
                    source="ml_model",
                )
            )

        if geco_prediction is not None:
            items.append(
                self._prediction_item(
                    evidence_type=DT1EvidenceType.GECO_PREDICTION,
                    value=float(geco_prediction),
                    source="geco_model",
                )
            )

        if fused_prediction is not None:
            items.append(
                self._prediction_item(
                    evidence_type=DT1EvidenceType.DT1_CONSISTENCY,
                    value=float(fused_prediction),
                    source="prediction_fusion",
                )
            )

        if disagreement is not None:
            item = self._disagreement_item(
                disagreement=float(disagreement),
                direction=disagreement_direction,
            )
            items.append(item)
            levels.append(item.level)
            confidence_values.append(item.confidence)

        if geco_residual is not None:
            item = self._residual_item(
                residual=float(geco_residual),
                normalized_residual=normalized_residual,
            )
            items.append(item)
            levels.append(item.level)
            confidence_values.append(item.confidence)

        if cusum_statistic is not None:
            item = self._cusum_item(
                statistic=float(cusum_statistic),
                warning=cusum_warning,
                alarm=cusum_alarm,
            )
            items.append(item)
            levels.append(item.level)
            confidence_values.append(item.confidence)

        if prediction_confidence is not None:
            item = self._confidence_item(
                confidence=float(prediction_confidence),
            )
            items.append(item)

            # Low model confidence is reliability evidence. It can raise
            # the monitoring level but is kept conceptually distinct from
            # residual/anomaly evidence.
            confidence_level = item.level

            if confidence_level != DT1EvidenceLevel.NORMAL:
                levels.append(confidence_level)

        overall_level = self._max_level(levels)

        # Overall confidence represents the strongest available DT1
        # evidence magnitude. It is intentionally not a probability.
        if confidence_values:
            overall_confidence = float(
                np.clip(
                    max(confidence_values),
                    0.0,
                    1.0,
                )
            )
        elif prediction_confidence is not None:
            overall_confidence = float(
                np.clip(
                    1.0 - float(prediction_confidence),
                    0.0,
                    1.0,
                )
            )
        else:
            overall_confidence = 0.0

        result_metadata = {
            "evidence_source": "DT1",
            "prediction_horizon_minutes": (
                self.config.prediction_horizon_minutes
            ),
            "evidence_is_probability": False,
        }

        if metadata:
            result_metadata.update(dict(metadata))

        return DT1Evidence(
            level=overall_level,
            confidence=overall_confidence,
            ml_prediction=(
                None
                if ml_prediction is None
                else float(ml_prediction)
            ),
            geco_prediction=(
                None
                if geco_prediction is None
                else float(geco_prediction)
            ),
            fused_prediction=(
                None
                if fused_prediction is None
                else float(fused_prediction)
            ),
            disagreement=(
                None
                if disagreement is None
                else float(disagreement)
            ),
            absolute_disagreement=(
                None
                if disagreement is None
                else abs(float(disagreement))
            ),
            geco_residual=(
                None
                if geco_residual is None
                else float(geco_residual)
            ),
            normalized_residual=(
                None
                if normalized_residual is None
                else float(normalized_residual)
            ),
            cusum_statistic=(
                None
                if cusum_statistic is None
                else float(cusum_statistic)
            ),
            cusum_warning=bool(cusum_warning),
            cusum_alarm=bool(cusum_alarm),
            evidence_items=items,
            metadata=result_metadata,
        )

    # ------------------------------------------------------------------
    # Sequence generation
    # ------------------------------------------------------------------

    @staticmethod
    def _as_optional_array(
        values: Optional[Sequence[float]],
        length: int,
        name: str,
    ) -> Optional[np.ndarray]:
        if values is None:
            return None

        array = np.asarray(
            values,
            dtype=float,
        ).reshape(-1)

        if array.size != length:
            raise ValueError(
                f"{name} must have the same length as the other signals."
            )

        return array

    def generate_sequence(
        self,
        ml_predictions: Optional[Sequence[float]] = None,
        geco_predictions: Optional[Sequence[float]] = None,
        fused_predictions: Optional[Sequence[float]] = None,
        disagreements: Optional[Sequence[float]] = None,
        geco_residuals: Optional[Sequence[float]] = None,
        normalized_residuals: Optional[Sequence[float]] = None,
        cusum_statistics: Optional[Sequence[float]] = None,
        cusum_warnings: Optional[Sequence[bool]] = None,
        cusum_alarms: Optional[Sequence[bool]] = None,
        prediction_confidences: Optional[Sequence[float]] = None,
        time_minutes: Optional[Sequence[float]] = None,
        metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> DT1EvidenceSequence:
        """Generate DT1 evidence for an aligned sequence."""
        candidates = [
            values
            for values in (
                ml_predictions,
                geco_predictions,
                fused_predictions,
                disagreements,
                geco_residuals,
                normalized_residuals,
                cusum_statistics,
                cusum_warnings,
                cusum_alarms,
                prediction_confidences,
                time_minutes,
                metadata,
            )
            if values is not None
        ]

        if not candidates:
            return DT1EvidenceSequence(
                evidence=[],
                total_samples=0,
                warning_count=0,
                high_count=0,
                critical_count=0,
                maximum_confidence=0.0,
                maximum_cusum=0.0,
            )

        length = len(candidates[0])

        for values in candidates:
            if len(values) != length:
                raise ValueError(
                    "All supplied sequences must have the same length."
                )

        arrays = {
            "ml": self._as_optional_array(
                ml_predictions,
                length,
                "ml_predictions",
            ),
            "geco": self._as_optional_array(
                geco_predictions,
                length,
                "geco_predictions",
            ),
            "fused": self._as_optional_array(
                fused_predictions,
                length,
                "fused_predictions",
            ),
            "disagreement": self._as_optional_array(
                disagreements,
                length,
                "disagreements",
            ),
            "residual": self._as_optional_array(
                geco_residuals,
                length,
                "geco_residuals",
            ),
            "normalized_residual": self._as_optional_array(
                normalized_residuals,
                length,
                "normalized_residuals",
            ),
            "cusum": self._as_optional_array(
                cusum_statistics,
                length,
                "cusum_statistics",
            ),
            "confidence": self._as_optional_array(
                prediction_confidences,
                length,
                "prediction_confidences",
            ),
            "time": (
                np.asarray(time_minutes, dtype=float).reshape(-1)
                if time_minutes is not None
                else np.arange(length, dtype=float)
            ),
        }

        evidence: List[DT1Evidence] = []

        for index in range(length):
            item_metadata = (
                None
                if metadata is None
                else dict(metadata[index])
            )

            evidence.append(
                self.generate(
                    ml_prediction=(
                        None
                        if arrays["ml"] is None
                        else arrays["ml"][index]
                    ),
                    geco_prediction=(
                        None
                        if arrays["geco"] is None
                        else arrays["geco"][index]
                    ),
                    fused_prediction=(
                        None
                        if arrays["fused"] is None
                        else arrays["fused"][index]
                    ),
                    disagreement=(
                        None
                        if arrays["disagreement"] is None
                        else arrays["disagreement"][index]
                    ),
                    geco_residual=(
                        None
                        if arrays["residual"] is None
                        else arrays["residual"][index]
                    ),
                    normalized_residual=(
                        None
                        if arrays["normalized_residual"] is None
                        else arrays["normalized_residual"][index]
                    ),
                    cusum_statistic=(
                        None
                        if arrays["cusum"] is None
                        else arrays["cusum"][index]
                    ),
                    cusum_warning=(
                        bool(cusum_warnings[index])
                        if cusum_warnings is not None
                        else False
                    ),
                    cusum_alarm=(
                        bool(cusum_alarms[index])
                        if cusum_alarms is not None
                        else False
                    ),
                    prediction_confidence=(
                        None
                        if arrays["confidence"] is None
                        else arrays["confidence"][index]
                    ),
                    metadata={
                        "sequence_index": index,
                        "time_minutes": float(
                            arrays["time"][index]
                        ),
                        **(item_metadata or {}),
                    },
                )
            )

        levels = [item.level for item in evidence]
        confidence = [
            item.confidence for item in evidence
        ]
        cusum_values = [
            item.cusum_statistic
            for item in evidence
            if item.cusum_statistic is not None
        ]

        return DT1EvidenceSequence(
            evidence=evidence,
            total_samples=len(evidence),
            warning_count=sum(
                level in (
                    DT1EvidenceLevel.WARNING,
                    DT1EvidenceLevel.HIGH,
                    DT1EvidenceLevel.CRITICAL,
                )
                for level in levels
            ),
            high_count=sum(
                level in (
                    DT1EvidenceLevel.HIGH,
                    DT1EvidenceLevel.CRITICAL,
                )
                for level in levels
            ),
            critical_count=sum(
                level == DT1EvidenceLevel.CRITICAL
                for level in levels
            ),
            maximum_confidence=(
                float(max(confidence))
                if confidence
                else 0.0
            ),
            maximum_cusum=(
                float(max(cusum_values))
                if cusum_values
                else 0.0
            ),
        )


def generate_dt1_evidence(
    ml_prediction: Optional[float] = None,
    geco_prediction: Optional[float] = None,
    fused_prediction: Optional[float] = None,
    disagreement: Optional[float] = None,
    geco_residual: Optional[float] = None,
    normalized_residual: Optional[float] = None,
    cusum_statistic: Optional[float] = None,
    cusum_warning: bool = False,
    cusum_alarm: bool = False,
    prediction_confidence: Optional[float] = None,
    config: Optional[EvidenceGeneratorConfig] = None,
) -> DT1Evidence:
    """Functional wrapper for single-step DT1 evidence generation."""
    generator = DT1EvidenceGenerator(config=config)

    return generator.generate(
        ml_prediction=ml_prediction,
        geco_prediction=geco_prediction,
        fused_prediction=fused_prediction,
        disagreement=disagreement,
        geco_residual=geco_residual,
        normalized_residual=normalized_residual,
        cusum_statistic=cusum_statistic,
        cusum_warning=cusum_warning,
        cusum_alarm=cusum_alarm,
        prediction_confidence=prediction_confidence,
    )


__all__ = [
    "DT1EvidenceLevel",
    "DT1EvidenceType",
    "EvidenceGeneratorConfig",
    "EvidenceItem",
    "DT1Evidence",
    "DT1EvidenceSequence",
    "DT1EvidenceGenerator",
    "generate_dt1_evidence",
]
