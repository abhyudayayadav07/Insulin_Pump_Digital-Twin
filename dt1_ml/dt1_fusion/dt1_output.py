"""
Canonical output container for the DT1 hybrid predictive digital twin.

This module provides the common output interface that downstream
components can consume without depending on the internal DT1 modules.

DT1 pipeline:

    ML/DL model
         |
         +------------------+
         |                  |
         v                  v
    ML prediction       GECO model
                            |
                            v
                       GECO prediction
                            |
             +--------------+
             |
             v
       model comparison
             |
             v
       prediction fusion
             |
             +---- residual / CUSUM
             |
             v
        DT1 evidence
             |
             v
        DT1Output

The output object preserves the individual predictions, disagreement,
residual/CUSUM evidence, fused prediction, and metadata.

This is an engineering/research data container. It is not a clinical
decision object and does not directly control the insulin pump.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


class DT1OutputStatus(str, Enum):
    """High-level status of one DT1 output."""

    NORMAL = "normal"
    MONITOR = "monitor"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DT1Prediction:
    """Prediction from one DT1 model."""

    value: float
    horizon_minutes: float
    model_name: str
    confidence: Optional[float] = None
    unit: str = "mg/dL"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise ValueError("Prediction value must be finite.")

        if self.horizon_minutes < 0:
            raise ValueError(
                "horizon_minutes must be >= 0."
            )

        if self.confidence is not None:
            if not np.isfinite(self.confidence):
                raise ValueError(
                    "Prediction confidence must be finite."
                )

            if not 0 <= self.confidence <= 1:
                raise ValueError(
                    "Prediction confidence must be between 0 and 1."
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "horizon_minutes": self.horizon_minutes,
            "model_name": self.model_name,
            "confidence": self.confidence,
            "unit": self.unit,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DT1ModelConsistency:
    """ML/GECO model-consistency information."""

    disagreement: Optional[float] = None
    absolute_disagreement: Optional[float] = None

    ml_weight: Optional[float] = None
    geco_weight: Optional[float] = None

    level: DT1OutputStatus = DT1OutputStatus.UNKNOWN
    confidence: Optional[float] = None

    direction: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("disagreement", self.disagreement),
            ("absolute_disagreement", self.absolute_disagreement),
            ("ml_weight", self.ml_weight),
            ("geco_weight", self.geco_weight),
            ("confidence", self.confidence),
        ):
            if value is not None and not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite when provided."
                )

        if self.absolute_disagreement is not None:
            if self.absolute_disagreement < 0:
                raise ValueError(
                    "absolute_disagreement must be >= 0."
                )

        for name, value in (
            ("ml_weight", self.ml_weight),
            ("geco_weight", self.geco_weight),
            ("confidence", self.confidence),
        ):
            if value is not None and not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disagreement": self.disagreement,
            "absolute_disagreement": self.absolute_disagreement,
            "ml_weight": self.ml_weight,
            "geco_weight": self.geco_weight,
            "level": self.level.value,
            "confidence": self.confidence,
            "direction": self.direction,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DT1ResidualEvidence:
    """GECO residual and CUSUM monitoring information."""

    residual: Optional[float] = None
    normalized_residual: Optional[float] = None

    cusum_statistic: Optional[float] = None
    cusum_positive: Optional[float] = None
    cusum_negative: Optional[float] = None

    cusum_warning: bool = False
    cusum_alarm: bool = False

    anomaly_level: DT1OutputStatus = DT1OutputStatus.UNKNOWN
    anomaly_confidence: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("residual", self.residual),
            ("normalized_residual", self.normalized_residual),
            ("cusum_statistic", self.cusum_statistic),
            ("cusum_positive", self.cusum_positive),
            ("cusum_negative", self.cusum_negative),
            ("anomaly_confidence", self.anomaly_confidence),
        ):
            if value is not None and not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite when provided."
                )

        if (
            self.anomaly_confidence is not None
            and not 0 <= self.anomaly_confidence <= 1
        ):
            raise ValueError(
                "anomaly_confidence must be between 0 and 1."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "residual": self.residual,
            "normalized_residual": self.normalized_residual,
            "cusum_statistic": self.cusum_statistic,
            "cusum_positive": self.cusum_positive,
            "cusum_negative": self.cusum_negative,
            "cusum_warning": self.cusum_warning,
            "cusum_alarm": self.cusum_alarm,
            "anomaly_level": self.anomaly_level.value,
            "anomaly_confidence": self.anomaly_confidence,
            "metadata": dict(self.metadata),
        }


@dataclass
class DT1Output:
    """
    Canonical output from one DT1 prediction cycle.

    The object is designed to be passed into:
        - security evidence generation
        - hazard analysis
        - evidence fusion
        - decision engine
        - evaluation
        - telemetry / Omniverse bridge
    """

    timestamp: Optional[Any] = None

    ml_prediction: Optional[DT1Prediction] = None
    geco_prediction: Optional[DT1Prediction] = None

    fused_prediction: Optional[DT1Prediction] = None

    model_consistency: Optional[DT1ModelConsistency] = None
    residual_evidence: Optional[DT1ResidualEvidence] = None

    status: DT1OutputStatus = DT1OutputStatus.UNKNOWN

    evidence_confidence: float = 0.0

    sequence_index: Optional[int] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.evidence_confidence):
            raise ValueError(
                "evidence_confidence must be finite."
            )

        if not 0 <= self.evidence_confidence <= 1:
            raise ValueError(
                "evidence_confidence must be between 0 and 1."
            )

        if self.sequence_index is not None:
            if self.sequence_index < 0:
                raise ValueError(
                    "sequence_index must be >= 0."
                )

    @property
    def prediction(self) -> Optional[float]:
        """Return the fused prediction value when available."""
        if self.fused_prediction is not None:
            return self.fused_prediction.value

        if self.ml_prediction is not None:
            return self.ml_prediction.value

        if self.geco_prediction is not None:
            return self.geco_prediction.value

        return None

    @property
    def anomaly_detected(self) -> bool:
        """Return whether DT1 contains anomaly-level evidence."""
        if self.residual_evidence is None:
            return False

        return (
            self.residual_evidence.cusum_warning
            or self.residual_evidence.cusum_alarm
            or self.residual_evidence.anomaly_level
            in (
                DT1OutputStatus.WARNING,
                DT1OutputStatus.HIGH,
                DT1OutputStatus.CRITICAL,
            )
        )

    @property
    def critical(self) -> bool:
        """Return whether DT1 is currently in critical evidence state."""
        return self.status == DT1OutputStatus.CRITICAL

    @property
    def model_disagreement(self) -> Optional[float]:
        """Return signed ML-GECO disagreement."""
        if self.model_consistency is None:
            return None

        return self.model_consistency.disagreement

    @property
    def residual(self) -> Optional[float]:
        """Return the GECO residual."""
        if self.residual_evidence is None:
            return None

        return self.residual_evidence.residual

    @property
    def cusum_statistic(self) -> Optional[float]:
        """Return the current CUSUM statistic."""
        if self.residual_evidence is None:
            return None

        return self.residual_evidence.cusum_statistic

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the complete DT1 output."""
        return {
            "timestamp": self.timestamp,
            "ml_prediction": (
                self.ml_prediction.to_dict()
                if self.ml_prediction is not None
                else None
            ),
            "geco_prediction": (
                self.geco_prediction.to_dict()
                if self.geco_prediction is not None
                else None
            ),
            "fused_prediction": (
                self.fused_prediction.to_dict()
                if self.fused_prediction is not None
                else None
            ),
            "model_consistency": (
                self.model_consistency.to_dict()
                if self.model_consistency is not None
                else None
            ),
            "residual_evidence": (
                self.residual_evidence.to_dict()
                if self.residual_evidence is not None
                else None
            ),
            "status": self.status.value,
            "evidence_confidence": self.evidence_confidence,
            "sequence_index": self.sequence_index,
            "metadata": dict(self.metadata),
        }

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Serialize into a flat dictionary convenient for CSV/telemetry.
        """
        output: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "status": self.status.value,
            "evidence_confidence": self.evidence_confidence,
            "sequence_index": self.sequence_index,
        }

        if self.ml_prediction is not None:
            output["ml_prediction"] = self.ml_prediction.value
            output["ml_confidence"] = (
                self.ml_prediction.confidence
            )

        if self.geco_prediction is not None:
            output["geco_prediction"] = self.geco_prediction.value
            output["geco_confidence"] = (
                self.geco_prediction.confidence
            )

        if self.fused_prediction is not None:
            output["fused_prediction"] = self.fused_prediction.value

        if self.model_consistency is not None:
            output.update(
                {
                    "model_disagreement": (
                        self.model_consistency.disagreement
                    ),
                    "absolute_model_disagreement": (
                        self.model_consistency.absolute_disagreement
                    ),
                    "ml_weight": (
                        self.model_consistency.ml_weight
                    ),
                    "geco_weight": (
                        self.model_consistency.geco_weight
                    ),
                    "model_consistency_level": (
                        self.model_consistency.level.value
                    ),
                }
            )

        if self.residual_evidence is not None:
            output.update(
                {
                    "geco_residual": (
                        self.residual_evidence.residual
                    ),
                    "normalized_residual": (
                        self.residual_evidence.normalized_residual
                    ),
                    "cusum_statistic": (
                        self.residual_evidence.cusum_statistic
                    ),
                    "cusum_positive": (
                        self.residual_evidence.cusum_positive
                    ),
                    "cusum_negative": (
                        self.residual_evidence.cusum_negative
                    ),
                    "cusum_warning": (
                        self.residual_evidence.cusum_warning
                    ),
                    "cusum_alarm": (
                        self.residual_evidence.cusum_alarm
                    ),
                    "anomaly_level": (
                        self.residual_evidence.anomaly_level.value
                    ),
                }
            )

        output.update(self.metadata)

        return output


@dataclass
class DT1OutputSequence:
    """Container for multiple DT1 output cycles."""

    outputs: List[DT1Output]

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def predictions(self) -> np.ndarray:
        values = [
            output.prediction
            for output in self.outputs
            if output.prediction is not None
        ]

        return np.asarray(values, dtype=float)

    @property
    def statuses(self) -> List[DT1OutputStatus]:
        return [
            output.status
            for output in self.outputs
        ]

    def critical_count(self) -> int:
        return sum(
            output.status == DT1OutputStatus.CRITICAL
            for output in self.outputs
        )

    def anomaly_count(self) -> int:
        return sum(
            output.anomaly_detected
            for output in self.outputs
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outputs": [
                output.to_dict()
                for output in self.outputs
            ],
            "metadata": dict(self.metadata),
        }


class DT1OutputBuilder:
    """
    Helper for constructing canonical DT1Output objects.

    This builder accepts the result objects generated by the previously
    created DT1 modules while keeping dt1_output.py independent of their
    internal implementations.
    """

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> Optional[float]:
        if value is None:
            return None

        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    @staticmethod
    def _status_from_level(
        level: Any,
    ) -> DT1OutputStatus:
        if level is None:
            return DT1OutputStatus.UNKNOWN

        value = (
            level.value
            if hasattr(level, "value")
            else str(level).lower()
        )

        mapping = {
            "normal": DT1OutputStatus.NORMAL,
            "none": DT1OutputStatus.NORMAL,
            "monitor": DT1OutputStatus.MONITOR,
            "warning": DT1OutputStatus.WARNING,
            "low": DT1OutputStatus.MONITOR,
            "moderate": DT1OutputStatus.WARNING,
            "high": DT1OutputStatus.HIGH,
            "critical": DT1OutputStatus.CRITICAL,
            "anomaly": DT1OutputStatus.HIGH,
        }

        return mapping.get(
            value,
            DT1OutputStatus.UNKNOWN,
        )

    @staticmethod
    def _combine_statuses(
        *statuses: DT1OutputStatus,
    ) -> DT1OutputStatus:
        order = {
            DT1OutputStatus.UNKNOWN: 0,
            DT1OutputStatus.NORMAL: 1,
            DT1OutputStatus.MONITOR: 2,
            DT1OutputStatus.WARNING: 3,
            DT1OutputStatus.HIGH: 4,
            DT1OutputStatus.CRITICAL: 5,
        }

        valid = [
            status
            for status in statuses
            if status is not None
        ]

        if not valid:
            return DT1OutputStatus.UNKNOWN

        return max(
            valid,
            key=lambda status: order[status],
        )

    def from_components(
        self,
        ml_prediction: Optional[float] = None,
        geco_prediction: Optional[float] = None,
        fused_prediction: Optional[float] = None,
        horizon_minutes: float = 30.0,
        ml_confidence: Optional[float] = None,
        geco_confidence: Optional[float] = None,
        model_consistency: Optional[Any] = None,
        residual_evidence: Optional[Any] = None,
        status: Optional[Any] = None,
        evidence_confidence: Optional[float] = None,
        timestamp: Optional[Any] = None,
        sequence_index: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> DT1Output:
        """
        Construct DT1Output from primitive values and optional component
        result objects.
        """
        ml_object = (
            None
            if ml_prediction is None
            else DT1Prediction(
                value=float(ml_prediction),
                horizon_minutes=horizon_minutes,
                model_name="ml",
                confidence=ml_confidence,
            )
        )

        geco_object = (
            None
            if geco_prediction is None
            else DT1Prediction(
                value=float(geco_prediction),
                horizon_minutes=horizon_minutes,
                model_name="geco",
                confidence=geco_confidence,
            )
        )

        fused_object = (
            None
            if fused_prediction is None
            else DT1Prediction(
                value=float(fused_prediction),
                horizon_minutes=horizon_minutes,
                model_name="dt1_fused",
            )
        )

        consistency_object = (
            self._convert_model_consistency(
                model_consistency
            )
            if model_consistency is not None
            else None
        )

        residual_object = (
            self._convert_residual_evidence(
                residual_evidence
            )
            if residual_evidence is not None
            else None
        )

        component_statuses: List[DT1OutputStatus] = []

        if consistency_object is not None:
            component_statuses.append(
                consistency_object.level
            )

        if residual_object is not None:
            component_statuses.append(
                residual_object.anomaly_level
            )

        if status is not None:
            component_statuses.append(
                self._status_from_level(status)
            )

        final_status = self._combine_statuses(
            *component_statuses
        )

        if evidence_confidence is None:
            confidence_values = []

            if consistency_object is not None:
                if consistency_object.confidence is not None:
                    confidence_values.append(
                        consistency_object.confidence
                    )

            if residual_object is not None:
                if residual_object.anomaly_confidence is not None:
                    confidence_values.append(
                        residual_object.anomaly_confidence
                    )

            final_confidence = (
                float(max(confidence_values))
                if confidence_values
                else 0.0
            )
        else:
            final_confidence = float(evidence_confidence)

        result_metadata = dict(metadata or {})

        return DT1Output(
            timestamp=timestamp,
            ml_prediction=ml_object,
            geco_prediction=geco_object,
            fused_prediction=fused_object,
            model_consistency=consistency_object,
            residual_evidence=residual_object,
            status=final_status,
            evidence_confidence=final_confidence,
            sequence_index=sequence_index,
            metadata=result_metadata,
        )

    def _convert_model_consistency(
        self,
        result: Any,
    ) -> DT1ModelConsistency:
        """Convert a model_comparison/prediction_fusion result point."""
        disagreement = self._safe_float(
            getattr(result, "disagreement", None)
        )

        absolute = self._safe_float(
            getattr(result, "absolute_disagreement", None)
        )

        if absolute is None and disagreement is not None:
            absolute = abs(disagreement)

        ml_weight = self._safe_float(
            getattr(result, "ml_weight", None)
        )

        geco_weight = self._safe_float(
            getattr(result, "geco_weight", None)
        )

        confidence = self._safe_float(
            getattr(result, "confidence", None)
        )

        if confidence is None:
            confidence = self._safe_float(
                getattr(result, "disagreement_confidence", None)
            )

        level = self._status_from_level(
            getattr(result, "level", None)
        )

        if level == DT1OutputStatus.UNKNOWN:
            level = self._status_from_level(
                getattr(result, "status", None)
            )

        direction = getattr(
            result,
            "direction",
            None,
        )

        if hasattr(direction, "value"):
            direction = direction.value

        return DT1ModelConsistency(
            disagreement=disagreement,
            absolute_disagreement=absolute,
            ml_weight=ml_weight,
            geco_weight=geco_weight,
            level=level,
            confidence=confidence,
            direction=(
                None
                if direction is None
                else str(direction)
            ),
        )

    def _convert_residual_evidence(
        self,
        result: Any,
    ) -> DT1ResidualEvidence:
        """Convert a CUSUM/anomaly result into residual evidence."""
        residual = self._safe_float(
            getattr(result, "residual", None)
        )

        normalized = self._safe_float(
            getattr(result, "normalized_residual", None)
        )

        cusum = self._safe_float(
            getattr(result, "cusum_statistic", None)
        )

        positive = self._safe_float(
            getattr(result, "positive_cusum", None)
        )

        negative = self._safe_float(
            getattr(result, "negative_cusum", None)
        )

        if cusum is None:
            cusum = self._safe_float(
                getattr(result, "statistic", None)
            )

        warning = bool(
            getattr(result, "warning", False)
        )

        alarm = bool(
            getattr(result, "alarm", False)
        )

        level = self._status_from_level(
            getattr(result, "level", None)
        )

        if level == DT1OutputStatus.UNKNOWN:
            level = self._status_from_level(
                getattr(result, "state", None)
            )

        confidence = self._safe_float(
            getattr(result, "confidence", None)
        )

        return DT1ResidualEvidence(
            residual=residual,
            normalized_residual=normalized,
            cusum_statistic=cusum,
            cusum_positive=positive,
            cusum_negative=negative,
            cusum_warning=warning,
            cusum_alarm=alarm,
            anomaly_level=level,
            anomaly_confidence=confidence,
        )


def create_dt1_output(
    ml_prediction: Optional[float] = None,
    geco_prediction: Optional[float] = None,
    fused_prediction: Optional[float] = None,
    horizon_minutes: float = 30.0,
    ml_confidence: Optional[float] = None,
    geco_confidence: Optional[float] = None,
    model_consistency: Optional[Any] = None,
    residual_evidence: Optional[Any] = None,
    status: Optional[Any] = None,
    evidence_confidence: Optional[float] = None,
    timestamp: Optional[Any] = None,
    sequence_index: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> DT1Output:
    """Functional helper for constructing a DT1Output."""
    builder = DT1OutputBuilder()

    return builder.from_components(
        ml_prediction=ml_prediction,
        geco_prediction=geco_prediction,
        fused_prediction=fused_prediction,
        horizon_minutes=horizon_minutes,
        ml_confidence=ml_confidence,
        geco_confidence=geco_confidence,
        model_consistency=model_consistency,
        residual_evidence=residual_evidence,
        status=status,
        evidence_confidence=evidence_confidence,
        timestamp=timestamp,
        sequence_index=sequence_index,
        metadata=metadata,
    )


def output_to_record(
    output: DT1Output,
) -> Dict[str, Any]:
    """Convert a DT1Output into a flat telemetry/CSV record."""
    return output.to_flat_dict()


__all__ = [
    "DT1OutputStatus",
    "DT1Prediction",
    "DT1ModelConsistency",
    "DT1ResidualEvidence",
    "DT1Output",
    "DT1OutputSequence",
    "DT1OutputBuilder",
    "create_dt1_output",
    "output_to_record",
]
