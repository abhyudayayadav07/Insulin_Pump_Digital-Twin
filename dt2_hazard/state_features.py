"""
State-feature construction for the DT2 hazard/survival digital twin.

DT2 is intentionally a hazard/survival digital twin rather than a
GlucOS-style reactive-safe controller.

This module defines the state vector x(t) used by DT2 hazard models:

    h(t | x(t))

It converts physiological, predictive, residual, CUSUM, and
cyber-physical integrity signals into a canonical, validated feature
representation.

The module is deliberately model-agnostic. It does NOT train a survival
model and does NOT assign clinical meaning to a feature by itself.

Recommended DT2 state groups
----------------------------

Physiology:
    glucose
    cgm
    glucose_rate
    glucose_acceleration
    insulin
    insulin_on_board
    meal

Predictive-twin evidence:
    dt1_prediction
    dt1_geco_prediction
    dt1_disagreement
    geco_residual
    cusum
    prediction_confidence

Cyber-physical integrity:
    sensor_integrity
    controller_integrity
    pump_integrity
    communication_integrity
    attack_indicator

Safety/risk:
    hazard_score
    hazard_probability
    risk_score

Time/context:
    time_minutes
    time_since_event
    time_of_day

The exact feature set can be configured. Missing optional features are
represented as NaN unless `fill_missing=True` is requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math


class FeatureGroup(str, Enum):
    """Logical groups of DT2 state features."""

    PHYSIOLOGY = "physiology"
    PREDICTIVE = "predictive"
    INTEGRITY = "integrity"
    SAFETY = "safety"
    CONTEXT = "context"
    CUSTOM = "custom"


class MissingValuePolicy(str, Enum):
    """How missing feature values should be handled."""

    NAN = "nan"
    ZERO = "zero"
    FORWARD_FILL = "forward_fill"
    ERROR = "error"


@dataclass(frozen=True)
class StateFeatureDefinition:
    """Definition of one canonical DT2 feature."""

    name: str
    group: FeatureGroup
    description: str
    required: bool = False
    default: float = math.nan
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def validate(self, value: Any) -> float:
        """Convert and validate a feature value."""
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Feature '{self.name}' must be numeric; "
                f"received {value!r}."
            ) from exc

        if not math.isfinite(numeric):
            raise ValueError(
                f"Feature '{self.name}' must be finite; "
                f"received {value!r}."
            )

        if (
            self.minimum is not None
            and numeric < self.minimum
        ):
            raise ValueError(
                f"Feature '{self.name}'={numeric} is below "
                f"minimum {self.minimum}."
            )

        if (
            self.maximum is not None
            and numeric > self.maximum
        ):
            raise ValueError(
                f"Feature '{self.name}'={numeric} is above "
                f"maximum {self.maximum}."
            )

        return numeric


@dataclass(frozen=True)
class StateFeatureConfig:
    """Configuration for constructing DT2 state features."""

    missing_policy: MissingValuePolicy = MissingValuePolicy.NAN

    glucose_min: float = 20.0
    glucose_max: float = 600.0

    insulin_min: float = 0.0
    meal_min: float = 0.0

    integrity_min: float = 0.0
    integrity_max: float = 1.0

    probability_min: float = 0.0
    probability_max: float = 1.0

    hazard_min: float = 0.0
    hazard_max: float = 100.0

    risk_min: float = 0.0
    risk_max: float = 25.0

    rate_window: int = 1

    include_context_features: bool = True
    include_predictive_features: bool = True
    include_integrity_features: bool = True
    include_safety_features: bool = True

    def __post_init__(self) -> None:
        if self.glucose_min >= self.glucose_max:
            raise ValueError(
                "glucose_min must be < glucose_max."
            )

        if self.insulin_min < 0:
            raise ValueError(
                "insulin_min must be >= 0."
            )

        if self.meal_min < 0:
            raise ValueError(
                "meal_min must be >= 0."
            )

        if not (
            0 <= self.integrity_min
            <= self.integrity_max
            <= 1
        ):
            raise ValueError(
                "Integrity bounds must lie in [0, 1]."
            )

        if not (
            0 <= self.probability_min
            <= self.probability_max
            <= 1
        ):
            raise ValueError(
                "Probability bounds must lie in [0, 1]."
            )

        if self.hazard_min < 0:
            raise ValueError(
                "hazard_min must be >= 0."
            )

        if self.hazard_max < self.hazard_min:
            raise ValueError(
                "hazard_max must be >= hazard_min."
            )

        if self.risk_min < 0 or self.risk_max < self.risk_min:
            raise ValueError(
                "Invalid risk bounds."
            )

        if self.rate_window < 1:
            raise ValueError(
                "rate_window must be >= 1."
            )


@dataclass
class DT2State:
    """
    Canonical DT2 state.

    Values are kept as a dictionary so the state can evolve without
    forcing every future hazard model to use exactly the same feature
    subset.
    """

    values: Dict[str, float]
    timestamp: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(
        self,
        name: str,
        default: float = math.nan,
    ) -> float:
        value = self.values.get(name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def to_dict(self) -> Dict[str, Any]:
        output = dict(self.values)
        if self.timestamp is not None:
            output["timestamp"] = self.timestamp
        return output

    def vector(
        self,
        feature_names: Sequence[str],
    ) -> List[float]:
        return [
            self.get(name)
            for name in feature_names
        ]


@dataclass(frozen=True)
class StateFeatureSchema:
    """Ordered schema used for deterministic model input."""

    definitions: Tuple[StateFeatureDefinition, ...]

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(
            definition.name
            for definition in self.definitions
        )

    def definition(
        self,
        name: str,
    ) -> StateFeatureDefinition:
        for item in self.definitions:
            if item.name == name:
                return item
        raise KeyError(
            f"Unknown DT2 state feature: {name}"
        )

    def groups(self) -> Dict[str, List[str]]:
        output: Dict[str, List[str]] = {}

        for definition in self.definitions:
            output.setdefault(
                definition.group.value,
                [],
            ).append(definition.name)

        return output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "features": [
                {
                    "name": item.name,
                    "group": item.group.value,
                    "description": item.description,
                    "required": item.required,
                    "default": item.default,
                    "minimum": item.minimum,
                    "maximum": item.maximum,
                }
                for item in self.definitions
            ]
        }


@dataclass(frozen=True)
class StateFeatureResult:
    """Output of state-feature construction."""

    state: DT2State
    feature_names: Tuple[str, ...]
    feature_vector: Tuple[float, ...]
    missing_features: Tuple[str, ...]
    source_fields: Dict[str, str]
    derived_features: Tuple[str, ...]
    warnings: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "feature_names": list(self.feature_names),
            "feature_vector": list(self.feature_vector),
            "missing_features": list(self.missing_features),
            "source_fields": dict(self.source_fields),
            "derived_features": list(self.derived_features),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Canonical feature schema
# ---------------------------------------------------------------------------

def default_feature_schema(
    config: Optional[StateFeatureConfig] = None,
) -> StateFeatureSchema:
    """Build the default DT2 feature schema."""
    config = config or StateFeatureConfig()

    definitions: List[StateFeatureDefinition] = [
        StateFeatureDefinition(
            name="glucose",
            group=FeatureGroup.PHYSIOLOGY,
            description="Current physiological glucose estimate.",
            required=True,
            minimum=config.glucose_min,
            maximum=config.glucose_max,
        ),
        StateFeatureDefinition(
            name="cgm",
            group=FeatureGroup.PHYSIOLOGY,
            description="Current continuous glucose monitor value.",
            required=True,
            minimum=config.glucose_min,
            maximum=config.glucose_max,
        ),
        StateFeatureDefinition(
            name="glucose_rate",
            group=FeatureGroup.PHYSIOLOGY,
            description="Glucose rate of change.",
        ),
        StateFeatureDefinition(
            name="glucose_acceleration",
            group=FeatureGroup.PHYSIOLOGY,
            description="Change in glucose rate.",
        ),
        StateFeatureDefinition(
            name="insulin",
            group=FeatureGroup.PHYSIOLOGY,
            description="Current insulin delivery.",
            minimum=config.insulin_min,
        ),
        StateFeatureDefinition(
            name="insulin_on_board",
            group=FeatureGroup.PHYSIOLOGY,
            description="Estimated active insulin remaining.",
            minimum=config.insulin_min,
        ),
        StateFeatureDefinition(
            name="meal",
            group=FeatureGroup.PHYSIOLOGY,
            description="Current or recent carbohydrate input.",
            minimum=config.meal_min,
        ),
    ]

    if config.include_predictive_features:
        definitions.extend(
            [
                StateFeatureDefinition(
                    name="dt1_prediction",
                    group=FeatureGroup.PREDICTIVE,
                    description="DT1 predicted future glucose.",
                    minimum=config.glucose_min,
                    maximum=config.glucose_max,
                ),
                StateFeatureDefinition(
                    name="dt1_geco_prediction",
                    group=FeatureGroup.PREDICTIVE,
                    description="GECO physiological prediction.",
                    minimum=config.glucose_min,
                    maximum=config.glucose_max,
                ),
                StateFeatureDefinition(
                    name="dt1_disagreement",
                    group=FeatureGroup.PREDICTIVE,
                    description="DT1 ML/GECO prediction disagreement.",
                    minimum=0.0,
                ),
                StateFeatureDefinition(
                    name="geco_residual",
                    group=FeatureGroup.PREDICTIVE,
                    description="Observed-minus-GECO residual.",
                ),
                StateFeatureDefinition(
                    name="cusum",
                    group=FeatureGroup.PREDICTIVE,
                    description="DT1 sequential anomaly statistic.",
                    minimum=0.0,
                ),
                StateFeatureDefinition(
                    name="prediction_confidence",
                    group=FeatureGroup.PREDICTIVE,
                    description="DT1 prediction confidence/evidence strength.",
                    minimum=config.probability_min,
                    maximum=config.probability_max,
                ),
            ]
        )

    if config.include_integrity_features:
        definitions.extend(
            [
                StateFeatureDefinition(
                    name="sensor_integrity",
                    group=FeatureGroup.INTEGRITY,
                    description="Sensor integrity score.",
                    minimum=config.integrity_min,
                    maximum=config.integrity_max,
                ),
                StateFeatureDefinition(
                    name="controller_integrity",
                    group=FeatureGroup.INTEGRITY,
                    description="Controller integrity score.",
                    minimum=config.integrity_min,
                    maximum=config.integrity_max,
                ),
                StateFeatureDefinition(
                    name="pump_integrity",
                    group=FeatureGroup.INTEGRITY,
                    description="Pump/actuator integrity score.",
                    minimum=config.integrity_min,
                    maximum=config.integrity_max,
                ),
                StateFeatureDefinition(
                    name="communication_integrity",
                    group=FeatureGroup.INTEGRITY,
                    description="Communication integrity score.",
                    minimum=config.integrity_min,
                    maximum=config.integrity_max,
                ),
                StateFeatureDefinition(
                    name="attack_indicator",
                    group=FeatureGroup.INTEGRITY,
                    description="Attack evidence indicator.",
                    minimum=config.probability_min,
                    maximum=config.probability_max,
                ),
            ]
        )

    if config.include_safety_features:
        definitions.extend(
            [
                StateFeatureDefinition(
                    name="hazard_score",
                    group=FeatureGroup.SAFETY,
                    description="Current hazard intensity/score.",
                    minimum=config.hazard_min,
                    maximum=config.hazard_max,
                ),
                StateFeatureDefinition(
                    name="hazard_probability",
                    group=FeatureGroup.SAFETY,
                    description="Estimated event probability/evidence.",
                    minimum=config.probability_min,
                    maximum=config.probability_max,
                ),
                StateFeatureDefinition(
                    name="risk_score",
                    group=FeatureGroup.SAFETY,
                    description="Current risk score.",
                    minimum=config.risk_min,
                    maximum=config.risk_max,
                ),
            ]
        )

    if config.include_context_features:
        definitions.extend(
            [
                StateFeatureDefinition(
                    name="time_minutes",
                    group=FeatureGroup.CONTEXT,
                    description="Simulation time in minutes.",
                    minimum=0.0,
                ),
                StateFeatureDefinition(
                    name="time_since_event",
                    group=FeatureGroup.CONTEXT,
                    description="Minutes since the relevant event began.",
                    minimum=0.0,
                ),
                StateFeatureDefinition(
                    name="time_of_day",
                    group=FeatureGroup.CONTEXT,
                    description="Time of day in hours.",
                    minimum=0.0,
                    maximum=24.0,
                ),
            ]
        )

    return StateFeatureSchema(
        definitions=tuple(definitions)
    )


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

class StateFeatureBuilder:
    """
    Build and validate a DT2 state from raw/derived observations.

    The builder supports aliases commonly encountered in SimGlucose,
    DT1, and security outputs.
    """

    DEFAULT_ALIASES: Dict[str, Tuple[str, ...]] = {
        "glucose": (
            "glucose",
            "blood_glucose",
            "bg",
            "plasma_glucose",
        ),
        "cgm": (
            "cgm",
            "CGM",
            "sensor_glucose",
        ),
        "insulin": (
            "insulin",
            "insulin_delivery",
            "actual_insulin",
            "delivered_insulin",
        ),
        "meal": (
            "meal",
            "carbs",
            "carbohydrate",
            "carbohydrates",
        ),
        "insulin_on_board": (
            "insulin_on_board",
            "iob",
            "IOB",
        ),
        "dt1_prediction": (
            "dt1_prediction",
            "ml_prediction",
            "predicted_glucose",
        ),
        "dt1_geco_prediction": (
            "dt1_geco_prediction",
            "geco_prediction",
            "geco_glucose",
        ),
        "dt1_disagreement": (
            "dt1_disagreement",
            "model_disagreement",
            "ml_geco_disagreement",
        ),
        "geco_residual": (
            "geco_residual",
            "residual",
            "physiological_residual",
        ),
        "cusum": (
            "cusum",
            "cusum_statistic",
            "cusum_score",
        ),
        "prediction_confidence": (
            "prediction_confidence",
            "dt1_confidence",
        ),
        "sensor_integrity": (
            "sensor_integrity",
            "cgm_integrity",
        ),
        "controller_integrity": (
            "controller_integrity",
        ),
        "pump_integrity": (
            "pump_integrity",
            "actuator_integrity",
        ),
        "communication_integrity": (
            "communication_integrity",
            "network_integrity",
        ),
        "attack_indicator": (
            "attack_indicator",
            "attack_score",
            "attack_probability",
        ),
        "hazard_score": (
            "hazard_score",
            "hazard_intensity",
        ),
        "hazard_probability": (
            "hazard_probability",
            "event_probability",
        ),
        "risk_score": (
            "risk_score",
            "risk",
        ),
        "time_minutes": (
            "time_minutes",
            "simulation_time",
            "time",
        ),
        "time_since_event": (
            "time_since_event",
        ),
        "time_of_day": (
            "time_of_day",
            "hour",
        ),
    }

    def __init__(
        self,
        config: Optional[StateFeatureConfig] = None,
        schema: Optional[StateFeatureSchema] = None,
        aliases: Optional[
            Mapping[str, Sequence[str]]
        ] = None,
    ) -> None:
        self.config = config or StateFeatureConfig()
        self.schema = (
            schema
            if schema is not None
            else default_feature_schema(self.config)
        )

        merged_aliases = dict(self.DEFAULT_ALIASES)

        if aliases:
            for canonical, names in aliases.items():
                merged_aliases[canonical] = tuple(names)

        self.aliases = merged_aliases

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return True

        return not math.isfinite(numeric)

    def _resolve_value(
        self,
        canonical_name: str,
        observations: Mapping[str, Any],
    ) -> Tuple[Optional[float], Optional[str]]:
        candidates = self.aliases.get(
            canonical_name,
            (canonical_name,),
        )

        for name in candidates:
            if name in observations:
                value = observations[name]

                if self._is_missing(value):
                    continue

                return float(value), name

        return None, None

    def _apply_missing_policy(
        self,
        definition: StateFeatureDefinition,
        value: Optional[float],
        previous_value: Optional[float] = None,
    ) -> float:
        if value is not None:
            return definition.validate(value)

        if self.config.missing_policy == MissingValuePolicy.ERROR:
            if definition.required:
                raise ValueError(
                    f"Required feature '{definition.name}' is missing."
                )

            raise ValueError(
                f"Feature '{definition.name}' is missing."
            )

        if self.config.missing_policy == MissingValuePolicy.ZERO:
            return 0.0

        if (
            self.config.missing_policy
            == MissingValuePolicy.FORWARD_FILL
        ):
            if previous_value is not None:
                return definition.validate(
                    previous_value
                )

        # NAN policy, and unresolved optional values under
        # forward-fill, use the definition default.
        if math.isnan(definition.default):
            return math.nan

        return definition.default

    def _derive_glucose_rate(
        self,
        current: Optional[float],
        previous: Optional[float],
        delta_minutes: Optional[float],
    ) -> Optional[float]:
        if (
            current is None
            or previous is None
            or delta_minutes is None
            or delta_minutes <= 0
        ):
            return None

        return (
            (current - previous)
            / delta_minutes
        )

    def _derive_acceleration(
        self,
        current_rate: Optional[float],
        previous_rate: Optional[float],
        delta_minutes: Optional[float],
    ) -> Optional[float]:
        if (
            current_rate is None
            or previous_rate is None
            or delta_minutes is None
            or delta_minutes <= 0
        ):
            return None

        return (
            (current_rate - previous_rate)
            / delta_minutes
        )

    def _derive_disagreement(
        self,
        dt1_prediction: Optional[float],
        geco_prediction: Optional[float],
    ) -> Optional[float]:
        if (
            dt1_prediction is None
            or geco_prediction is None
        ):
            return None

        return abs(
            dt1_prediction - geco_prediction
        )

    def _derive_time_of_day(
        self,
        time_minutes: Optional[float],
    ) -> Optional[float]:
        if time_minutes is None:
            return None

        minutes = float(time_minutes) % 1440.0
        return minutes / 60.0

    def build(
        self,
        observations: Mapping[str, Any],
        *,
        previous_observations: Optional[
            Mapping[str, Any]
        ] = None,
        timestamp: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> StateFeatureResult:
        """
        Build a canonical DT2 state.

        `previous_observations` is optional and is used to derive glucose
        rate and acceleration when those features are not explicitly
        supplied.
        """
        previous = previous_observations or {}

        values: Dict[str, float] = {}
        source_fields: Dict[str, str] = {}
        missing: List[str] = []
        derived: List[str] = []
        warnings: List[str] = []

        # First resolve direct fields.
        resolved: Dict[str, Optional[float]] = {}

        for definition in self.schema.definitions:
            value, source = self._resolve_value(
                definition.name,
                observations,
            )

            resolved[definition.name] = value

            if source is not None:
                source_fields[definition.name] = source

        # Derive glucose rate if absent.
        if resolved.get("glucose_rate") is None:
            current_glucose = resolved.get("glucose")
            previous_glucose, _ = self._resolve_value(
                "glucose",
                previous,
            )

            current_time, _ = self._resolve_value(
                "time_minutes",
                observations,
            )
            previous_time, _ = self._resolve_value(
                "time_minutes",
                previous,
            )

            delta_minutes = None

            if (
                current_time is not None
                and previous_time is not None
            ):
                delta_minutes = (
                    current_time - previous_time
                )

            rate = self._derive_glucose_rate(
                current_glucose,
                previous_glucose,
                delta_minutes,
            )

            if rate is not None:
                resolved["glucose_rate"] = rate
                derived.append("glucose_rate")

        # Derive glucose acceleration if absent.
        if resolved.get("glucose_acceleration") is None:
            current_rate = resolved.get("glucose_rate")

            previous_rate, _ = self._resolve_value(
                "glucose_rate",
                previous,
            )

            current_time, _ = self._resolve_value(
                "time_minutes",
                observations,
            )
            previous_time, _ = self._resolve_value(
                "time_minutes",
                previous,
            )

            delta_minutes = None

            if (
                current_time is not None
                and previous_time is not None
            ):
                delta_minutes = (
                    current_time - previous_time
                )

            acceleration = self._derive_acceleration(
                current_rate,
                previous_rate,
                delta_minutes,
            )

            if acceleration is not None:
                resolved["glucose_acceleration"] = acceleration
                derived.append("glucose_acceleration")

        # Derive DT1/GECO disagreement if absent.
        if resolved.get("dt1_disagreement") is None:
            disagreement = self._derive_disagreement(
                resolved.get("dt1_prediction"),
                resolved.get("dt1_geco_prediction"),
            )

            if disagreement is not None:
                resolved["dt1_disagreement"] = disagreement
                derived.append("dt1_disagreement")

        # Derive time of day from simulation time if absent.
        if (
            self.config.include_context_features
            and resolved.get("time_of_day") is None
        ):
            time_of_day = self._derive_time_of_day(
                resolved.get("time_minutes")
            )

            if time_of_day is not None:
                resolved["time_of_day"] = time_of_day
                derived.append("time_of_day")

        # Validate / fill.
        for definition in self.schema.definitions:
            raw_value = resolved.get(
                definition.name
            )

            previous_value = None

            if (
                self.config.missing_policy
                == MissingValuePolicy.FORWARD_FILL
            ):
                previous_value, _ = self._resolve_value(
                    definition.name,
                    previous,
                )

            try:
                value = self._apply_missing_policy(
                    definition,
                    raw_value,
                    previous_value,
                )
            except ValueError:
                raise

            values[definition.name] = value

            if (
                raw_value is None
                and math.isnan(value)
            ):
                missing.append(
                    definition.name
                )

        # Research-quality warnings.
        if "glucose" in values and "cgm" in values:
            glucose = values["glucose"]
            cgm = values["cgm"]

            if (
                math.isfinite(glucose)
                and math.isfinite(cgm)
            ):
                if abs(glucose - cgm) > 30:
                    warnings.append(
                        "CGM and glucose differ by more than "
                        "30 mg/dL."
                    )

        if "sensor_integrity" in values:
            integrity = values["sensor_integrity"]
            if (
                math.isfinite(integrity)
                and integrity < 0.5
            ):
                warnings.append(
                    "Sensor integrity is below 0.5."
                )

        if "attack_indicator" in values:
            attack = values["attack_indicator"]
            if (
                math.isfinite(attack)
                and attack >= 0.5
            ):
                warnings.append(
                    "Attack indicator is >= 0.5."
                )

        state = DT2State(
            values=values,
            timestamp=timestamp,
            metadata=dict(metadata or {}),
        )

        feature_names = self.schema.names

        feature_vector = tuple(
            state.get(name)
            for name in feature_names
        )

        return StateFeatureResult(
            state=state,
            feature_names=feature_names,
            feature_vector=feature_vector,
            missing_features=tuple(missing),
            source_fields=source_fields,
            derived_features=tuple(derived),
            warnings=tuple(warnings),
        )

    def build_sequence(
        self,
        observations: Sequence[
            Mapping[str, Any]
        ],
        *,
        timestamps: Optional[
            Sequence[Any]
        ] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> List[StateFeatureResult]:
        """Build DT2 states sequentially from observations."""
        results: List[StateFeatureResult] = []

        previous: Optional[
            Mapping[str, Any]
        ] = None

        for i, observation in enumerate(observations):
            timestamp = (
                timestamps[i]
                if timestamps is not None
                else None
            )

            result = self.build(
                observation,
                previous_observations=previous,
                timestamp=timestamp,
                metadata=metadata,
            )

            results.append(result)
            previous = observation

        return results

    def vectorize(
        self,
        state: DT2State,
    ) -> Tuple[float, ...]:
        """Convert a DT2State to the configured ordered feature vector."""
        return tuple(
            state.get(name)
            for name in self.schema.names
        )

    def feature_names(self) -> Tuple[str, ...]:
        return self.schema.names

    def feature_groups(self) -> Dict[str, List[str]]:
        return self.schema.groups()


def build_state_features(
    observations: Mapping[str, Any],
    *,
    previous_observations: Optional[
        Mapping[str, Any]
    ] = None,
    config: Optional[StateFeatureConfig] = None,
    schema: Optional[StateFeatureSchema] = None,
) -> StateFeatureResult:
    """Functional convenience wrapper."""
    builder = StateFeatureBuilder(
        config=config,
        schema=schema,
    )

    return builder.build(
        observations,
        previous_observations=previous_observations,
    )


__all__ = [
    "FeatureGroup",
    "MissingValuePolicy",
    "StateFeatureDefinition",
    "StateFeatureConfig",
    "DT2State",
    "StateFeatureSchema",
    "StateFeatureResult",
    "StateFeatureBuilder",
    "default_feature_schema",
    "build_state_features",
]
