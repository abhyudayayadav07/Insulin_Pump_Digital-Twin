"""Default formal safety specification registry for DT2."""

from .safety_specification import (
    SafetySpecification,
    SafetySpecificationRegistry,
    SpecificationSeverity,
    SpecificationType,
)


def create_default_safety_registry():
    specs = [
        SafetySpecification(
            "FS01",
            "Glucose lower bound",
            "Current glucose must remain above the configured physiological floor.",
            SpecificationType.STATE,
            SpecificationSeverity.CRITICAL,
            lambda c: float(c.get("glucose", 0.0)) >= 20.0,
        ),
        SafetySpecification(
            "FS02",
            "Glucose upper bound",
            "Current glucose must remain below the configured physiological ceiling.",
            SpecificationType.STATE,
            SpecificationSeverity.CRITICAL,
            lambda c: float(c.get("glucose", 0.0)) <= 600.0,
        ),
        SafetySpecification(
            "FS03",
            "Insulin nonnegative",
            "Insulin command must not be negative.",
            SpecificationType.CONTROL,
            SpecificationSeverity.HIGH,
            lambda c: float(c.get("insulin", 0.0)) >= 0.0,
        ),
        SafetySpecification(
            "FS04",
            "Predictive integrity",
            "Predictive authority requires acceptable model and sensor integrity.",
            SpecificationType.INTEGRITY,
            SpecificationSeverity.HIGH,
            lambda c: (
                float(c.get("sensor_integrity", 1.0)) >= 0.5
                and float(c.get("prediction_confidence", 1.0)) >= 0.25
            ),
        ),
        SafetySpecification(
            "FS05",
            "Attack-free authority",
            "Predictive control is not permitted when an active attack indicator is present.",
            SpecificationType.INTEGRITY,
            SpecificationSeverity.CRITICAL,
            lambda c: float(c.get("attack_indicator", 0.0)) < 0.5,
        ),
    ]
    return SafetySpecificationRegistry(specs)
