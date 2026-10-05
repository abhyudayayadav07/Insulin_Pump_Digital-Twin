"""
Pump attack and actuator-fault models for the insulin-pump Digital Twin.

This module simulates attacks/faults at the pump-actuator layer after a
controller has generated an insulin command.

Supported mechanisms include:
    - overdelivery
    - underdelivery
    - stuck-on
    - stuck-off
    - delivery scaling
    - delivery bias
    - delivery delay
    - intermittent delivery
    - command rejection
    - actuator saturation

The module models the relationship between:

    commanded insulin
             |
             v
        pump actuator
             |
             v
       actual delivery

This distinction is important for the Digital Twin because the security layer
can compare commanded versus observed delivery and generate actuator residuals.

This module is for simulation/security evaluation only and does not control or
interface with physical medical devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class PumpAttackType(str, Enum):
    """Supported simulated pump/actuator attack mechanisms."""

    OVERDELIVERY = "overdelivery"
    UNDERDELIVERY = "underdelivery"
    STUCK_ON = "stuck_on"
    STUCK_OFF = "stuck_off"
    DELIVERY_SCALE = "delivery_scale"
    DELIVERY_BIAS = "delivery_bias"
    DELIVERY_DELAY = "delivery_delay"
    INTERMITTENT_DELIVERY = "intermittent_delivery"
    COMMAND_REJECTION = "command_rejection"
    ACTUATOR_SATURATION = "actuator_saturation"


@dataclass
class PumpAttackConfig:
    """Configuration for one simulated pump attack."""

    attack_type: PumpAttackType

    start_index: int = 0
    end_index: Optional[int] = None

    magnitude: float = 0.0
    scale_factor: float = 1.0
    fixed_delivery: Optional[float] = None

    delay_steps: int = 1
    delivery_interval: int = 2

    saturation_min: Optional[float] = 0.0
    saturation_max: Optional[float] = None

    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.start_index < 0:
            raise ValueError("start_index must be >= 0.")

        if self.end_index is not None and self.end_index < self.start_index:
            raise ValueError("end_index must be >= start_index.")

        if self.delay_steps < 0:
            raise ValueError("delay_steps must be >= 0.")

        if self.delivery_interval <= 0:
            raise ValueError("delivery_interval must be > 0.")

        if (
            self.saturation_min is not None
            and self.saturation_max is not None
            and self.saturation_min > self.saturation_max
        ):
            raise ValueError("saturation_min must be <= saturation_max.")

        if self.scale_factor < 0:
            raise ValueError("scale_factor must be >= 0.")


@dataclass
class PumpAttackResult:
    """Result of applying a pump attack."""

    commanded_delivery: np.ndarray
    clean_delivery: np.ndarray
    attacked_delivery: np.ndarray

    attack_type: PumpAttackType

    attack_mask: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=bool)
    )
    affected_indices: List[int] = field(default_factory=list)

    delivery_residual: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def changed_samples(self) -> int:
        return int(np.sum(self.attack_mask))

    @property
    def changed_fraction(self) -> float:
        if len(self.attack_mask) == 0:
            return 0.0
        return self.changed_samples / len(self.attack_mask)

    @property
    def total_commanded(self) -> float:
        return float(np.nansum(self.commanded_delivery))

    @property
    def total_actual(self) -> float:
        return float(np.nansum(self.attacked_delivery))

    @property
    def total_delivery_error(self) -> float:
        return self.total_actual - self.total_commanded

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_type": self.attack_type.value,
            "affected_indices": list(self.affected_indices),
            "changed_samples": self.changed_samples,
            "changed_fraction": self.changed_fraction,
            "total_commanded": self.total_commanded,
            "total_actual": self.total_actual,
            "total_delivery_error": self.total_delivery_error,
            "metadata": dict(self.metadata),
        }


def _attack_indices(
    n: int,
    start_index: int,
    end_index: Optional[int],
) -> np.ndarray:
    """Return indices inside the configured attack interval."""
    start = max(0, int(start_index))
    end = n if end_index is None else min(n, int(end_index))

    if start >= n or start >= end:
        return np.array([], dtype=int)

    return np.arange(start, end, dtype=int)


def apply_overdelivery(
    delivery: np.ndarray,
    indices: np.ndarray,
    magnitude: float,
) -> np.ndarray:
    """Increase actual pump delivery by a fixed amount."""
    output = delivery.copy()
    output[indices] += float(magnitude)
    return np.maximum(output, 0.0)


def apply_underdelivery(
    delivery: np.ndarray,
    indices: np.ndarray,
    magnitude: float,
) -> np.ndarray:
    """Decrease actual pump delivery by a fixed amount."""
    output = delivery.copy()
    output[indices] -= float(magnitude)
    return np.maximum(output, 0.0)


def apply_stuck_on(
    delivery: np.ndarray,
    indices: np.ndarray,
    fixed_delivery: Optional[float] = None,
) -> np.ndarray:
    """
    Simulate a pump stuck in an active delivery state.

    If fixed_delivery is omitted, the first attacked command is used as the
    stuck delivery level.
    """
    output = delivery.copy()

    if len(indices) == 0:
        return output

    value = (
        float(fixed_delivery)
        if fixed_delivery is not None
        else float(delivery[indices[0]])
    )

    output[indices] = max(0.0, value)
    return output


def apply_stuck_off(
    delivery: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Simulate a pump that stops delivering insulin."""
    output = delivery.copy()
    output[indices] = 0.0
    return output


def apply_delivery_scale(
    delivery: np.ndarray,
    indices: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """Scale actual pump delivery relative to commanded delivery."""
    output = delivery.copy()
    output[indices] *= float(scale_factor)
    return np.maximum(output, 0.0)


def apply_delivery_bias(
    delivery: np.ndarray,
    indices: np.ndarray,
    magnitude: float,
) -> np.ndarray:
    """Add a fixed delivery offset."""
    output = delivery.copy()
    output[indices] += float(magnitude)
    return np.maximum(output, 0.0)


def apply_delivery_delay(
    delivery: np.ndarray,
    indices: np.ndarray,
    delay_steps: int,
) -> np.ndarray:
    """
    Delay actual delivery by a fixed number of simulation steps.

    Values before the attack interval remain unchanged.
    """
    output = delivery.copy()
    original = delivery.copy()

    if len(indices) == 0 or delay_steps == 0:
        return output

    for index in indices:
        source = index - int(delay_steps)

        if source >= 0:
            output[index] = original[source]
        else:
            output[index] = 0.0

    return np.maximum(output, 0.0)


def apply_intermittent_delivery(
    delivery: np.ndarray,
    indices: np.ndarray,
    delivery_interval: int,
) -> np.ndarray:
    """
    Deliver only periodically during the attack window.

    Samples that do not fall on the configured interval are suppressed.
    """
    output = delivery.copy()

    for position, index in enumerate(indices):
        if position % int(delivery_interval) != 0:
            output[index] = 0.0

    return np.maximum(output, 0.0)


def apply_command_rejection(
    delivery: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Reject all selected insulin commands at the actuator."""
    output = delivery.copy()
    output[indices] = 0.0
    return output


def apply_actuator_saturation(
    delivery: np.ndarray,
    indices: np.ndarray,
    saturation_min: Optional[float] = 0.0,
    saturation_max: Optional[float] = None,
) -> np.ndarray:
    """Apply actuator delivery limits to selected samples."""
    output = delivery.copy()

    values = output[indices]

    if saturation_min is not None:
        values = np.maximum(values, float(saturation_min))

    if saturation_max is not None:
        values = np.minimum(values, float(saturation_max))

    output[indices] = values
    return np.maximum(output, 0.0)


def apply_pump_attack(
    commanded_delivery: Sequence[float],
    config: PumpAttackConfig,
    *,
    clean_delivery: Optional[Sequence[float]] = None,
) -> PumpAttackResult:
    """
    Apply a simulated pump attack to an insulin-delivery sequence.

    Parameters
    ----------
    commanded_delivery:
        Controller-generated insulin commands.
    config:
        Attack configuration.
    clean_delivery:
        Optional clean actuator-response sequence. If omitted, commanded
        delivery is treated as the nominal clean pump response.

    Returns
    -------
    PumpAttackResult
        Contains command, clean delivery, attacked delivery, attack mask, and
        command-vs-actual residual.
    """
    config.validate()

    commanded = np.asarray(commanded_delivery, dtype=float).copy()

    if commanded.ndim != 1:
        raise ValueError("commanded_delivery must be one-dimensional.")

    if clean_delivery is None:
        clean = commanded.copy()
    else:
        clean = np.asarray(clean_delivery, dtype=float).copy()

        if clean.shape != commanded.shape:
            raise ValueError(
                "clean_delivery must have the same shape as commanded_delivery."
            )

    attacked = clean.copy()

    if not config.enabled:
        residual = attacked - commanded
        return PumpAttackResult(
            commanded_delivery=commanded,
            clean_delivery=clean,
            attacked_delivery=attacked,
            attack_type=config.attack_type,
            attack_mask=np.zeros(len(commanded), dtype=bool),
            affected_indices=[],
            delivery_residual=residual,
            metadata={"enabled": False},
        )

    indices = _attack_indices(
        len(commanded),
        config.start_index,
        config.end_index,
    )

    attack = config.attack_type

    if attack == PumpAttackType.OVERDELIVERY:
        attacked = apply_overdelivery(
            clean,
            indices,
            config.magnitude,
        )

    elif attack == PumpAttackType.UNDERDELIVERY:
        attacked = apply_underdelivery(
            clean,
            indices,
            config.magnitude,
        )

    elif attack == PumpAttackType.STUCK_ON:
        attacked = apply_stuck_on(
            clean,
            indices,
            config.fixed_delivery,
        )

    elif attack == PumpAttackType.STUCK_OFF:
        attacked = apply_stuck_off(
            clean,
            indices,
        )

    elif attack == PumpAttackType.DELIVERY_SCALE:
        attacked = apply_delivery_scale(
            clean,
            indices,
            config.scale_factor,
        )

    elif attack == PumpAttackType.DELIVERY_BIAS:
        attacked = apply_delivery_bias(
            clean,
            indices,
            config.magnitude,
        )

    elif attack == PumpAttackType.DELIVERY_DELAY:
        attacked = apply_delivery_delay(
            clean,
            indices,
            config.delay_steps,
        )

    elif attack == PumpAttackType.INTERMITTENT_DELIVERY:
        attacked = apply_intermittent_delivery(
            clean,
            indices,
            config.delivery_interval,
        )

    elif attack == PumpAttackType.COMMAND_REJECTION:
        attacked = apply_command_rejection(
            clean,
            indices,
        )

    elif attack == PumpAttackType.ACTUATOR_SATURATION:
        attacked = apply_actuator_saturation(
            clean,
            indices,
            config.saturation_min,
            config.saturation_max,
        )

    else:
        raise ValueError(f"Unsupported pump attack type: {attack}")

    mask = ~np.isclose(
        clean,
        attacked,
        equal_nan=True,
    )

    mask |= np.logical_xor(
        np.isnan(clean),
        np.isnan(attacked),
    )

    affected = np.flatnonzero(mask).astype(int).tolist()
    residual = attacked - commanded

    return PumpAttackResult(
        commanded_delivery=commanded,
        clean_delivery=clean,
        attacked_delivery=attacked,
        attack_type=config.attack_type,
        attack_mask=mask,
        affected_indices=affected,
        delivery_residual=residual,
        metadata={
            "enabled": True,
            "start_index": config.start_index,
            "end_index": config.end_index,
            "magnitude": config.magnitude,
            "scale_factor": config.scale_factor,
            "delay_steps": config.delay_steps,
            "delivery_interval": config.delivery_interval,
            "saturation_min": config.saturation_min,
            "saturation_max": config.saturation_max,
            **config.metadata,
        },
    )


class PumpAttackInjector:
    """Reusable pump-attack injector for simulation experiments."""

    def __init__(self):
        self.history: List[PumpAttackResult] = []

    def inject(
        self,
        commanded_delivery: Sequence[float],
        config: PumpAttackConfig,
        *,
        clean_delivery: Optional[Sequence[float]] = None,
    ) -> PumpAttackResult:
        """Apply an attack and retain its result."""
        result = apply_pump_attack(
            commanded_delivery,
            config,
            clean_delivery=clean_delivery,
        )
        self.history.append(result)
        return result

    def clear_history(self) -> None:
        self.history.clear()


def apply_attack_to_dataframe(
    dataframe: Any,
    config: PumpAttackConfig,
    *,
    command_column: str = "insulin",
    clean_delivery_column: Optional[str] = None,
    attacked_column: str = "actual_insulin",
) -> Any:
    """
    Apply a pump attack to a pandas-like DataFrame.

    Adds:
        attacked_column
        attacked_column_attack_mask
        attacked_column_delivery_residual
    """
    if not hasattr(dataframe, "copy") or not hasattr(dataframe, "__getitem__"):
        raise TypeError("dataframe must provide pandas-like DataFrame behavior.")

    if command_column not in dataframe.columns:
        raise KeyError(f"Command column '{command_column}' not found.")

    output = dataframe.copy()

    clean_delivery = None

    if clean_delivery_column is not None:
        if clean_delivery_column not in output.columns:
            raise KeyError(
                f"Clean delivery column '{clean_delivery_column}' not found."
            )
        clean_delivery = output[clean_delivery_column].to_numpy(dtype=float)

    result = apply_pump_attack(
        output[command_column].to_numpy(dtype=float),
        config,
        clean_delivery=clean_delivery,
    )

    output[attacked_column] = result.attacked_delivery
    output[f"{attacked_column}_attack_mask"] = result.attack_mask
    output[f"{attacked_column}_delivery_residual"] = result.delivery_residual

    return output


def create_overdelivery_attack(
    magnitude: float,
    start_index: int,
    end_index: Optional[int] = None,
) -> PumpAttackConfig:
    """Convenience constructor for overdelivery."""
    return PumpAttackConfig(
        attack_type=PumpAttackType.OVERDELIVERY,
        start_index=start_index,
        end_index=end_index,
        magnitude=magnitude,
    )


def create_underdelivery_attack(
    magnitude: float,
    start_index: int,
    end_index: Optional[int] = None,
) -> PumpAttackConfig:
    """Convenience constructor for underdelivery."""
    return PumpAttackConfig(
        attack_type=PumpAttackType.UNDERDELIVERY,
        start_index=start_index,
        end_index=end_index,
        magnitude=magnitude,
    )


def create_stuck_off_attack(
    start_index: int,
    end_index: Optional[int] = None,
) -> PumpAttackConfig:
    """Convenience constructor for a stuck-off pump."""
    return PumpAttackConfig(
        attack_type=PumpAttackType.STUCK_OFF,
        start_index=start_index,
        end_index=end_index,
    )


def create_delivery_scale_attack(
    scale_factor: float,
    start_index: int,
    end_index: Optional[int] = None,
) -> PumpAttackConfig:
    """Convenience constructor for delivery scaling."""
    return PumpAttackConfig(
        attack_type=PumpAttackType.DELIVERY_SCALE,
        start_index=start_index,
        end_index=end_index,
        scale_factor=scale_factor,
    )


__all__ = [
    "PumpAttackType",
    "PumpAttackConfig",
    "PumpAttackResult",
    "PumpAttackInjector",
    "apply_pump_attack",
    "apply_overdelivery",
    "apply_underdelivery",
    "apply_stuck_on",
    "apply_stuck_off",
    "apply_delivery_scale",
    "apply_delivery_bias",
    "apply_delivery_delay",
    "apply_intermittent_delivery",
    "apply_command_rejection",
    "apply_actuator_saturation",
    "apply_attack_to_dataframe",
    "create_overdelivery_attack",
    "create_underdelivery_attack",
    "create_stuck_off_attack",
    "create_delivery_scale_attack",
]
