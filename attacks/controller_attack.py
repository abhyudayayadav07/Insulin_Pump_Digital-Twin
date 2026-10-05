"""
Controller attack models for the insulin-pump Digital Twin.

This module provides controlled, reproducible simulation of attacks or faults
that affect controller logic and insulin-command generation.

Supported mechanisms include:
    - biasing the controller target
    - modifying the controller output
    - delaying commands
    - replaying previous commands
    - suppressing commands
    - forcing a fixed command
    - scaling commands
    - sign/direction manipulation
    - command burst injection

The module is intended for simulation and security evaluation only. It does
not control a physical insulin pump.

Typical pipeline:

    physiological state
          |
          v
    controller policy
          |
          v
    controller_attack.py
          |
          v
    attacked insulin command
          |
          +------> DT2 reactive-safe comparison
          |
          +------> security / hazard analysis
          |
          v
       pump simulator
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


class ControllerAttackType(str, Enum):
    """Supported simulated controller attack mechanisms."""

    TARGET_BIAS = "target_bias"
    COMMAND_BIAS = "command_bias"
    COMMAND_SCALE = "command_scale"
    COMMAND_DELAY = "command_delay"
    COMMAND_REPLAY = "command_replay"
    COMMAND_SUPPRESSION = "command_suppression"
    FIXED_COMMAND = "fixed_command"
    DIRECTION_FLIP = "direction_flip"
    BURST_INJECTION = "burst_injection"
    COMMAND_CLAMP = "command_clamp"


@dataclass
class ControllerAttackConfig:
    """Configuration for one controller attack."""

    attack_type: ControllerAttackType

    start_index: int = 0
    end_index: Optional[int] = None

    magnitude: float = 0.0
    scale_factor: float = 1.0
    fixed_command: Optional[float] = None

    delay_steps: int = 1
    replay_start_index: Optional[int] = None
    replay_length: Optional[int] = None

    burst_value: float = 0.0
    burst_interval: int = 1

    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None

    suppression_value: float = 0.0
    enabled: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.start_index < 0:
            raise ValueError("start_index must be >= 0.")

        if self.end_index is not None and self.end_index < self.start_index:
            raise ValueError("end_index must be >= start_index.")

        if self.delay_steps < 0:
            raise ValueError("delay_steps must be >= 0.")

        if self.replay_start_index is not None and self.replay_start_index < 0:
            raise ValueError("replay_start_index must be >= 0.")

        if self.replay_length is not None and self.replay_length <= 0:
            raise ValueError("replay_length must be > 0.")

        if self.burst_interval <= 0:
            raise ValueError("burst_interval must be > 0.")

        if (
            self.clamp_min is not None
            and self.clamp_max is not None
            and self.clamp_min > self.clamp_max
        ):
            raise ValueError("clamp_min must be <= clamp_max.")


@dataclass
class ControllerAttackResult:
    """Result of applying a controller attack."""

    clean_command: np.ndarray
    attacked_command: np.ndarray

    attack_type: ControllerAttackType

    affected_indices: List[int] = field(default_factory=list)
    attack_mask: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=bool)
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_type": self.attack_type.value,
            "affected_indices": list(self.affected_indices),
            "changed_samples": self.changed_samples,
            "changed_fraction": self.changed_fraction,
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


def apply_target_bias(
    commands: np.ndarray,
    indices: np.ndarray,
    magnitude: float,
) -> np.ndarray:
    """
    Bias the controller's effective target-related command.

    At the command-stream level this is represented as an additive command
    perturbation. A richer implementation can apply the bias to the target
    glucose before invoking the controller.
    """
    output = commands.copy()
    output[indices] += float(magnitude)
    return output


def apply_command_bias(
    commands: np.ndarray,
    indices: np.ndarray,
    magnitude: float,
) -> np.ndarray:
    """Add an attacker-controlled offset to controller commands."""
    output = commands.copy()
    output[indices] += float(magnitude)
    return output


def apply_command_scale(
    commands: np.ndarray,
    indices: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """Multiply selected controller commands by a scale factor."""
    output = commands.copy()
    output[indices] *= float(scale_factor)
    return output


def apply_command_delay(
    commands: np.ndarray,
    indices: np.ndarray,
    delay_steps: int,
) -> np.ndarray:
    """
    Delay commands by a fixed number of simulation steps.

    Values before the attack window remain unchanged. During the attack,
    commands are replaced by earlier commands where possible.
    """
    output = commands.copy()

    if len(indices) == 0 or delay_steps == 0:
        return output

    original = commands.copy()

    for index in indices:
        source = index - int(delay_steps)

        if source >= 0:
            output[index] = original[source]
        else:
            output[index] = 0.0

    return output


def apply_command_replay(
    commands: np.ndarray,
    indices: np.ndarray,
    replay_start_index: int,
    replay_length: Optional[int] = None,
) -> np.ndarray:
    """Replay an earlier sequence of controller commands."""
    output = commands.copy()

    if len(indices) == 0:
        return output

    source_start = int(replay_start_index)

    if source_start < 0 or source_start >= len(commands):
        raise ValueError("replay_start_index is outside the command sequence.")

    available = len(commands) - source_start
    length = (
        available
        if replay_length is None
        else min(int(replay_length), available)
    )

    if length <= 0:
        raise ValueError("Replay source must contain at least one command.")

    source = commands[source_start : source_start + length]

    for position, index in enumerate(indices):
        output[index] = source[position % len(source)]

    return output


def apply_command_suppression(
    commands: np.ndarray,
    indices: np.ndarray,
    suppression_value: float = 0.0,
) -> np.ndarray:
    """Suppress selected controller commands."""
    output = commands.copy()
    output[indices] = float(suppression_value)
    return output


def apply_fixed_command(
    commands: np.ndarray,
    indices: np.ndarray,
    fixed_command: float,
) -> np.ndarray:
    """Replace selected controller outputs with an attacker-selected command."""
    output = commands.copy()
    output[indices] = float(fixed_command)
    return output


def apply_direction_flip(
    commands: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Invert the sign/direction of selected controller commands."""
    output = commands.copy()
    output[indices] *= -1.0
    return output


def apply_burst_injection(
    commands: np.ndarray,
    indices: np.ndarray,
    burst_value: float,
    burst_interval: int = 1,
) -> np.ndarray:
    """Inject an attacker-selected command at a configurable interval."""
    output = commands.copy()

    if len(indices) == 0:
        return output

    for position, index in enumerate(indices):
        if position % int(burst_interval) == 0:
            output[index] += float(burst_value)

    return output


def apply_command_clamp(
    commands: np.ndarray,
    indices: np.ndarray,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
) -> np.ndarray:
    """Force selected controller commands into an attacker-selected interval."""
    output = commands.copy()

    values = output[indices]

    if clamp_min is not None:
        values = np.maximum(values, float(clamp_min))

    if clamp_max is not None:
        values = np.minimum(values, float(clamp_max))

    output[indices] = values
    return output


def apply_controller_attack(
    commands: Sequence[float],
    config: ControllerAttackConfig,
) -> ControllerAttackResult:
    """
    Apply one configured controller attack to a command sequence.

    Parameters
    ----------
    commands:
        Clean controller output, normally insulin command values.
    config:
        Attack configuration.

    Returns
    -------
    ControllerAttackResult
        Clean and attacked command streams plus attack metadata.
    """
    config.validate()

    clean = np.asarray(commands, dtype=float).copy()

    if clean.ndim != 1:
        raise ValueError("commands must be one-dimensional.")

    attacked = clean.copy()

    if not config.enabled:
        return ControllerAttackResult(
            clean_command=clean,
            attacked_command=attacked,
            attack_type=config.attack_type,
            attack_mask=np.zeros(len(clean), dtype=bool),
            metadata={"enabled": False},
        )

    indices = _attack_indices(
        len(clean),
        config.start_index,
        config.end_index,
    )

    attack = config.attack_type

    if attack == ControllerAttackType.TARGET_BIAS:
        attacked = apply_target_bias(
            clean,
            indices,
            config.magnitude,
        )

    elif attack == ControllerAttackType.COMMAND_BIAS:
        attacked = apply_command_bias(
            clean,
            indices,
            config.magnitude,
        )

    elif attack == ControllerAttackType.COMMAND_SCALE:
        attacked = apply_command_scale(
            clean,
            indices,
            config.scale_factor,
        )

    elif attack == ControllerAttackType.COMMAND_DELAY:
        attacked = apply_command_delay(
            clean,
            indices,
            config.delay_steps,
        )

    elif attack == ControllerAttackType.COMMAND_REPLAY:
        if config.replay_start_index is None:
            raise ValueError(
                "replay_start_index is required for command replay."
            )

        attacked = apply_command_replay(
            clean,
            indices,
            config.replay_start_index,
            config.replay_length,
        )

    elif attack == ControllerAttackType.COMMAND_SUPPRESSION:
        attacked = apply_command_suppression(
            clean,
            indices,
            config.suppression_value,
        )

    elif attack == ControllerAttackType.FIXED_COMMAND:
        if config.fixed_command is None:
            raise ValueError(
                "fixed_command is required for FIXED_COMMAND attacks."
            )

        attacked = apply_fixed_command(
            clean,
            indices,
            config.fixed_command,
        )

    elif attack == ControllerAttackType.DIRECTION_FLIP:
        attacked = apply_direction_flip(
            clean,
            indices,
        )

    elif attack == ControllerAttackType.BURST_INJECTION:
        attacked = apply_burst_injection(
            clean,
            indices,
            config.burst_value,
            config.burst_interval,
        )

    elif attack == ControllerAttackType.COMMAND_CLAMP:
        attacked = apply_command_clamp(
            clean,
            indices,
            config.clamp_min,
            config.clamp_max,
        )

    else:
        raise ValueError(
            f"Unsupported controller attack type: {attack}"
        )

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

    return ControllerAttackResult(
        clean_command=clean,
        attacked_command=attacked,
        attack_type=config.attack_type,
        affected_indices=affected,
        attack_mask=mask,
        metadata={
            "enabled": True,
            "start_index": config.start_index,
            "end_index": config.end_index,
            "magnitude": config.magnitude,
            "scale_factor": config.scale_factor,
            "delay_steps": config.delay_steps,
            "replay_start_index": config.replay_start_index,
            "replay_length": config.replay_length,
            **config.metadata,
        },
    )


class ControllerAttackInjector:
    """
    Reusable controller-attack injector.

    Keeps attack scenarios independent from controller implementation and
    Digital Twin models.
    """

    def __init__(self):
        self.history: List[ControllerAttackResult] = []

    def inject(
        self,
        commands: Sequence[float],
        config: ControllerAttackConfig,
    ) -> ControllerAttackResult:
        """Apply an attack and store the result."""
        result = apply_controller_attack(commands, config)
        self.history.append(result)
        return result

    def clear_history(self) -> None:
        self.history.clear()


def apply_attack_to_dataframe(
    dataframe: Any,
    config: ControllerAttackConfig,
    *,
    command_column: str = "insulin",
    attacked_column: Optional[str] = None,
) -> Any:
    """
    Apply a controller attack to a pandas-like DataFrame.

    An explicit attack-mask column is added to support evaluation and
    synchronization with the hazard/security layer.
    """
    if not hasattr(dataframe, "copy") or not hasattr(dataframe, "__getitem__"):
        raise TypeError("dataframe must provide pandas-like DataFrame behavior.")

    if command_column not in dataframe.columns:
        raise KeyError(f"Command column '{command_column}' not found.")

    output = dataframe.copy()

    result = apply_controller_attack(
        output[command_column].to_numpy(dtype=float),
        config,
    )

    target_column = attacked_column or command_column
    output[target_column] = result.attacked_command
    output[f"{target_column}_attack_mask"] = result.attack_mask

    return output


def create_command_bias_attack(
    magnitude: float,
    start_index: int,
    end_index: Optional[int] = None,
) -> ControllerAttackConfig:
    """Convenience constructor for additive command manipulation."""
    return ControllerAttackConfig(
        attack_type=ControllerAttackType.COMMAND_BIAS,
        start_index=start_index,
        end_index=end_index,
        magnitude=magnitude,
    )


def create_command_scale_attack(
    scale_factor: float,
    start_index: int,
    end_index: Optional[int] = None,
) -> ControllerAttackConfig:
    """Convenience constructor for command scaling."""
    return ControllerAttackConfig(
        attack_type=ControllerAttackType.COMMAND_SCALE,
        start_index=start_index,
        end_index=end_index,
        scale_factor=scale_factor,
    )


def create_command_replay_attack(
    replay_start_index: int,
    start_index: int,
    end_index: Optional[int] = None,
    replay_length: Optional[int] = None,
) -> ControllerAttackConfig:
    """Convenience constructor for command replay."""
    return ControllerAttackConfig(
        attack_type=ControllerAttackType.COMMAND_REPLAY,
        start_index=start_index,
        end_index=end_index,
        replay_start_index=replay_start_index,
        replay_length=replay_length,
    )


def create_command_suppression_attack(
    start_index: int,
    end_index: Optional[int] = None,
) -> ControllerAttackConfig:
    """Convenience constructor for command suppression."""
    return ControllerAttackConfig(
        attack_type=ControllerAttackType.COMMAND_SUPPRESSION,
        start_index=start_index,
        end_index=end_index,
        suppression_value=0.0,
    )


__all__ = [
    "ControllerAttackType",
    "ControllerAttackConfig",
    "ControllerAttackResult",
    "ControllerAttackInjector",
    "apply_controller_attack",
    "apply_target_bias",
    "apply_command_bias",
    "apply_command_scale",
    "apply_command_delay",
    "apply_command_replay",
    "apply_command_suppression",
    "apply_fixed_command",
    "apply_direction_flip",
    "apply_burst_injection",
    "apply_command_clamp",
    "apply_attack_to_dataframe",
    "create_command_bias_attack",
    "create_command_scale_attack",
    "create_command_replay_attack",
    "create_command_suppression_attack",
]
