"""
Sensor attack models for the insulin-pump Digital Twin.

This module provides controlled, reproducible sensor-attack injection for
simulation and security evaluation. It operates on sensor observations rather
than real devices.

Supported attack modes include:
    - bias
    - spike
    - drift
    - noise
    - dropout
    - stuck_at
    - replay
    - spoofing

The module is designed to feed the resulting attacked signal into DT1, the
reactive-safe DT2, and the security/risk layer.

Typical flow:

    SimGlucose
        |
        v
    SensorAttackInjector
        |
        +---- clean CGM
        |
        +---- attacked CGM
                  |
                  v
        Predictive / Reactive DTs
                  |
                  v
             Security layer

This is a simulation component only. It does not interact with physical
medical devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


class SensorAttackType(str, Enum):
    """Supported simulated sensor attack mechanisms."""

    BIAS = "bias"
    SPIKE = "spike"
    DRIFT = "drift"
    NOISE = "noise"
    DROPOUT = "dropout"
    STUCK_AT = "stuck_at"
    REPLAY = "replay"
    SPOOFING = "spoofing"


@dataclass
class SensorAttackConfig:
    """Configuration for one sensor attack."""

    attack_type: SensorAttackType
    target_signal: str = "cgm"

    start_index: int = 0
    end_index: Optional[int] = None

    magnitude: float = 0.0
    noise_std: float = 0.0
    drift_rate: float = 0.0
    dropout_value: float = np.nan
    stuck_value: Optional[float] = None

    replay_start_index: Optional[int] = None
    replay_length: Optional[int] = None

    random_seed: Optional[int] = 42
    enabled: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.start_index < 0:
            raise ValueError("start_index must be >= 0.")

        if self.end_index is not None and self.end_index < self.start_index:
            raise ValueError("end_index must be >= start_index.")

        if self.replay_start_index is not None and self.replay_start_index < 0:
            raise ValueError("replay_start_index must be >= 0.")

        if self.replay_length is not None and self.replay_length <= 0:
            raise ValueError("replay_length must be > 0.")

        if self.noise_std < 0:
            raise ValueError("noise_std must be >= 0.")


@dataclass
class SensorAttackResult:
    """Result of applying a simulated sensor attack."""

    clean_signal: np.ndarray
    attacked_signal: np.ndarray

    attack_type: SensorAttackType
    target_signal: str

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
            "target_signal": self.target_signal,
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
    """Return the indices affected by an attack window."""
    start = max(0, int(start_index))
    end = n if end_index is None else min(n, int(end_index))

    if start >= n or start >= end:
        return np.array([], dtype=int)

    return np.arange(start, end, dtype=int)


def apply_bias(
    signal: np.ndarray,
    indices: np.ndarray,
    magnitude: float,
) -> np.ndarray:
    """Add a constant offset to the selected samples."""
    output = signal.copy()
    output[indices] += magnitude
    return output


def apply_spike(
    signal: np.ndarray,
    indices: np.ndarray,
    magnitude: float,
) -> np.ndarray:
    """Add an abrupt offset to the selected samples."""
    output = signal.copy()
    output[indices] += magnitude
    return output


def apply_drift(
    signal: np.ndarray,
    indices: np.ndarray,
    drift_rate: float,
) -> np.ndarray:
    """Apply a linearly increasing/decreasing sensor offset."""
    output = signal.copy()

    if len(indices) == 0:
        return output

    offset = np.arange(len(indices), dtype=float) * float(drift_rate)
    output[indices] += offset
    return output


def apply_noise(
    signal: np.ndarray,
    indices: np.ndarray,
    noise_std: float,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Add Gaussian measurement noise."""
    output = signal.copy()

    if len(indices) == 0 or noise_std == 0:
        return output

    rng = np.random.default_rng(random_seed)
    output[indices] += rng.normal(
        loc=0.0,
        scale=float(noise_std),
        size=len(indices),
    )

    return output


def apply_dropout(
    signal: np.ndarray,
    indices: np.ndarray,
    dropout_value: float = np.nan,
) -> np.ndarray:
    """Replace selected sensor observations with a dropout value."""
    output = signal.copy()
    output[indices] = dropout_value
    return output


def apply_stuck_at(
    signal: np.ndarray,
    indices: np.ndarray,
    stuck_value: Optional[float] = None,
) -> np.ndarray:
    """Hold the sensor at a fixed value."""
    output = signal.copy()

    if len(indices) == 0:
        return output

    value = (
        float(stuck_value)
        if stuck_value is not None
        else float(signal[indices[0]])
    )

    output[indices] = value
    return output


def apply_replay(
    signal: np.ndarray,
    indices: np.ndarray,
    replay_start_index: int,
    replay_length: Optional[int] = None,
) -> np.ndarray:
    """
    Replay an earlier segment of the sensor signal.

    The source segment is repeated cyclically across the attack window.
    """
    output = signal.copy()

    if len(indices) == 0:
        return output

    source_start = int(replay_start_index)

    if source_start < 0 or source_start >= len(signal):
        raise ValueError("replay_start_index is outside the signal.")

    available = len(signal) - source_start
    length = (
        available
        if replay_length is None
        else min(int(replay_length), available)
    )

    if length <= 0:
        raise ValueError("Replay source segment must contain samples.")

    source = signal[source_start : source_start + length]

    for position, index in enumerate(indices):
        output[index] = source[position % len(source)]

    return output


def apply_spoofing(
    signal: np.ndarray,
    indices: np.ndarray,
    spoof_value: float,
) -> np.ndarray:
    """Replace selected observations with an attacker-controlled value."""
    output = signal.copy()
    output[indices] = float(spoof_value)
    return output


def apply_sensor_attack(
    signal: Sequence[float],
    config: SensorAttackConfig,
) -> SensorAttackResult:
    """
    Apply one configured sensor attack to a one-dimensional signal.

    Parameters
    ----------
    signal:
        Clean sensor samples, e.g. CGM readings.
    config:
        Attack configuration.

    Returns
    -------
    SensorAttackResult
        Contains clean signal, attacked signal, and attack metadata.
    """
    config.validate()

    clean = np.asarray(signal, dtype=float).copy()

    if clean.ndim != 1:
        raise ValueError("signal must be one-dimensional.")

    attacked = clean.copy()

    if not config.enabled:
        return SensorAttackResult(
            clean_signal=clean,
            attacked_signal=attacked,
            attack_type=config.attack_type,
            target_signal=config.target_signal,
            affected_indices=[],
            attack_mask=np.zeros(len(clean), dtype=bool),
            metadata={"enabled": False},
        )

    indices = _attack_indices(
        len(clean),
        config.start_index,
        config.end_index,
    )

    if config.attack_type == SensorAttackType.BIAS:
        attacked = apply_bias(
            clean,
            indices,
            config.magnitude,
        )

    elif config.attack_type == SensorAttackType.SPIKE:
        attacked = apply_spike(
            clean,
            indices,
            config.magnitude,
        )

    elif config.attack_type == SensorAttackType.DRIFT:
        attacked = apply_drift(
            clean,
            indices,
            config.drift_rate,
        )

    elif config.attack_type == SensorAttackType.NOISE:
        attacked = apply_noise(
            clean,
            indices,
            config.noise_std,
            config.random_seed,
        )

    elif config.attack_type == SensorAttackType.DROPOUT:
        attacked = apply_dropout(
            clean,
            indices,
            config.dropout_value,
        )

    elif config.attack_type == SensorAttackType.STUCK_AT:
        attacked = apply_stuck_at(
            clean,
            indices,
            config.stuck_value,
        )

    elif config.attack_type == SensorAttackType.REPLAY:
        if config.replay_start_index is None:
            raise ValueError(
                "replay_start_index is required for replay attacks."
            )

        attacked = apply_replay(
            clean,
            indices,
            config.replay_start_index,
            config.replay_length,
        )

    elif config.attack_type == SensorAttackType.SPOOFING:
        spoof_value = (
            config.stuck_value
            if config.stuck_value is not None
            else config.magnitude
        )

        attacked = apply_spoofing(
            clean,
            indices,
            spoof_value,
        )

    else:
        raise ValueError(
            f"Unsupported sensor attack type: {config.attack_type}"
        )

    mask = ~np.isclose(
        clean,
        attacked,
        equal_nan=True,
    )

    # A dropout from a finite value to NaN is correctly considered a change.
    mask |= np.logical_xor(
        np.isnan(clean),
        np.isnan(attacked),
    )

    affected = np.flatnonzero(mask).astype(int).tolist()

    return SensorAttackResult(
        clean_signal=clean,
        attacked_signal=attacked,
        attack_type=config.attack_type,
        target_signal=config.target_signal,
        affected_indices=affected,
        attack_mask=mask,
        metadata={
            "enabled": True,
            "start_index": config.start_index,
            "end_index": config.end_index,
            "magnitude": config.magnitude,
            "noise_std": config.noise_std,
            "drift_rate": config.drift_rate,
            **config.metadata,
        },
    )


class SensorAttackInjector:
    """
    Reusable sensor-attack injector.

    The injector can be used to run multiple scenarios against the same clean
    trajectory while keeping attack configuration separate from the DT models.
    """

    def __init__(
        self,
        default_target_signal: str = "cgm",
    ):
        self.default_target_signal = default_target_signal
        self.history: List[SensorAttackResult] = []

    def inject(
        self,
        signal: Sequence[float],
        config: SensorAttackConfig,
    ) -> SensorAttackResult:
        """Apply an attack and store the result."""
        if not config.target_signal:
            config.target_signal = self.default_target_signal

        result = apply_sensor_attack(signal, config)
        self.history.append(result)
        return result

    def clear_history(self) -> None:
        self.history.clear()


def apply_attack_to_dataframe(
    dataframe: Any,
    config: SensorAttackConfig,
    *,
    signal_column: Optional[str] = None,
    attacked_column: Optional[str] = None,
) -> Any:
    """
    Apply an attack to a pandas-like DataFrame.

    Pandas is imported lazily so the core attack functions remain usable
    without pandas.
    """
    if not hasattr(dataframe, "copy") or not hasattr(dataframe, "__getitem__"):
        raise TypeError("dataframe must provide pandas-like DataFrame behavior.")

    column = signal_column or config.target_signal

    if column not in dataframe.columns:
        raise KeyError(f"Signal column '{column}' not found.")

    output = dataframe.copy()
    result = apply_sensor_attack(
        output[column].to_numpy(dtype=float),
        config,
    )

    target_column = attacked_column or column
    output[target_column] = result.attacked_signal

    # Add an explicit attack indicator without overwriting an existing one.
    mask_column = f"{target_column}_attack_mask"
    output[mask_column] = result.attack_mask

    return output


def create_bias_attack(
    magnitude: float,
    start_index: int,
    end_index: Optional[int] = None,
    target_signal: str = "cgm",
) -> SensorAttackConfig:
    """Convenience constructor for a bias attack."""
    return SensorAttackConfig(
        attack_type=SensorAttackType.BIAS,
        target_signal=target_signal,
        start_index=start_index,
        end_index=end_index,
        magnitude=magnitude,
    )


def create_drift_attack(
    drift_rate: float,
    start_index: int,
    end_index: Optional[int] = None,
    target_signal: str = "cgm",
) -> SensorAttackConfig:
    """Convenience constructor for a sensor-drift attack."""
    return SensorAttackConfig(
        attack_type=SensorAttackType.DRIFT,
        target_signal=target_signal,
        start_index=start_index,
        end_index=end_index,
        drift_rate=drift_rate,
    )


def create_dropout_attack(
    start_index: int,
    end_index: Optional[int] = None,
    target_signal: str = "cgm",
) -> SensorAttackConfig:
    """Convenience constructor for sensor dropout."""
    return SensorAttackConfig(
        attack_type=SensorAttackType.DROPOUT,
        target_signal=target_signal,
        start_index=start_index,
        end_index=end_index,
    )


def create_replay_attack(
    replay_start_index: int,
    start_index: int,
    end_index: Optional[int] = None,
    replay_length: Optional[int] = None,
    target_signal: str = "cgm",
) -> SensorAttackConfig:
    """Convenience constructor for a replay attack."""
    return SensorAttackConfig(
        attack_type=SensorAttackType.REPLAY,
        target_signal=target_signal,
        start_index=start_index,
        end_index=end_index,
        replay_start_index=replay_start_index,
        replay_length=replay_length,
    )


__all__ = [
    "SensorAttackType",
    "SensorAttackConfig",
    "SensorAttackResult",
    "SensorAttackInjector",
    "apply_sensor_attack",
    "apply_bias",
    "apply_spike",
    "apply_drift",
    "apply_noise",
    "apply_dropout",
    "apply_stuck_at",
    "apply_replay",
    "apply_spoofing",
    "apply_attack_to_dataframe",
    "create_bias_attack",
    "create_drift_attack",
    "create_dropout_attack",
    "create_replay_attack",
]
