"""Temporal sequence construction utilities for DT2 prediction."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]

@dataclass(frozen=True)
class SequenceConfig:
    """Configuration for chronological sequence construction."""
    history_length: int = 12
    horizons: Tuple[int, ...] = (1,)
    stride: int = 1
    target_columns: Optional[Tuple[str, ...]] = None
    drop_invalid: bool = True

    def __post_init__(self) -> None:
        if self.history_length <= 0:
            raise ValueError("history_length must be positive")
        if not self.horizons or any(int(h) <= 0 for h in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        object.__setattr__(self, "horizons", tuple(sorted(set(map(int, self.horizons)))))

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

@dataclass
class SequenceBatch:
    """Batch of fixed-length temporal inputs and future targets."""
    x: np.ndarray
    y: Dict[int, np.ndarray]
    end_indices: np.ndarray
    target_indices: Dict[int, np.ndarray]
    feature_names: Optional[Tuple[str, ...]] = None
    target_names: Optional[Tuple[str, ...]] = None
    timestamps: Optional[np.ndarray] = None

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.x.shape[2])

    def target(self, horizon: int) -> np.ndarray:
        return self.y[horizon]


def _array(data: ArrayLike) -> np.ndarray:
    a = np.asarray(data, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"Expected 2-D data, got shape {a.shape}")
    return a


def build_sequences(
    data: Union[ArrayLike, "pd.DataFrame"],
    config: Optional[SequenceConfig] = None,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    target_columns: Optional[Sequence[str]] = None,
    timestamps: Optional[Sequence[object]] = None,
) -> SequenceBatch:
    """Build chronological windows without future leakage.

    For an input ending at row ``t`` and history length ``L``::

        X = data[t-L+1 : t+1]
        y[h] = target at t+h

    ``horizons`` are row offsets into the future (e.g. 1, 3, 6).
    """
    config = config or SequenceConfig()
    feature_names = target_names = None

    if pd is not None and isinstance(data, pd.DataFrame):
        if feature_columns is None:
            feature_columns = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
        if not feature_columns:
            raise ValueError("No feature columns supplied or detected")
        target_columns = target_columns or config.target_columns or feature_columns
        missing = [c for c in list(feature_columns) + list(target_columns) if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {sorted(set(missing))}")
        features = data.loc[:, feature_columns].to_numpy(dtype=float)
        targets = data.loc[:, target_columns].to_numpy(dtype=float)
        feature_names = tuple(feature_columns)
        target_names = tuple(target_columns)
        ts = data.index.to_numpy() if timestamps is None else np.asarray(timestamps, dtype=object)
    else:
        features = _array(data)
        targets = features
        ts = None if timestamps is None else np.asarray(timestamps, dtype=object)

    if len(features) != len(targets):
        raise ValueError("features and targets must have equal length")
    if ts is not None and len(ts) != len(features):
        raise ValueError("timestamps must have the same length as data")

    n = len(features)
    first_end = config.history_length - 1
    last_end = n - 1 - config.max_horizon
    x_list = []
    y_lists = {h: [] for h in config.horizons}
    ends = []
    target_idx = {h: [] for h in config.horizons}
    out_ts = []

    if last_end < first_end:
        x = np.empty((0, config.history_length, features.shape[1]), dtype=float)
        y = {h: np.empty((0, targets.shape[1]), dtype=float) for h in config.horizons}
        return SequenceBatch(x, y, np.empty(0, dtype=int),
                             {h: np.empty(0, dtype=int) for h in config.horizons},
                             feature_names, target_names,
                             np.empty(0, dtype=object) if ts is not None else None)

    for end in range(first_end, last_end + 1, config.stride):
        xw = features[end - config.history_length + 1:end + 1]
        ys = {h: targets[end + h] for h in config.horizons}
        valid = np.isfinite(xw).all() and all(np.isfinite(v).all() for v in ys.values())
        if not valid and config.drop_invalid:
            continue
        x_list.append(xw)
        for h, value in ys.items():
            y_lists[h].append(value)
            target_idx[h].append(end + h)
        ends.append(end)
        if ts is not None:
            out_ts.append(ts[end])

    x = np.stack(x_list) if x_list else np.empty((0, config.history_length, features.shape[1]))
    y = {h: np.stack(vals) if vals else np.empty((0, targets.shape[1])) for h, vals in y_lists.items()}
    return SequenceBatch(
        x=x.astype(float, copy=False), y=y,
        end_indices=np.asarray(ends, dtype=int),
        target_indices={h: np.asarray(v, dtype=int) for h, v in target_idx.items()},
        feature_names=feature_names, target_names=target_names,
        timestamps=np.asarray(out_ts, dtype=object) if ts is not None else None,
    )


def build_single_target_sequences(data, history_length=12, horizon=1, **kwargs) -> SequenceBatch:
    """Convenience wrapper for one future horizon."""
    return build_sequences(data, SequenceConfig(history_length, (horizon,),
                                                kwargs.pop("stride", 1),
                                                tuple(kwargs.get("target_columns")) if kwargs.get("target_columns") else None,
                                                kwargs.pop("drop_invalid", True)), **kwargs)


def build_multi_horizon_sequences(data, history_length, horizons, **kwargs) -> SequenceBatch:
    """Convenience wrapper for multiple future horizons."""
    return build_sequences(data, SequenceConfig(history_length, tuple(horizons),
                                                kwargs.pop("stride", 1),
                                                tuple(kwargs.get("target_columns")) if kwargs.get("target_columns") else None,
                                                kwargs.pop("drop_invalid", True)), **kwargs)


def split_temporal_batch(batch: SequenceBatch, fractions=(0.70, 0.15, 0.15)):
    """Chronologically split a sequence batch; no shuffling is performed."""
    if len(fractions) != 3 or any(f < 0 for f in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must be non-negative and sum to 1")
    n = batch.n_samples
    a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    def part(s, e):
        return SequenceBatch(batch.x[s:e], {h:v[s:e] for h,v in batch.y.items()},
                             batch.end_indices[s:e], {h:v[s:e] for h,v in batch.target_indices.items()},
                             batch.feature_names, batch.target_names,
                             batch.timestamps[s:e] if batch.timestamps is not None else None)
    return part(0,a), part(a,b), part(b,n)


def sequence_summary(batch: SequenceBatch) -> Dict[str, object]:
    """Return compact metadata for logging and validation."""
    return {"n_samples": batch.n_samples, "sequence_length": batch.sequence_length,
            "n_features": batch.n_features, "horizons": sorted(batch.y),
            "feature_names": batch.feature_names, "target_names": batch.target_names}

__all__ = ["SequenceConfig", "SequenceBatch", "build_sequences", "build_single_target_sequences",
           "build_multi_horizon_sequences", "split_temporal_batch", "sequence_summary"]
