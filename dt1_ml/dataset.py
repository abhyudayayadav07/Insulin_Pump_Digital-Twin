"""
dataset.py

Dataset and sliding-window sequence utilities for Digital Twin 1 (DT1).

Purpose
-------
Convert engineered time-series data into supervised temporal samples for
glucose forecasting models such as LSTM, GRU, and Transformer.

For a sequence length L and forecast horizon H:

    X[i] = [features(t-L+1), ..., features(t)]
    y[i] = glucose(t+H)

The implementation is chronological and does not randomly mix future
observations into past input windows.

Typical DT1 flow
----------------
data_loader.py
    -> preprocessing.py
    -> feature_engineering.py
    -> dataset.py
    -> LSTM / GRU / Transformer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_TARGET_COLUMN = "glucose"


def _validate_sequence_parameters(
    sequence_length: int,
    horizon: int,
) -> None:
    """Validate sliding-window parameters."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer.")

    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")


def _validate_feature_columns(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> list[str]:
    """Check that all requested model features exist."""
    columns = list(feature_columns)

    if not columns:
        raise ValueError("feature_columns cannot be empty.")

    missing = [
        column for column in columns
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            "The following feature columns are missing: "
            + ", ".join(missing)
        )

    non_numeric = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric:
        raise TypeError(
            "DT1 model features must be numeric. Non-numeric columns: "
            + ", ".join(non_numeric)
        )

    return columns


def create_sliding_windows(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str = DEFAULT_TARGET_COLUMN,
    sequence_length: int = 12,
    horizon: int = 1,
    drop_nan: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a time-series DataFrame into supervised sliding-window arrays.

    Parameters
    ----------
    df:
        Chronologically ordered feature DataFrame.
    feature_columns:
        Numerical columns supplied to the ML/DL model.
    target_column:
        Column to predict, normally ``glucose``.
    sequence_length:
        Number of past observations in each input sequence.
    horizon:
        Number of samples ahead to predict.

        horizon=1 means:
            X contains t-L+1 ... t
            y is glucose at t+1

        horizon=6 means:
            X contains t-L+1 ... t
            y is glucose at t+6

    drop_nan:
        If True, windows containing NaN/Inf values are removed.

    Returns
    -------
    X:
        Shape ``(num_samples, sequence_length, num_features)``.
    y:
        Shape ``(num_samples,)``.
    """
    _validate_sequence_parameters(sequence_length, horizon)

    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' is not present."
        )

    features = _validate_feature_columns(
        df,
        feature_columns,
    )

    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise TypeError(
            f"Target column '{target_column}' must be numeric."
        )

    feature_values = df[features].to_numpy(dtype=np.float32)
    target_values = df[target_column].to_numpy(dtype=np.float32)

    n_rows = len(df)

    # Need L observations for X and H future observations for y.
    max_start = n_rows - sequence_length - horizon + 1

    if max_start <= 0:
        return (
            np.empty(
                (0, sequence_length, len(features)),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.float32),
        )

    X_windows: list[np.ndarray] = []
    y_values: list[np.float32] = []

    for start in range(max_start):
        end = start + sequence_length
        target_index = end + horizon - 1

        X_window = feature_values[start:end]
        y_value = target_values[target_index]

        if drop_nan:
            if (
                not np.isfinite(X_window).all()
                or not np.isfinite(y_value)
            ):
                continue

        X_windows.append(X_window)
        y_values.append(y_value)

    if not X_windows:
        return (
            np.empty(
                (0, sequence_length, len(features)),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.float32),
        )

    return (
        np.stack(X_windows).astype(np.float32),
        np.asarray(y_values, dtype=np.float32),
    )


def create_multi_horizon_windows(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str = DEFAULT_TARGET_COLUMN,
    sequence_length: int = 12,
    horizons: Sequence[int] = (1, 3, 6, 12),
    drop_nan: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create windows for multiple prediction horizons.

    Returns
    -------
    X:
        Shape ``(N, sequence_length, num_features)``.

    y:
        Shape ``(N, num_horizons)``.

    Only windows for which every requested horizon is available are created.
    """
    if not horizons:
        raise ValueError("horizons cannot be empty.")

    if any(h <= 0 for h in horizons):
        raise ValueError("All horizons must be positive integers.")

    features = _validate_feature_columns(
        df,
        feature_columns,
    )

    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' is not present."
        )

    feature_values = df[features].to_numpy(dtype=np.float32)
    target_values = df[target_column].to_numpy(dtype=np.float32)

    max_horizon = max(horizons)
    max_start = (
        len(df) - sequence_length - max_horizon + 1
    )

    if max_start <= 0:
        return (
            np.empty(
                (0, sequence_length, len(features)),
                dtype=np.float32,
            ),
            np.empty(
                (0, len(horizons)),
                dtype=np.float32,
            ),
        )

    X_windows = []
    y_windows = []

    for start in range(max_start):
        end = start + sequence_length

        X_window = feature_values[start:end]
        y_window = np.asarray(
            [
                target_values[
                    end + horizon - 1
                ]
                for horizon in horizons
            ],
            dtype=np.float32,
        )

        if drop_nan:
            if (
                not np.isfinite(X_window).all()
                or not np.isfinite(y_window).all()
            ):
                continue

        X_windows.append(X_window)
        y_windows.append(y_window)

    if not X_windows:
        return (
            np.empty(
                (0, sequence_length, len(features)),
                dtype=np.float32,
            ),
            np.empty(
                (0, len(horizons)),
                dtype=np.float32,
            ),
        )

    return (
        np.stack(X_windows).astype(np.float32),
        np.stack(y_windows).astype(np.float32),
    )


def create_sequence_timestamps(
    df: pd.DataFrame,
    *,
    sequence_length: int = 12,
    horizon: int = 1,
    time_column: str = "time",
    drop_nan: bool = True,
    feature_columns: Optional[Sequence[str]] = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Create timestamp metadata corresponding to generated sequences.

    This is useful for plotting predicted vs. actual glucose later.

    The returned table contains:
        input_start_time
        input_end_time
        target_time
    """
    _validate_sequence_parameters(sequence_length, horizon)

    if time_column not in df.columns:
        raise KeyError(
            f"Time column '{time_column}' is required."
        )

    timestamps = pd.to_datetime(
        df[time_column],
        errors="coerce",
    )

    metadata = []

    max_start = (
        len(df) - sequence_length - horizon + 1
    )

    for start in range(max(0, max_start)):
        end = start + sequence_length
        target_index = end + horizon - 1

        values = timestamps.iloc[
            start:end
        ]

        target_time = timestamps.iloc[target_index]

        if drop_nan:
            if (
                values.isna().any()
                or pd.isna(target_time)
            ):
                continue

            if feature_columns is not None:
                feature_values = df.iloc[
                    start:end
                ][list(feature_columns)]

                target_values = df.iloc[
                    target_index
                ][target_column]

                if (
                    feature_values.isna().any().any()
                    or pd.isna(target_values)
                ):
                    continue

        metadata.append(
            {
                "input_start_time": values.iloc[0],
                "input_end_time": values.iloc[-1],
                "target_time": target_time,
            }
        )

    return pd.DataFrame(metadata)


class GlucoseSequenceDataset(Dataset):
    """
    PyTorch Dataset for single-horizon DT1 glucose prediction.

    Each item returns:
        {
            "x": tensor of shape (sequence_length, num_features),
            "y": tensor containing target glucose,
        }

    Returning a dictionary makes the dataset easier to extend later with
    timestamps, patient IDs, or additional supervision signals.
    """

    def __init__(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        self.X = X.to(dtype=dtype)
        self.y = y.to(dtype=dtype)

        if self.X.ndim != 3:
            raise ValueError(
                "X must have shape "
                "(num_samples, sequence_length, num_features)."
            )

        if self.y.ndim not in {1, 2}:
            raise ValueError(
                "y must have one or two dimensions."
            )

        if len(self.X) != len(self.y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.X[index],
            "y": self.y[index],
        }

    @property
    def num_features(self) -> int:
        """Number of input features per time step."""
        return int(self.X.shape[-1])

    @property
    def sequence_length(self) -> int:
        """Number of historical time steps in each sample."""
        return int(self.X.shape[1])


@dataclass
class SequenceBuilder:
    """
    Reusable configuration for constructing DT1 temporal datasets.
    """

    sequence_length: int = 12
    horizon: int = 1
    target_column: str = DEFAULT_TARGET_COLUMN
    drop_nan: bool = True

    def transform(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build X/y arrays using the configured sequence parameters."""
        return create_sliding_windows(
            df,
            feature_columns=feature_columns,
            target_column=self.target_column,
            sequence_length=self.sequence_length,
            horizon=self.horizon,
            drop_nan=self.drop_nan,
        )

    def create_dataset(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> GlucoseSequenceDataset:
        """Build a PyTorch Dataset directly."""
        X, y = self.transform(
            df,
            feature_columns,
        )

        return GlucoseSequenceDataset(X, y)


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
):
    """
    Create a PyTorch DataLoader.

    ``shuffle=False`` is the default because DT1 data is temporal. Training
    code can explicitly choose whether to shuffle already-created training
    windows while keeping validation/test loaders chronological.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def build_train_validation_test_datasets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str = DEFAULT_TARGET_COLUMN,
    sequence_length: int = 12,
    horizon: int = 1,
    drop_nan: bool = True,
) -> tuple[
    GlucoseSequenceDataset,
    GlucoseSequenceDataset,
    GlucoseSequenceDataset,
]:
    """
    Construct PyTorch datasets for chronological train/validation/test splits.
    """
    builder = SequenceBuilder(
        sequence_length=sequence_length,
        horizon=horizon,
        target_column=target_column,
        drop_nan=drop_nan,
    )

    train_dataset = builder.create_dataset(
        train_df,
        feature_columns,
    )
    validation_dataset = builder.create_dataset(
        validation_df,
        feature_columns,
    )
    test_dataset = builder.create_dataset(
        test_df,
        feature_columns,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


def dataset_summary(
    dataset: GlucoseSequenceDataset,
) -> dict[str, Any]:
    """Return useful metadata for debugging and experiment logging."""
    return {
        "samples": len(dataset),
        "sequence_length": dataset.sequence_length,
        "num_features": dataset.num_features,
        "x_shape": tuple(dataset.X.shape),
        "y_shape": tuple(dataset.y.shape),
        "x_dtype": str(dataset.X.dtype),
        "y_dtype": str(dataset.y.dtype),
    }


__all__ = [
    "DEFAULT_TARGET_COLUMN",
    "create_sliding_windows",
    "create_multi_horizon_windows",
    "create_sequence_timestamps",
    "GlucoseSequenceDataset",
    "SequenceBuilder",
    "build_dataloader",
    "build_train_validation_test_datasets",
    "dataset_summary",
]
