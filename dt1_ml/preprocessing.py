"""
preprocessing.py

Preprocessing utilities for Digital Twin 1 (DT1).

Purpose
-------
Prepare glucose/CGM time-series data for ML/DL forecasting models.

Pipeline responsibilities
-------------------------
1. Standardize required columns.
2. Sort observations chronologically.
3. Handle missing values.
4. Optionally resample to a fixed interval.
5. Clip or filter physically invalid values.
6. Scale numerical features using statistics learned from training data.
7. Apply the same fitted transformation to validation/test/inference data.

Important
---------
The scaler must be fitted ONLY on the training split to avoid temporal
data leakage. The fitted scaler can then be reused for validation, test,
and deployment/inference data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


DEFAULT_NUMERIC_COLUMNS = ["glucose", "cgm", "insulin", "meal"]


def ensure_dataframe(data: Any) -> pd.DataFrame:
    """Convert supported input data into a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, Mapping):
        return pd.DataFrame([data])

    try:
        return pd.DataFrame(data)
    except Exception as exc:
        raise TypeError(
            "Input data could not be converted to a pandas DataFrame."
        ) from exc


def standardize_columns(
    df: pd.DataFrame,
    *,
    column_mapping: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """
    Normalize common simulator/clinical column names.

    Example
    -------
    ``blood_glucose`` -> ``glucose``
    ``sensor_glucose`` -> ``cgm``
    ``carbs`` -> ``meal``
    """
    result = ensure_dataframe(df)

    if column_mapping:
        result = result.rename(columns=dict(column_mapping))

    aliases = {
        "time": [
            "time",
            "timestamp",
            "datetime",
            "date",
            "t",
        ],
        "glucose": [
            "glucose",
            "blood_glucose",
            "blood glucose",
            "bg",
            "glucose_level",
        ],
        "cgm": [
            "cgm",
            "cgm_glucose",
            "sensor_glucose",
            "sensor glucose",
            "sg",
        ],
        "insulin": [
            "insulin",
            "insulin_delivery",
            "insulin delivered",
            "total_insulin",
        ],
        "meal": [
            "meal",
            "carbs",
            "carbohydrates",
            "cho",
            "meal_carbs",
        ],
    }

    lookup = {
        str(column).strip().lower(): column
        for column in result.columns
    }

    rename_map: dict[Any, str] = {}

    for canonical, candidates in aliases.items():
        if canonical in result.columns:
            continue

        for candidate in candidates:
            source = lookup.get(candidate.lower())
            if source is not None:
                rename_map[source] = canonical
                break

    result = result.rename(columns=rename_map)

    return result


def sort_by_time(
    df: pd.DataFrame,
    *,
    time_column: str = "time",
) -> pd.DataFrame:
    """Sort data chronologically when a time column is available."""
    result = df.copy()

    if time_column not in result.columns:
        return result.reset_index(drop=True)

    parsed = pd.to_datetime(result[time_column], errors="coerce")

    if parsed.notna().any():
        result[time_column] = parsed

    return result.sort_values(
        time_column,
        kind="stable",
    ).reset_index(drop=True)


def convert_numeric_columns(
    df: pd.DataFrame,
    columns: Sequence[str] = DEFAULT_NUMERIC_COLUMNS,
) -> pd.DataFrame:
    """Convert selected signal columns to numeric values."""
    result = df.copy()

    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def remove_duplicate_timestamps(
    df: pd.DataFrame,
    *,
    time_column: str = "time",
    keep: str = "first",
) -> pd.DataFrame:
    """Remove duplicate timestamps."""
    result = df.copy()

    if time_column not in result.columns:
        return result

    return result.drop_duplicates(
        subset=[time_column],
        keep=keep,
    ).reset_index(drop=True)


def handle_missing_values(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] = DEFAULT_NUMERIC_COLUMNS,
    method: str = "interpolate",
    limit: Optional[int] = None,
    fill_remaining: Optional[float] = None,
) -> pd.DataFrame:
    """
    Handle missing values in selected numerical signals.

    Methods
    -------
    ``interpolate``
        Linear interpolation followed by forward/backward filling.

    ``ffill``
        Forward-fill followed by backward-fill.

    ``bfill``
        Backward-fill followed by forward-fill.

    ``median``
        Fill using each column's median.

    ``drop``
        Drop rows containing missing values in selected columns.

    ``none``
        Leave missing values unchanged.

    ``fill``
        Fill with ``fill_remaining``.
    """
    result = df.copy()

    available = [
        column for column in columns
        if column in result.columns
    ]

    if not available or method == "none":
        return result

    if method == "interpolate":
        result[available] = result[available].interpolate(
            method="linear",
            limit=limit,
            limit_direction="both",
        )
        result[available] = result[available].ffill().bfill()

    elif method == "ffill":
        result[available] = result[available].ffill(limit=limit).bfill()

    elif method == "bfill":
        result[available] = result[available].bfill(limit=limit).ffill()

    elif method == "median":
        for column in available:
            median = result[column].median()
            result[column] = result[column].fillna(median)

    elif method == "drop":
        result = result.dropna(subset=available)

    elif method == "fill":
        if fill_remaining is None:
            raise ValueError(
                "fill_remaining must be provided when method='fill'."
            )
        result[available] = result[available].fillna(fill_remaining)

    else:
        raise ValueError(
            f"Unknown missing-value method: {method}. "
            "Use interpolate, ffill, bfill, median, drop, fill, or none."
        )

    return result.reset_index(drop=True)


def resample_time_series(
    df: pd.DataFrame,
    *,
    interval: str = "5min",
    time_column: str = "time",
    aggregation: str = "mean",
) -> pd.DataFrame:
    """
    Resample a time-series to a fixed interval.

    This is useful when simulator/clinical observations are not already
    aligned to the desired DT1 timestep.

    Parameters
    ----------
    interval:
        Pandas frequency such as ``5min`` or ``15min``.

    aggregation:
        ``mean``, ``median``, ``sum``, ``first``, or ``last``.
    """
    if time_column not in df.columns:
        raise KeyError(
            f"Time column '{time_column}' is required for resampling."
        )

    if aggregation not in {
        "mean",
        "median",
        "sum",
        "first",
        "last",
    }:
        raise ValueError(
            "aggregation must be one of: mean, median, sum, first, last."
        )

    result = df.copy()
    result[time_column] = pd.to_datetime(
        result[time_column],
        errors="coerce",
    )

    result = result.dropna(subset=[time_column])
    result = result.sort_values(time_column)
    result = result.set_index(time_column)

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    if aggregation == "mean":
        result = result[numeric_columns].resample(interval).mean()
    elif aggregation == "median":
        result = result[numeric_columns].resample(interval).median()
    elif aggregation == "sum":
        result = result[numeric_columns].resample(interval).sum()
    elif aggregation == "first":
        result = result[numeric_columns].resample(interval).first()
    else:
        result = result[numeric_columns].resample(interval).last()

    return result.reset_index()


def clip_physiological_values(
    df: pd.DataFrame,
    *,
    glucose_range: tuple[float, float] = (20.0, 600.0),
    cgm_range: tuple[float, float] = (20.0, 600.0),
    insulin_min: float = 0.0,
    meal_min: float = 0.0,
) -> pd.DataFrame:
    """
    Remove obviously invalid physiological/control values.

    Values outside the configured ranges are replaced by NaN rather than
    silently clipped to the boundary. Missing-value handling can then decide
    how they should be treated.
    """
    result = df.copy()

    if "glucose" in result.columns:
        low, high = glucose_range
        mask = (
            (result["glucose"] < low)
            | (result["glucose"] > high)
        )
        result.loc[mask, "glucose"] = np.nan

    if "cgm" in result.columns:
        low, high = cgm_range
        mask = (
            (result["cgm"] < low)
            | (result["cgm"] > high)
        )
        result.loc[mask, "cgm"] = np.nan

    if "insulin" in result.columns:
        result.loc[
            result["insulin"] < insulin_min,
            "insulin",
        ] = np.nan

    if "meal" in result.columns:
        result.loc[
            result["meal"] < meal_min,
            "meal",
        ] = np.nan

    return result


def add_time_features(
    df: pd.DataFrame,
    *,
    time_column: str = "time",
    include_cyclical: bool = True,
) -> pd.DataFrame:
    """
    Add useful calendar/time-of-day features.

    Features
    --------
    hour
    minute
    day_of_week

    If ``include_cyclical=True``:
    hour_sin, hour_cos
    day_of_week_sin, day_of_week_cos
    """
    result = df.copy()

    if time_column not in result.columns:
        return result

    timestamp = pd.to_datetime(
        result[time_column],
        errors="coerce",
    )

    result["hour"] = timestamp.dt.hour
    result["minute"] = timestamp.dt.minute
    result["day_of_week"] = timestamp.dt.dayofweek

    if include_cyclical:
        hour_decimal = (
            result["hour"].fillna(0)
            + result["minute"].fillna(0) / 60.0
        )

        result["hour_sin"] = np.sin(
            2.0 * np.pi * hour_decimal / 24.0
        )
        result["hour_cos"] = np.cos(
            2.0 * np.pi * hour_decimal / 24.0
        )

        result["day_of_week_sin"] = np.sin(
            2.0 * np.pi * result["day_of_week"].fillna(0) / 7.0
        )
        result["day_of_week_cos"] = np.cos(
            2.0 * np.pi * result["day_of_week"].fillna(0) / 7.0
        )

    return result


@dataclass
class DT1Scaler:
    """
    Fit and apply a numerical scaler for DT1.

    Supported methods:
        - standard
        - minmax
        - robust

    The object stores the fitted sklearn scaler and feature names so the
    exact same transformation can be applied during inference.
    """

    method: str = "standard"
    feature_columns: list[str] = field(default_factory=list)
    scaler: Any = field(default=None, init=False, repr=False)

    def _create_scaler(self) -> Any:
        if self.method == "standard":
            return StandardScaler()
        if self.method == "minmax":
            return MinMaxScaler()
        if self.method == "robust":
            return RobustScaler()

        raise ValueError(
            f"Unknown scaler method: {self.method}. "
            "Use standard, minmax, or robust."
        )

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> "DT1Scaler":
        """Fit the scaler using training data only."""
        columns = list(feature_columns)

        missing = [
            column for column in columns
            if column not in df.columns
        ]
        if missing:
            raise KeyError(
                "Cannot fit scaler; missing columns: "
                + ", ".join(missing)
            )

        values = df[columns].astype(float)

        if values.isna().any().any():
            raise ValueError(
                "Training data contains NaN values in scaler features. "
                "Handle missing values before fitting."
            )

        self.feature_columns = columns
        self.scaler = self._create_scaler()
        self.scaler.fit(values)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform selected features using the fitted scaler."""
        if self.scaler is None:
            raise RuntimeError(
                "Scaler has not been fitted. Call fit() first."
            )

        result = df.copy()

        missing = [
            column for column in self.feature_columns
            if column not in result.columns
        ]
        if missing:
            raise KeyError(
                "Cannot transform data; missing columns: "
                + ", ".join(missing)
            )

        result[self.feature_columns] = self.scaler.transform(
            result[self.feature_columns].astype(float)
        )

        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> pd.DataFrame:
        """Fit on training data and immediately transform it."""
        self.fit(df, feature_columns)
        return self.transform(df)

    def inverse_transform(
        self,
        values: np.ndarray | pd.DataFrame,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Convert scaled values back to original units.

        Useful for converting predicted glucose values back to mg/dL.
        """
        if self.scaler is None:
            raise RuntimeError(
                "Scaler has not been fitted. Call fit() first."
            )

        if isinstance(values, pd.DataFrame):
            selected_columns = list(columns or values.columns)
            array = values[selected_columns].to_numpy(dtype=float)
        else:
            array = np.asarray(values, dtype=float)

        return self.scaler.inverse_transform(array)


def preprocess(
    df: pd.DataFrame,
    *,
    numeric_columns: Sequence[str] = DEFAULT_NUMERIC_COLUMNS,
    missing_method: str = "interpolate",
    add_time_features_flag: bool = False,
    remove_duplicates: bool = True,
    physiological_filter: bool = True,
) -> pd.DataFrame:
    """
    Run the standard DT1 preprocessing pipeline.

    This function does not fit a scaler. Scaling should be performed
    separately after the chronological train/validation/test split.
    """
    result = standardize_columns(df)
    result = sort_by_time(result)
    result = convert_numeric_columns(result, numeric_columns)

    if remove_duplicates:
        result = remove_duplicate_timestamps(result)

    if physiological_filter:
        result = clip_physiological_values(result)

    result = handle_missing_values(
        result,
        columns=numeric_columns,
        method=missing_method,
    )

    if add_time_features_flag:
        result = add_time_features(result)

    return result


def prepare_train_validation_test(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    scaler_method: str = "standard",
    preprocessing_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    DT1Scaler,
]:
    """
    Preprocess three chronological splits without data leakage.

    The scaler is fitted exclusively on ``train_df`` and then applied to
    validation and test data.
    """
    kwargs = dict(preprocessing_kwargs or {})

    train = preprocess(train_df, **kwargs)
    validation = preprocess(validation_df, **kwargs)
    test = preprocess(test_df, **kwargs)

    scaler = DT1Scaler(method=scaler_method)

    train = scaler.fit_transform(
        train,
        feature_columns,
    )
    validation = scaler.transform(validation)
    test = scaler.transform(test)

    return train, validation, test, scaler


__all__ = [
    "DEFAULT_NUMERIC_COLUMNS",
    "DT1Scaler",
    "ensure_dataframe",
    "standardize_columns",
    "sort_by_time",
    "convert_numeric_columns",
    "remove_duplicate_timestamps",
    "handle_missing_values",
    "resample_time_series",
    "clip_physiological_values",
    "add_time_features",
    "preprocess",
    "prepare_train_validation_test",
]
