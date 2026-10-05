"""
feature_engineering.py

Feature engineering utilities for Digital Twin 1 (DT1).

Purpose
-------
Convert preprocessed insulin-pump / glucose time-series data into informative
temporal features for ML/DL glucose prediction.

The module focuses on features that describe:
    - recent glucose dynamics,
    - CGM dynamics,
    - insulin exposure,
    - meal/carbohydrate exposure,
    - short-term temporal context.

These features are generated causally from current and past observations.
Future values are never used to construct an input feature.

Typical flow
------------
data_loader.py
    -> preprocessing.py
    -> feature_engineering.py
    -> dataset.py
    -> LSTM / GRU / Transformer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_BASE_FEATURES = [
    "glucose",
    "cgm",
    "insulin",
    "meal",
]

DEFAULT_GLUC0SE_LAGS = [1, 2, 3, 6, 12]
DEFAULT_CGM_LAGS = [1, 2, 3, 6, 12]
DEFAULT_INSULIN_WINDOWS = [3, 6, 12]
DEFAULT_MEAL_WINDOWS = [3, 6, 12]


def _ensure_sorted(
    df: pd.DataFrame,
    time_column: str = "time",
) -> pd.DataFrame:
    """Return a copy sorted chronologically when a time column exists."""
    result = df.copy()

    if time_column in result.columns:
        parsed = pd.to_datetime(result[time_column], errors="coerce")
        if parsed.notna().any():
            result[time_column] = parsed

        result = result.sort_values(
            time_column,
            kind="stable",
        )

    return result.reset_index(drop=True)


def add_lag_features(
    df: pd.DataFrame,
    *,
    column: str = "glucose",
    lags: Sequence[int] = DEFAULT_GLUC0SE_LAGS,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add historical lag features.

    For example, ``glucose_lag_1`` represents the previous observation and
    ``glucose_lag_12`` represents the value 12 samples in the past.

    No future information is used.
    """
    result = _ensure_sorted(df)

    if column not in result.columns:
        return result

    prefix = prefix or column

    for lag in lags:
        if lag <= 0:
            raise ValueError("Lag values must be positive integers.")

        result[f"{prefix}_lag_{lag}"] = result[column].shift(lag)

    return result


def add_difference_features(
    df: pd.DataFrame,
    *,
    column: str = "glucose",
    differences: Sequence[int] = (1, 3, 6),
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add temporal difference features.

    ``glucose_diff_1 = glucose(t) - glucose(t-1)``

    These features provide a simple representation of glucose direction and
    short-term change.
    """
    result = _ensure_sorted(df)

    if column not in result.columns:
        return result

    prefix = prefix or column

    for period in differences:
        if period <= 0:
            raise ValueError(
                "Difference periods must be positive integers."
            )

        result[f"{prefix}_diff_{period}"] = (
            result[column] - result[column].shift(period)
        )

    return result


def add_rate_of_change_features(
    df: pd.DataFrame,
    *,
    column: str = "glucose",
    periods: Sequence[int] = (1, 3, 6),
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add rate-of-change features.

    For equally sampled data this is the change per sample. The actual
    physical rate can be obtained by dividing by the sampling interval.
    """
    result = _ensure_sorted(df)

    if column not in result.columns:
        return result

    prefix = prefix or column

    for period in periods:
        if period <= 0:
            raise ValueError(
                "Rate-of-change periods must be positive integers."
            )

        result[f"{prefix}_roc_{period}"] = (
            result[column] - result[column].shift(period)
        ) / float(period)

    return result


def add_rolling_statistics(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] = ("glucose", "cgm"),
    windows: Sequence[int] = (3, 6, 12),
    statistics: Sequence[str] = ("mean", "std", "min", "max"),
) -> pd.DataFrame:
    """
    Add rolling statistics based only on current and past observations.

    ``min_periods=1`` allows the feature generator to work at the beginning
    of a sequence. Sequence construction later decides how much warm-up
    history is required.
    """
    result = _ensure_sorted(df)

    valid_statistics = {"mean", "std", "min", "max", "median"}

    invalid = set(statistics) - valid_statistics
    if invalid:
        raise ValueError(
            f"Unsupported rolling statistics: {sorted(invalid)}"
        )

    for column in columns:
        if column not in result.columns:
            continue

        series = pd.to_numeric(result[column], errors="coerce")

        for window in windows:
            if window <= 0:
                raise ValueError("Rolling windows must be positive.")

            rolling = series.rolling(
                window=window,
                min_periods=1,
            )

            for statistic in statistics:
                result[
                    f"{column}_rolling_{statistic}_{window}"
                ] = getattr(rolling, statistic)()

    return result


def add_exponential_features(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] = ("glucose", "cgm"),
    spans: Sequence[int] = (3, 6, 12),
) -> pd.DataFrame:
    """
    Add exponentially weighted moving averages.

    EWM features emphasize recent observations while retaining historical
    information.
    """
    result = _ensure_sorted(df)

    for column in columns:
        if column not in result.columns:
            continue

        series = pd.to_numeric(result[column], errors="coerce")

        for span in spans:
            if span <= 0:
                raise ValueError("EWM spans must be positive.")

            result[
                f"{column}_ewm_{span}"
            ] = series.ewm(
                span=span,
                adjust=False,
                min_periods=1,
            ).mean()

    return result


def add_insulin_exposure_features(
    df: pd.DataFrame,
    *,
    column: str = "insulin",
    windows: Sequence[int] = DEFAULT_INSULIN_WINDOWS,
) -> pd.DataFrame:
    """
    Add insulin exposure features.

    Features include:
        - recent insulin sum,
        - recent insulin mean,
        - exponentially weighted insulin exposure.

    The features are useful for representing recent insulin delivery rather
    than relying only on the instantaneous insulin value.
    """
    result = _ensure_sorted(df)

    if column not in result.columns:
        return result

    series = pd.to_numeric(result[column], errors="coerce")

    for window in windows:
        if window <= 0:
            raise ValueError("Insulin windows must be positive.")

        result[
            f"insulin_sum_{window}"
        ] = series.rolling(
            window=window,
            min_periods=1,
        ).sum()

        result[
            f"insulin_mean_{window}"
        ] = series.rolling(
            window=window,
            min_periods=1,
        ).mean()

        result[
            f"insulin_ewm_{window}"
        ] = series.ewm(
            span=window,
            adjust=False,
            min_periods=1,
        ).mean()

    return result


def add_meal_exposure_features(
    df: pd.DataFrame,
    *,
    column: str = "meal",
    windows: Sequence[int] = DEFAULT_MEAL_WINDOWS,
) -> pd.DataFrame:
    """
    Add recent meal/carbohydrate exposure features.

    ``meal_sum_N`` represents the cumulative meal input over the latest N
    samples, which is more informative for forecasting than the instantaneous
    meal value alone.
    """
    result = _ensure_sorted(df)

    if column not in result.columns:
        return result

    series = pd.to_numeric(result[column], errors="coerce")

    for window in windows:
        if window <= 0:
            raise ValueError("Meal windows must be positive.")

        result[
            f"meal_sum_{window}"
        ] = series.rolling(
            window=window,
            min_periods=1,
        ).sum()

        result[
            f"meal_mean_{window}"
        ] = series.rolling(
            window=window,
            min_periods=1,
        ).mean()

    return result


def add_glucose_cgm_consistency_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add features describing the relationship between CGM and glucose.

    If both signals exist:
        cgm_glucose_diff = cgm - glucose
        cgm_glucose_abs_diff = |cgm - glucose|

    These can also become useful signals for anomaly/residual analysis.
    """
    result = _ensure_sorted(df)

    if "glucose" not in result.columns or "cgm" not in result.columns:
        return result

    result["cgm_glucose_diff"] = (
        result["cgm"] - result["glucose"]
    )
    result["cgm_glucose_abs_diff"] = (
        result["cgm"] - result["glucose"]
    ).abs()

    return result


def add_temporal_context_features(
    df: pd.DataFrame,
    *,
    time_column: str = "time",
) -> pd.DataFrame:
    """
    Add time-of-day and day-of-week context.

    Cyclical encoding avoids treating 23:00 and 00:00 as far apart.
    """
    result = _ensure_sorted(df, time_column=time_column)

    if time_column not in result.columns:
        return result

    timestamp = pd.to_datetime(
        result[time_column],
        errors="coerce",
    )

    hour = timestamp.dt.hour.fillna(0)
    minute = timestamp.dt.minute.fillna(0)
    day = timestamp.dt.dayofweek.fillna(0)

    hour_decimal = hour + minute / 60.0

    result["hour_sin"] = np.sin(
        2.0 * np.pi * hour_decimal / 24.0
    )
    result["hour_cos"] = np.cos(
        2.0 * np.pi * hour_decimal / 24.0
    )
    result["day_of_week_sin"] = np.sin(
        2.0 * np.pi * day / 7.0
    )
    result["day_of_week_cos"] = np.cos(
        2.0 * np.pi * day / 7.0
    )

    return result


def add_sample_interval_feature(
    df: pd.DataFrame,
    *,
    time_column: str = "time",
    unit: str = "minutes",
) -> pd.DataFrame:
    """
    Add the elapsed time since the previous observation.

    This is useful when the dataset contains irregular sampling intervals.
    """
    result = _ensure_sorted(df, time_column=time_column)

    if time_column not in result.columns:
        return result

    timestamp = pd.to_datetime(
        result[time_column],
        errors="coerce",
    )

    delta = timestamp.diff()

    if unit == "seconds":
        result["delta_time"] = delta.dt.total_seconds()
    elif unit == "minutes":
        result["delta_time"] = delta.dt.total_seconds() / 60.0
    elif unit == "hours":
        result["delta_time"] = delta.dt.total_seconds() / 3600.0
    else:
        raise ValueError(
            "unit must be seconds, minutes, or hours."
        )

    return result


def add_glucose_derivative_features(
    df: pd.DataFrame,
    *,
    glucose_column: str = "glucose",
    time_column: str = "time",
) -> pd.DataFrame:
    """
    Estimate first and second glucose derivatives.

    If timestamps are available, the derivatives are expressed per minute.
    Otherwise they are computed per sample.
    """
    result = _ensure_sorted(df, time_column=time_column)

    if glucose_column not in result.columns:
        return result

    glucose = pd.to_numeric(
        result[glucose_column],
        errors="coerce",
    )

    if time_column in result.columns:
        timestamps = pd.to_datetime(
            result[time_column],
            errors="coerce",
        )
        delta_minutes = (
            timestamps.diff().dt.total_seconds() / 60.0
        )

        safe_delta = delta_minutes.replace(
            [np.inf, -np.inf, 0],
            np.nan,
        )

        first_derivative = glucose.diff() / safe_delta
        second_derivative = first_derivative.diff() / safe_delta

        result["glucose_rate_per_min"] = first_derivative
        result["glucose_acceleration_per_min"] = second_derivative
    else:
        result["glucose_rate_per_sample"] = glucose.diff()
        result["glucose_acceleration_per_sample"] = (
            glucose.diff().diff()
        )

    return result


def build_features(
    df: pd.DataFrame,
    *,
    include_base_features: bool = True,
    include_glucose_lags: bool = True,
    glucose_lags: Sequence[int] = DEFAULT_GLUC0SE_LAGS,
    include_cgm_lags: bool = True,
    cgm_lags: Sequence[int] = DEFAULT_CGM_LAGS,
    include_differences: bool = True,
    difference_periods: Sequence[int] = (1, 3, 6),
    include_rate_of_change: bool = True,
    rate_periods: Sequence[int] = (1, 3, 6),
    include_rolling: bool = True,
    rolling_windows: Sequence[int] = (3, 6, 12),
    include_ewm: bool = True,
    ewm_spans: Sequence[int] = (3, 6, 12),
    include_insulin_exposure: bool = True,
    insulin_windows: Sequence[int] = DEFAULT_INSULIN_WINDOWS,
    include_meal_exposure: bool = True,
    meal_windows: Sequence[int] = DEFAULT_MEAL_WINDOWS,
    include_cgm_consistency: bool = True,
    include_temporal_context: bool = False,
    include_delta_time: bool = False,
    include_derivatives: bool = False,
) -> pd.DataFrame:
    """
    Build the complete DT1 feature set.

    The defaults are intentionally suitable for a first glucose forecasting
    model. Individual feature groups can be disabled when running ablation
    experiments or when the available simulator signals are limited.
    """
    result = _ensure_sorted(df)

    if include_glucose_lags:
        result = add_lag_features(
            result,
            column="glucose",
            lags=glucose_lags,
        )

    if include_cgm_lags:
        result = add_lag_features(
            result,
            column="cgm",
            lags=cgm_lags,
        )

    if include_differences:
        result = add_difference_features(
            result,
            column="glucose",
            differences=difference_periods,
        )

    if include_rate_of_change:
        result = add_rate_of_change_features(
            result,
            column="glucose",
            periods=rate_periods,
        )

    if include_rolling:
        result = add_rolling_statistics(
            result,
            columns=("glucose", "cgm"),
            windows=rolling_windows,
        )

    if include_ewm:
        result = add_exponential_features(
            result,
            columns=("glucose", "cgm"),
            spans=ewm_spans,
        )

    if include_insulin_exposure:
        result = add_insulin_exposure_features(
            result,
            windows=insulin_windows,
        )

    if include_meal_exposure:
        result = add_meal_exposure_features(
            result,
            windows=meal_windows,
        )

    if include_cgm_consistency:
        result = add_glucose_cgm_consistency_features(result)

    if include_temporal_context:
        result = add_temporal_context_features(result)

    if include_delta_time:
        result = add_sample_interval_feature(result)

    if include_derivatives:
        result = add_glucose_derivative_features(result)

    if not include_base_features:
        engineered_columns = [
            column
            for column in result.columns
            if column not in {
                "time",
                "glucose",
                "cgm",
                "insulin",
                "meal",
            }
        ]
        result = result[engineered_columns]

    return result


def get_feature_columns(
    df: pd.DataFrame,
    *,
    exclude: Optional[Iterable[str]] = None,
    include: Optional[Sequence[str]] = None,
) -> list[str]:
    """
    Return model-ready feature column names.

    ``time`` is excluded by default because neural networks should receive
    numerical temporal encodings rather than raw datetime objects.
    """
    if include is not None:
        missing = [
            column for column in include
            if column not in df.columns
        ]
        if missing:
            raise KeyError(
                "Requested feature columns are missing: "
                + ", ".join(missing)
            )
        return list(include)

    excluded = set(exclude or [])
    excluded.update({"time"})

    return [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def remove_feature_warmup_rows(
    df: pd.DataFrame,
    *,
    warmup: int,
) -> pd.DataFrame:
    """
    Remove the initial rows affected by lag/rolling warm-up.

    Example:
        warmup=12 removes the first 12 observations after features are built.
    """
    if warmup < 0:
        raise ValueError("warmup must be non-negative.")

    if warmup == 0:
        return df.copy().reset_index(drop=True)

    return df.iloc[warmup:].reset_index(drop=True)


@dataclass
class FeatureEngineer:
    """
    Reusable DT1 feature-engineering configuration.

    Calling ``transform()`` applies the same deterministic feature recipe to
    new data, which is useful for validation, testing, and deployment.
    """

    glucose_lags: list[int] = field(
        default_factory=lambda: list(DEFAULT_GLUC0SE_LAGS)
    )
    cgm_lags: list[int] = field(
        default_factory=lambda: list(DEFAULT_CGM_LAGS)
    )
    difference_periods: list[int] = field(
        default_factory=lambda: [1, 3, 6]
    )
    rate_periods: list[int] = field(
        default_factory=lambda: [1, 3, 6]
    )
    rolling_windows: list[int] = field(
        default_factory=lambda: [3, 6, 12]
    )
    ewm_spans: list[int] = field(
        default_factory=lambda: [3, 6, 12]
    )
    insulin_windows: list[int] = field(
        default_factory=lambda: list(DEFAULT_INSULIN_WINDOWS)
    )
    meal_windows: list[int] = field(
        default_factory=lambda: list(DEFAULT_MEAL_WINDOWS)
    )
    include_temporal_context: bool = False
    include_delta_time: bool = False
    include_derivatives: bool = False

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured feature-engineering pipeline."""
        return build_features(
            df,
            glucose_lags=self.glucose_lags,
            cgm_lags=self.cgm_lags,
            difference_periods=self.difference_periods,
            rate_periods=self.rate_periods,
            rolling_windows=self.rolling_windows,
            ewm_spans=self.ewm_spans,
            insulin_windows=self.insulin_windows,
            meal_windows=self.meal_windows,
            include_temporal_context=self.include_temporal_context,
            include_delta_time=self.include_delta_time,
            include_derivatives=self.include_derivatives,
        )


__all__ = [
    "DEFAULT_BASE_FEATURES",
    "DEFAULT_GLUC0SE_LAGS",
    "DEFAULT_CGM_LAGS",
    "DEFAULT_INSULIN_WINDOWS",
    "DEFAULT_MEAL_WINDOWS",
    "add_lag_features",
    "add_difference_features",
    "add_rate_of_change_features",
    "add_rolling_statistics",
    "add_exponential_features",
    "add_insulin_exposure_features",
    "add_meal_exposure_features",
    "add_glucose_cgm_consistency_features",
    "add_temporal_context_features",
    "add_sample_interval_feature",
    "add_glucose_derivative_features",
    "build_features",
    "get_feature_columns",
    "remove_feature_warmup_rows",
    "FeatureEngineer",
]
