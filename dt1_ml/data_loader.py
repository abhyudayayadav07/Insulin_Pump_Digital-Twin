"""
data_loader.py

Data loading utilities for Digital Twin 1 (DT1).

Purpose
-------
Load simulation/clinical time-series data from CSV, Parquet, JSON, or
pandas DataFrames and prepare it for the DT1 preprocessing pipeline.

The loader is intentionally responsible only for:
    1. locating/loading data,
    2. basic column handling,
    3. sorting by time,
    4. optional time filtering,
    5. basic validation.

Detailed normalization, feature engineering, sequence construction, and
model-specific processing belong in the corresponding DT1 modules.

Expected canonical signals
--------------------------
time, glucose, cgm, insulin, meal

Typical pipeline
----------------
data_loader -> preprocessing -> feature_engineering -> dataset
             -> LSTM/GRU/Transformer -> prediction
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd


DEFAULT_COLUMNS = ["time", "glucose", "cgm", "insulin", "meal"]


class DataLoaderError(Exception):
    """Base exception for DT1 data loading errors."""


def _canonicalize_columns(
    df: pd.DataFrame,
    column_mapping: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """
    Normalize common column-name variations to DT1 canonical names.

    Parameters
    ----------
    df:
        Input DataFrame.
    column_mapping:
        Optional explicit mapping in the form:
        {"source_column": "canonical_column"}.
    """
    result = df.copy()

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

    existing = {
        str(column).strip().lower(): column
        for column in result.columns
    }

    rename_map: dict[Any, str] = {}

    for canonical, candidates in aliases.items():
        if canonical in result.columns:
            continue

        for candidate in candidates:
            source = existing.get(candidate.lower())
            if source is not None:
                rename_map[source] = canonical
                break

    return result.rename(columns=rename_map)


def _convert_types(
    df: pd.DataFrame,
    *,
    time_column: str = "time",
    numeric_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Convert time and signal columns to useful pandas dtypes."""
    result = df.copy()

    if time_column in result.columns:
        parsed_time = pd.to_datetime(result[time_column], errors="coerce")

        # If the data contains a numeric time axis and datetime parsing
        # failed, preserve it as numeric instead of destroying the values.
        if parsed_time.notna().any():
            result[time_column] = parsed_time

    numeric_columns = numeric_columns or [
        "glucose",
        "cgm",
        "insulin",
        "meal",
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def load_dataframe(
    data: Any,
    *,
    column_mapping: Optional[Mapping[str, str]] = None,
    sort_by_time: bool = True,
    drop_duplicate_times: bool = False,
) -> pd.DataFrame:
    """
    Convert supported in-memory data into a standardized DataFrame.

    Parameters
    ----------
    data:
        DataFrame, mapping, iterable of mappings, or pandas-compatible data.
    column_mapping:
        Optional source-to-canonical column mapping.
    sort_by_time:
        Sort by ``time`` when that column exists.
    drop_duplicate_times:
        Remove duplicate timestamps, keeping the first occurrence.

    Returns
    -------
    pandas.DataFrame
        Cleanly loaded DT1 input data.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, Mapping):
        df = pd.DataFrame([data])
    else:
        try:
            df = pd.DataFrame(data)
        except Exception as exc:
            raise DataLoaderError(
                "Could not convert the supplied data to a DataFrame."
            ) from exc

    if df.empty:
        return df.copy()

    df = _canonicalize_columns(
        df,
        column_mapping=column_mapping,
    )
    df = _convert_types(df)

    if "time" in df.columns and sort_by_time:
        df = df.sort_values("time", kind="stable")

    if "time" in df.columns and drop_duplicate_times:
        df = df.drop_duplicates(
            subset=["time"],
            keep="first",
        )

    return df.reset_index(drop=True)


def load_file(
    path: str | Path,
    *,
    column_mapping: Optional[Mapping[str, str]] = None,
    sort_by_time: bool = True,
    drop_duplicate_times: bool = False,
    **read_kwargs: Any,
) -> pd.DataFrame:
    """
    Load DT1 data from a file.

    Supported formats
    -----------------
    CSV       -> ``.csv``
    Parquet   -> ``.parquet`` / ``.pq``
    JSON      -> ``.json``
    Excel     -> ``.xlsx`` / ``.xls``

    Additional keyword arguments are forwarded to the corresponding
    pandas reader.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"DT1 data file not found: {path}")

    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            raw = pd.read_csv(path, **read_kwargs)
        elif suffix in {".parquet", ".pq"}:
            raw = pd.read_parquet(path, **read_kwargs)
        elif suffix == ".json":
            raw = pd.read_json(path, **read_kwargs)
        elif suffix in {".xlsx", ".xls"}:
            raw = pd.read_excel(path, **read_kwargs)
        else:
            raise DataLoaderError(
                f"Unsupported file format: {suffix}. "
                "Supported formats are CSV, Parquet, JSON, and Excel."
            )
    except DataLoaderError:
        raise
    except Exception as exc:
        raise DataLoaderError(
            f"Failed to read DT1 data from '{path}'."
        ) from exc

    return load_dataframe(
        raw,
        column_mapping=column_mapping,
        sort_by_time=sort_by_time,
        drop_duplicate_times=drop_duplicate_times,
    )


def load_simglucose_data(
    path: str | Path,
    *,
    required_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Convenience loader for previously logged SimGlucose output.

    This function expects data produced by the simulator/data_logger layer
    and checks that the requested DT1 signals are present.
    """
    required = list(
        required_columns
        if required_columns is not None
        else ["time", "glucose"]
    )

    df = load_file(path)

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataLoaderError(
            "SimGlucose data is missing required DT1 columns: "
            + ", ".join(missing)
        )

    return df


def validate_data(
    df: pd.DataFrame,
    *,
    required_columns: Sequence[str] = ("time", "glucose"),
    min_rows: int = 1,
    allow_missing_values: bool = True,
) -> dict[str, Any]:
    """
    Validate a loaded DT1 dataset.

    Returns a dictionary rather than raising for ordinary data-quality
    problems, allowing the training pipeline to decide how strict it should be.
    """
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    missing_values = {
        column: int(df[column].isna().sum())
        for column in required_columns
        if column in df.columns
    }

    valid = (
        len(df) >= min_rows
        and len(missing_columns) == 0
        and (
            allow_missing_values
            or all(value == 0 for value in missing_values.values())
        )
    )

    return {
        "valid": bool(valid),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_required_columns": missing_columns,
        "missing_values": missing_values,
        "allow_missing_values": allow_missing_values,
        "min_rows": min_rows,
    }


def select_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Select DT1 input columns while optionally checking their existence."""
    missing = [column for column in columns if column not in df.columns]

    if strict and missing:
        raise KeyError(
            "Requested DT1 columns are not present: "
            + ", ".join(missing)
        )

    available = [column for column in columns if column in df.columns]
    return df[available].copy()


def filter_time_range(
    df: pd.DataFrame,
    *,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    time_column: str = "time",
) -> pd.DataFrame:
    """
    Filter data to an optional inclusive time interval.
    """
    if time_column not in df.columns:
        raise KeyError(
            f"Time column '{time_column}' is not present in the dataset."
        )

    result = df.copy()

    if start is not None:
        start_value = pd.to_datetime(start, errors="coerce")
        if pd.isna(start_value):
            raise ValueError(f"Invalid start time: {start}")
        result = result[result[time_column] >= start_value]

    if end is not None:
        end_value = pd.to_datetime(end, errors="coerce")
        if pd.isna(end_value):
            raise ValueError(f"Invalid end time: {end}")
        result = result[result[time_column] <= end_value]

    return result.reset_index(drop=True)


def train_validation_test_split(
    df: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split time-series data into train/validation/test sets.

    Unlike a random split, this preserves temporal ordering, which is
    important for DT1 glucose forecasting and drift/anomaly analysis.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1."
        )

    ordered = df.copy()

    if "time" in ordered.columns:
        ordered = ordered.sort_values("time", kind="stable")

    n = len(ordered)
    train_end = int(n * train_ratio)
    validation_end = int(n * (train_ratio + validation_ratio))

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def load_and_split(
    path: str | Path,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    **load_kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a DT1 dataset and perform a chronological train/validation/test split."""
    df = load_file(path, **load_kwargs)
    return train_validation_test_split(
        df,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )


__all__ = [
    "DEFAULT_COLUMNS",
    "DataLoaderError",
    "load_dataframe",
    "load_file",
    "load_simglucose_data",
    "validate_data",
    "select_columns",
    "filter_time_range",
    "train_validation_test_split",
    "load_and_split",
]
