"""
data_logger.py

Simulation data logging utilities for the Insulin Digital Twin project.

Purpose
-------
Collect, standardize, validate, and persist outputs produced by SimGlucose
or another compatible simulator. The resulting tabular data is intended to
be consumed by Digital Twin 1 (DT1), Digital Twin 2 (DT2), attack modules,
and evaluation modules.

Expected canonical columns
--------------------------
time, glucose, cgm, insulin, meal

The logger is deliberately simulator-agnostic: it accepts pandas DataFrames,
iterables of dictionaries/records, or objects that can be converted into
DataFrames.

Notes
-----
- No model training or prediction is performed here.
- Missing optional signals are retained as NaN rather than fabricated.
- Time is converted to pandas datetime when possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import pandas as pd


CANONICAL_COLUMNS = ["time", "glucose", "cgm", "insulin", "meal"]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first matching column, case-insensitively."""
    normalized = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    return None


def standardize_dataframe(
    data: Any,
    *,
    keep_extra_columns: bool = True,
) -> pd.DataFrame:
    """
    Convert simulation output into a canonical DataFrame.

    Parameters
    ----------
    data:
        pandas DataFrame, iterable of mappings, or another object accepted
        by pandas.DataFrame().
    keep_extra_columns:
        If True, simulator-specific columns are preserved.

    Returns
    -------
    pandas.DataFrame
        Standardized simulation data.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, Mapping):
        df = pd.DataFrame([data])
    else:
        try:
            df = pd.DataFrame(data)
        except Exception as exc:
            raise TypeError(
                "Simulation output could not be converted to a pandas DataFrame."
            ) from exc

    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

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

    rename_map: dict[Any, str] = {}

    for canonical_name, candidates in aliases.items():
        source_column = _find_column(df, candidates)
        if source_column is not None and source_column != canonical_name:
            rename_map[source_column] = canonical_name

    df = df.rename(columns=rename_map)

    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    # Put canonical columns first while retaining simulator-specific fields.
    if keep_extra_columns:
        extra_columns = [
            col for col in df.columns if col not in CANONICAL_COLUMNS
        ]
        df = df[CANONICAL_COLUMNS + extra_columns]
    else:
        df = df[CANONICAL_COLUMNS]

    # Normalize time if the column contains time-like values.
    if "time" in df.columns:
        parsed_time = pd.to_datetime(df["time"], errors="coerce")
        if parsed_time.notna().any():
            df["time"] = parsed_time

    # Numeric conversion for the signals used by downstream modules.
    for column in ["glucose", "cgm", "insulin", "meal"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.reset_index(drop=True)


def validate_dataframe(
    df: pd.DataFrame,
    *,
    require_time: bool = True,
    require_glucose: bool = False,
) -> dict[str, Any]:
    """
    Validate simulation data without modifying it.

    Returns a dictionary containing validity, row count, missing-value
    information, and basic signal statistics.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required = ["time"] if require_time else []
    if require_glucose:
        required.append("glucose")

    missing_columns = [col for col in required if col not in df.columns]

    missing_values = {
        column: int(df[column].isna().sum())
        for column in CANONICAL_COLUMNS
        if column in df.columns
    }

    result = {
        "valid": len(missing_columns) == 0 and not df.empty,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_required_columns": missing_columns,
        "missing_values": missing_values,
    }

    if "glucose" in df.columns:
        valid_glucose = df["glucose"].dropna()
        if not valid_glucose.empty:
            result["glucose_min"] = float(valid_glucose.min())
            result["glucose_max"] = float(valid_glucose.max())
            result["glucose_mean"] = float(valid_glucose.mean())

    return result


@dataclass
class SimulationDataLogger:
    """
    Collect simulation records and save them to disk.

    The logger can be used either incrementally with ``log()`` or directly
    with a complete DataFrame using ``set_data()``.
    """

    output_dir: str | Path = "data/raw"
    filename: str = "simglucose_results.csv"
    keep_extra_columns: bool = True
    records: list[dict[str, Any]] = field(default_factory=list)
    _data: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def log(self, record: Mapping[str, Any]) -> None:
        """Append one simulation record."""
        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping/dictionary.")
        self.records.append(dict(record))
        self._data = None

    def log_many(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Append multiple simulation records."""
        for record in records:
            self.log(record)

    def set_data(self, data: Any) -> pd.DataFrame:
        """Replace the current logger data with standardized simulation data."""
        self._data = standardize_dataframe(
            data,
            keep_extra_columns=self.keep_extra_columns,
        )
        self.records = []
        return self._data

    def get_data(self) -> pd.DataFrame:
        """Return the current standardized DataFrame."""
        if self._data is None:
            self._data = standardize_dataframe(
                self.records,
                keep_extra_columns=self.keep_extra_columns,
            )
        return self._data.copy()

    def validate(
        self,
        *,
        require_time: bool = True,
        require_glucose: bool = False,
    ) -> dict[str, Any]:
        """Validate the current simulation data."""
        return validate_dataframe(
            self.get_data(),
            require_time=require_time,
            require_glucose=require_glucose,
        )

    def save(
        self,
        filename: Optional[str] = None,
        *,
        index: bool = False,
    ) -> Path:
        """Save standardized simulation data as CSV."""
        df = self.get_data()

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        target = output_dir / (filename or self.filename)
        df.to_csv(target, index=index)

        return target

    def save_parquet(self, filename: str = "simglucose_results.parquet") -> Path:
        """Save standardized simulation data as Parquet."""
        df = self.get_data()

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        target = output_dir / filename
        df.to_parquet(target, index=False)

        return target

    def clear(self) -> None:
        """Clear all collected simulation data."""
        self.records.clear()
        self._data = None


def load_logged_data(path: str | Path) -> pd.DataFrame:
    """
    Load previously saved simulation data.

    CSV and Parquet files are supported.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Simulation data file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        data = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        data = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Use CSV or Parquet."
        )

    return standardize_dataframe(data)


def create_logger(
    output_dir: str | Path = "data/raw",
    filename: str = "simglucose_results.csv",
) -> SimulationDataLogger:
    """Convenience factory for creating a SimulationDataLogger."""
    return SimulationDataLogger(
        output_dir=output_dir,
        filename=filename,
    )


__all__ = [
    "CANONICAL_COLUMNS",
    "SimulationDataLogger",
    "standardize_dataframe",
    "validate_dataframe",
    "load_logged_data",
    "create_logger",
]
