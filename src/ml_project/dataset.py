"""Dataset loading and validation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_project.schema import FEATURE_COLUMNS, MIN_ROWS, TARGET_COLUMN


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a profiling dataset from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame, *, allow_small_dataset: bool = False) -> None:
    """Validate the dataset required by the first load prediction model."""
    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Dataset contains duplicate columns: {duplicated}")

    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if len(df) < MIN_ROWS and not allow_small_dataset:
        raise ValueError(
            f"Dataset has {len(df)} rows, but at least {MIN_ROWS} rows are required. "
            "Use --allow-small-dataset only for smoke tests."
        )

    required_data = df[required_columns]
    if required_data.isna().any().any():
        columns_with_nulls = required_data.columns[required_data.isna().any()].tolist()
        raise ValueError(f"Dataset has missing values in required columns: {columns_with_nulls}")

    non_numeric_columns = [
        column for column in required_columns if not pd.api.types.is_numeric_dtype(required_data[column])
    ]
    if non_numeric_columns:
        raise ValueError(f"Required columns must be numeric: {non_numeric_columns}")

    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError(f"Target column must contain at least two distinct values: {TARGET_COLUMN}")


__all__ = ["load_dataset", "validate_dataset"]
