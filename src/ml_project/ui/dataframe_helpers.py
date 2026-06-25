"""Dataframe parsing, validation, and summary helpers for the Streamlit UI."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

from ml_project.dataset import validate_dataset
from ml_project.schema import FEATURE_COLUMNS, TARGET_COLUMN


def parse_csv(file: bytes | io.BytesIO | io.StringIO | None) -> tuple[pd.DataFrame | None, str]:
    if file is None:
        return None, ""
    try:
        if isinstance(file, bytes):
            return pd.read_csv(io.BytesIO(file)), ""
        if isinstance(file, io.BytesIO):
            return pd.read_csv(file), ""
        if isinstance(file, io.StringIO):
            return pd.read_csv(file), ""
        return pd.read_csv(file), ""
    except Exception as exc:
        return None, str(exc)


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    required = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_required = int(df[required].isna().sum().sum()) if set(required) <= set(df.columns) else 0
    present_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    target_present = TARGET_COLUMN in df.columns
    target_unique = int(df[TARGET_COLUMN].nunique()) if target_present else 0

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "feature_count": len(present_features),
        "features": present_features,
        "target_present": target_present,
        "target_unique": target_unique,
        "missing_required": missing_required,
        "missing_total": int(df.isna().sum().sum()),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
    }


def validate_frontend_dataset(df: pd.DataFrame, *, allow_small_dataset: bool) -> tuple[bool, str]:
    try:
        validate_dataset(df, allow_small_dataset=allow_small_dataset)
        return True, ""
    except ValueError as exc:
        return False, str(exc)


def target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN not in df.columns:
        return pd.DataFrame()
    return df[[TARGET_COLUMN]].copy()


def feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN not in df.columns:
        return pd.DataFrame()
    present_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    if not present_features:
        return pd.DataFrame()
    numeric_df = df[present_features + [TARGET_COLUMN]].select_dtypes(include="number")
    if numeric_df.empty or TARGET_COLUMN not in numeric_df.columns:
        return pd.DataFrame()
    correlations = numeric_df.corr()[TARGET_COLUMN].drop(TARGET_COLUMN, errors="ignore")
    return correlations.to_frame(name="correlation").reset_index().rename(columns={"index": "feature"})


__all__ = [
    "feature_correlations",
    "parse_csv",
    "summarize_dataset",
    "target_distribution",
    "validate_frontend_dataset",
]
