from __future__ import annotations

import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ml_project.i18n import Language, t
from ml_project.train import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TrainingResult,
    format_optional_metric,
    validate_dataset,
)


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
    missing_required = (
        int(df[required].isna().sum().sum()) if set(required) <= set(df.columns) else 0
    )
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


def format_metric_rows(metrics: list, lang: Language | str) -> list[dict[str, str]]:
    rows = []
    for metric in metrics:
        rows.append(
            {
                t("model_name", lang): metric.model_name,
                t("mae", lang): f"{metric.mae:.4f}",
                t("rmse", lang): f"{metric.rmse:.4f}",
                t("r2", lang): format_optional_metric(metric.r2),
            }
        )
    return rows


def prepare_downloads(result: TrainingResult) -> tuple[bytes, bytes]:
    model_path = Path(result.model_path)
    metrics_path = model_path.with_name("metrics.json")
    model_bytes = model_path.read_bytes() if model_path.exists() else b""
    metrics_bytes = (
        metrics_path.read_bytes()
        if metrics_path.exists()
        else json.dumps(asdict(result), indent=2).encode("utf-8")
    )
    return model_bytes, metrics_bytes


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
    return (
        correlations.to_frame(name="correlation").reset_index().rename(columns={"index": "feature"})
    )
