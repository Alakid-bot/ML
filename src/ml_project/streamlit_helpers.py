from __future__ import annotations

import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ml_project.i18n import Language, t
from ml_project.train import (
    DEFAULT_MODELS,
    FEATURE_COLUMNS,
    ModelMetrics,
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
                t("model_name", lang): model_label(metric.model_name, lang),
                t("mae", lang): f"{metric.mae:.4f}",
                t("rmse", lang): f"{metric.rmse:.4f}",
                t("r2", lang): format_optional_metric(metric.r2),
            }
        )
    return rows


def selected_model_names(selected: list[str] | tuple[str, ...] | None) -> list[str]:
    if not selected:
        return DEFAULT_MODELS.copy()
    valid = [model for model in selected if model in DEFAULT_MODELS]
    return valid or DEFAULT_MODELS.copy()


def metric_chart_frame(metrics: list[ModelMetrics], lang: Language | str) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        rows.append(
            {
                "model": model_label(metric.model_name, lang),
                "model_key": metric.model_name,
                "MAE": metric.mae,
                "RMSE": metric.rmse,
                "R2": metric.r2,
            }
        )
    return pd.DataFrame(rows)


def metric_rank_frame(metrics: list[ModelMetrics], lang: Language | str) -> pd.DataFrame:
    sorted_metrics = sorted(metrics, key=lambda metric: metric.rmse)
    rows = []
    for rank, metric in enumerate(sorted_metrics, start=1):
        rows.append(
            {
                t("rank", lang): rank,
                t("model_name", lang): model_label(metric.model_name, lang),
                t("rmse", lang): metric.rmse,
                t("mae", lang): metric.mae,
                t("r2", lang): metric.r2,
            }
        )
    return pd.DataFrame(rows)


def selection_rationale(result: TrainingResult, lang: Language | str) -> str:
    selected = next(
        (metric for metric in result.metrics if metric.model_name == result.selected_model),
        None,
    )
    if selected is None:
        return t("selection_rationale_fallback", lang)

    model_name = model_label(result.selected_model, lang)
    if lang == Language.ZH or lang == Language.ZH.value:
        return (
            f"系统选择 {model_name}，因为它在验证集上的 RMSE 最低 "
            f"({selected.rmse:.4f})。RMSE 对较大预测误差更敏感，适合作为本项目的主选择指标；"
            "MAE 用于观察平均误差，R² 用于辅助判断整体解释能力。"
        )

    return (
        f"The system selected {model_name} because it achieved the lowest validation RMSE "
        f"({selected.rmse:.4f}). RMSE is the primary selection metric because it penalizes "
        "larger prediction errors more strongly; MAE shows average absolute error and R² is "
        "used as a supporting goodness-of-fit signal."
    )


def model_label(model_name: str, lang: Language | str) -> str:
    return t(f"model_label_{model_name}", lang)


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
