"""Formatting helpers for Streamlit metric and diagnostic displays."""

from __future__ import annotations

import pandas as pd

from ml_project.i18n import Language, t
from ml_project.pipeline import format_optional_metric
from ml_project.schema import DEFAULT_MODELS
from ml_project.training_types import ModelMetrics, TrainingResult


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


def adaptive_diagnostic_rows(metadata: dict[str, object], lang: Language | str) -> list[dict[str, str]]:
    """Format adaptive hybrid metadata for display."""
    fields = [
        ("residual_gate_reason", "residual_gate_reason"),
        ("backbone_rmse", "backbone_rmse"),
        ("hybrid_rmse", "hybrid_rmse"),
        ("residual_improvement", "residual_improvement"),
        ("min_improvement", "min_improvement"),
        ("training_rows", "training_rows"),
        ("validation_rows", "validation_rows"),
    ]
    rows = []
    for key, label_key in fields:
        value = metadata.get(key)
        if value is None:
            continue
        rows.append(
            {
                t("diagnostic_name", lang): t(label_key, lang),
                t("diagnostic_value", lang): _format_diagnostic_value(value),
            }
        )
    return rows


def adaptive_candidate_metadata(result: TrainingResult) -> dict[str, object]:
    """Return adaptive hybrid metadata even when another model was selected."""
    metadata = result.candidate_model_metadata.get("adaptive_hybrid")
    if metadata:
        return metadata
    if result.selected_model == "adaptive_hybrid":
        return result.model_metadata
    return {}


def _format_diagnostic_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format_optional_metric(value)
    return str(value)


__all__ = [
    "adaptive_candidate_metadata",
    "adaptive_diagnostic_rows",
    "format_metric_rows",
    "metric_chart_frame",
    "metric_rank_frame",
    "model_label",
    "selected_model_names",
    "selection_rationale",
]
