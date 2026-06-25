"""Serializable training result types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelMetrics:
    """Evaluation metrics for one trained model."""

    model_name: str
    mae: float
    rmse: float
    r2: float | None


@dataclass(frozen=True)
class CrossValidationMetrics:
    """Report-only aggregate metrics for one cross-validated model."""

    model_name: str
    folds: int
    rmse_mean: float
    rmse_std: float
    mae_mean: float
    mae_std: float
    r2_mean: float | None
    r2_std: float | None


@dataclass(frozen=True)
class TrainingResult:
    """Serializable summary of one training run."""

    selected_model: str
    metrics: list[ModelMetrics]
    feature_columns: list[str]
    target_column: str
    row_count: int
    train_rows: int
    test_rows: int
    test_size: float
    random_state: int
    created_at: str
    sklearn_version: str
    model_path: str
    model_metadata: dict[str, object] = field(default_factory=dict)
    candidate_model_metadata: dict[str, dict[str, object]] = field(default_factory=dict)
    cross_validation: list[CrossValidationMetrics] = field(default_factory=list)


__all__ = ["CrossValidationMetrics", "ModelMetrics", "TrainingResult"]
