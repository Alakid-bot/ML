"""Model evaluation helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold

from ml_project.model_factory import build_model
from ml_project.schema import FEATURE_COLUMNS, TARGET_COLUMN
from ml_project.training_types import CrossValidationMetrics, ModelMetrics


def evaluate_model(model_name: str, y_true: pd.Series, predictions: object) -> ModelMetrics:
    """Compute regression metrics for one model."""
    r2 = float(r2_score(y_true, predictions))
    return ModelMetrics(
        model_name=model_name,
        mae=float(mean_absolute_error(y_true, predictions)),
        rmse=float(root_mean_squared_error(y_true, predictions)),
        r2=None if math.isnan(r2) else r2,
    )


def evaluate_cross_validation(
    df: pd.DataFrame,
    *,
    model_names: Iterable[str],
    cv_folds: int,
    random_state: int,
) -> list[CrossValidationMetrics]:
    """Run report-only K-fold validation without affecting holdout model selection."""
    row_count = len(df)
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if cv_folds > row_count:
        raise ValueError(f"cv_folds must be no greater than row count ({row_count}).")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    folds = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_values: dict[str, dict[str, list[float]]] = {
        model_name: {"mae": [], "rmse": [], "r2": []} for model_name in model_names
    }

    for train_index, validation_index in folds.split(X):
        X_train = X.iloc[train_index]
        X_validation = X.iloc[validation_index]
        y_train = y.iloc[train_index]
        y_validation = y.iloc[validation_index]

        for model_name, values in fold_values.items():
            model = build_model(model_name, random_state=random_state, row_count=len(X_train))
            model.fit(X_train, y_train)
            predictions = model.predict(X_validation)
            fold_metrics = evaluate_model(model_name, y_validation, predictions)
            values["mae"].append(fold_metrics.mae)
            values["rmse"].append(fold_metrics.rmse)
            if fold_metrics.r2 is not None and math.isfinite(fold_metrics.r2):
                values["r2"].append(fold_metrics.r2)

    return [
        CrossValidationMetrics(
            model_name=model_name,
            folds=cv_folds,
            rmse_mean=finite_mean(values["rmse"]),
            rmse_std=finite_std(values["rmse"]),
            mae_mean=finite_mean(values["mae"]),
            mae_std=finite_std(values["mae"]),
            r2_mean=finite_mean_or_none(values["r2"]),
            r2_std=finite_std_or_none(values["r2"]),
        )
        for model_name, values in fold_values.items()
    ]


def finite_mean(values: list[float]) -> float:
    """Return the mean of finite metric values."""
    finite_values = _finite_values(values)
    return float(np.mean(finite_values)) if finite_values else 0.0


def finite_std(values: list[float]) -> float:
    """Return the population standard deviation of finite metric values."""
    finite_values = _finite_values(values)
    return float(np.std(finite_values, ddof=0)) if finite_values else 0.0


def finite_mean_or_none(values: list[float]) -> float | None:
    """Return the mean of finite values, or None when none exist."""
    finite_values = _finite_values(values)
    return float(np.mean(finite_values)) if finite_values else None


def finite_std_or_none(values: list[float]) -> float | None:
    """Return the population standard deviation of finite values, or None when none exist."""
    finite_values = _finite_values(values)
    return float(np.std(finite_values, ddof=0)) if finite_values else None


def _finite_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


__all__ = [
    "evaluate_cross_validation",
    "evaluate_model",
    "finite_mean",
    "finite_mean_or_none",
    "finite_std",
    "finite_std_or_none",
]
