"""Training pipeline orchestration."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_project.dataset import load_dataset, validate_dataset
from ml_project.evaluation import evaluate_cross_validation, evaluate_model
from ml_project.model_factory import build_model
from ml_project.schema import FEATURE_COLUMNS, TARGET_COLUMN
from ml_project.training_types import ModelMetrics, TrainingResult


def extract_model_metadata(model: Pipeline) -> dict[str, object]:
    """Extract optional metadata from the final estimator in a fitted pipeline."""
    final_estimator = model.steps[-1][1]
    metadata_method = getattr(final_estimator, "model_metadata", None)
    if callable(metadata_method):
        return cast(dict[str, object], metadata_method())
    return {"model_type": type(final_estimator).__name__}


def train_candidates(
    df: pd.DataFrame,
    *,
    model_names: Iterable[str],
    test_size: float,
    random_state: int,
) -> tuple[Pipeline, str, list[ModelMetrics], int, int, dict[str, Pipeline]]:
    """Train candidate models and return the best model by validation RMSE."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    trained_models: dict[str, Pipeline] = {}
    metrics: list[ModelMetrics] = []

    for model_name in model_names:
        model = build_model(model_name, random_state=random_state, row_count=len(df))
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        trained_models[model_name] = model
        metrics.append(evaluate_model(model_name, y_test, predictions))

    best_metrics = min(metrics, key=lambda item: item.rmse)
    return (
        trained_models[best_metrics.model_name],
        best_metrics.model_name,
        metrics,
        len(X_train),
        len(X_test),
        trained_models,
    )


def train_candidate_metadata(
    model_names: Iterable[str],
    trained_models: dict[str, Pipeline],
) -> dict[str, dict[str, object]]:
    """Extract metadata for every fitted candidate model."""
    return {model_name: extract_model_metadata(trained_models[model_name]) for model_name in model_names}


def train(
    *,
    dataset_path: Path,
    output_dir: Path,
    model_names: list[str],
    test_size: float,
    random_state: int,
    allow_small_dataset: bool = False,
    cv_folds: int | None = None,
) -> TrainingResult:
    """Train candidate models, persist the best pipeline, and write metrics."""
    df = load_dataset(dataset_path)
    validate_dataset(df, allow_small_dataset=allow_small_dataset)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "load_predictor.joblib"
    metrics_path = output_dir / "metrics.json"

    best_model, selected_model, metrics, train_rows, test_rows, trained_models = train_candidates(
        df,
        model_names=model_names,
        test_size=test_size,
        random_state=random_state,
    )
    candidate_model_metadata = train_candidate_metadata(model_names, trained_models)

    cross_validation = (
        evaluate_cross_validation(
            df,
            model_names=model_names,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        if cv_folds is not None
        else []
    )

    joblib.dump(best_model, model_path)

    result = TrainingResult(
        selected_model=selected_model,
        metrics=metrics,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        row_count=len(df),
        train_rows=train_rows,
        test_rows=test_rows,
        test_size=test_size,
        random_state=random_state,
        created_at=datetime.now(UTC).isoformat(),
        sklearn_version=sklearn.__version__,
        model_path=str(model_path),
        model_metadata=extract_model_metadata(best_model),
        candidate_model_metadata=candidate_model_metadata,
        cross_validation=cross_validation,
    )

    metrics_payload = make_json_safe(asdict(result))
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, allow_nan=False), encoding="utf-8")

    return result


def format_optional_metric(value: float | None) -> str:
    """Format a metric that may be unavailable for very small validation sets."""
    return "n/a" if value is None else f"{value:.4f}"


def make_json_safe(value: object) -> object:
    """Recursively replace non-finite floats with None before strict JSON encoding."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in cast(dict[Any, object], value).items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    return value


__all__ = [
    "extract_model_metadata",
    "format_optional_metric",
    "make_json_safe",
    "train",
    "train_candidate_metadata",
    "train_candidates",
]
