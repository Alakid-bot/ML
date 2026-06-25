"""Compatibility facade and CLI for network service load prediction training."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml_project.dataset import load_dataset, validate_dataset
from ml_project.evaluation import (
    evaluate_cross_validation,
    evaluate_model,
    finite_mean,
    finite_mean_or_none,
    finite_std,
    finite_std_or_none,
)
from ml_project.model_factory import build_model
from ml_project.pipeline import (
    extract_model_metadata,
    format_optional_metric,
    make_json_safe,
    train,
    train_candidate_metadata,
    train_candidates,
)
from ml_project.schema import DEFAULT_MODELS, FEATURE_COLUMNS, MIN_ROWS, TARGET_COLUMN
from ml_project.training_types import CrossValidationMetrics, ModelMetrics, TrainingResult

__all__ = [
    "CrossValidationMetrics",
    "DEFAULT_MODELS",
    "FEATURE_COLUMNS",
    "MIN_ROWS",
    "ModelMetrics",
    "TARGET_COLUMN",
    "TrainingResult",
    "build_model",
    "evaluate_cross_validation",
    "evaluate_model",
    "extract_model_metadata",
    "finite_mean",
    "finite_mean_or_none",
    "finite_std",
    "finite_std_or_none",
    "format_optional_metric",
    "load_dataset",
    "main",
    "make_json_safe",
    "parse_args",
    "train",
    "train_candidate_metadata",
    "train_candidates",
    "validate_dataset",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train network service load prediction models.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the profiling CSV file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for the saved model and metrics JSON.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Candidate models to train. Supported: {', '.join(DEFAULT_MODELS)}.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--allow-small-dataset",
        action="store_true",
        help="Allow fewer than 20 rows for smoke testing only.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Optional report-only K-fold metrics. Holdout RMSE still selects the saved model.",
    )
    return parser.parse_args()


def main() -> None:
    """Run training from the command line."""
    args = parse_args()
    result = train(
        dataset_path=args.data,
        output_dir=args.output_dir,
        model_names=args.models,
        test_size=args.test_size,
        random_state=args.random_state,
        allow_small_dataset=args.allow_small_dataset,
        cv_folds=args.cv_folds,
    )

    print(f"Selected model: {result.selected_model}")
    for metric in result.metrics:
        print(
            f"{metric.model_name}: "
            f"MAE={metric.mae:.4f} RMSE={metric.rmse:.4f} R2={format_optional_metric(metric.r2)}"
        )
    print(f"Saved model: {result.model_path}")

    if result.cross_validation:
        print(f"Cross-validation folds: {result.cross_validation[0].folds}")


if __name__ == "__main__":
    main()
