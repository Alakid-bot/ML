"""Training pipeline for network service load prediction."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "traffic_input_mbps",
    "cpu_cores",
    "ram_gb",
    "link_capacity_mbps",
    "cpu_utilization_percent",
    "memory_utilization_percent",
    "latency_ms",
    "throughput_mbps",
    "packet_loss_percent",
]

TARGET_COLUMN = "max_supported_load_mbps"
DEFAULT_MODELS = ["dummy_mean", "ridge", "mlp"]
MIN_ROWS = 20


@dataclass(frozen=True)
class ModelMetrics:
    """Evaluation metrics for one trained model."""

    model_name: str
    mae: float
    rmse: float
    r2: float | None


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


def build_model(model_name: str, *, random_state: int) -> Pipeline:
    """Build a candidate model pipeline by name."""
    if model_name == "dummy_mean":
        return make_pipeline(DummyRegressor(strategy="mean"))

    if model_name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=random_state))

    if model_name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                early_stopping=True,
                max_iter=1000,
                random_state=random_state,
            ),
        )

    raise ValueError(f"Unsupported model '{model_name}'. Choose from: {DEFAULT_MODELS}")


def evaluate_model(model_name: str, y_true: pd.Series, predictions: object) -> ModelMetrics:
    """Compute regression metrics for one model."""
    r2 = float(r2_score(y_true, predictions))
    return ModelMetrics(
        model_name=model_name,
        mae=float(mean_absolute_error(y_true, predictions)),
        rmse=float(root_mean_squared_error(y_true, predictions)),
        r2=None if math.isnan(r2) else r2,
    )


def train_candidates(
    df: pd.DataFrame,
    *,
    model_names: Iterable[str],
    test_size: float,
    random_state: int,
) -> tuple[Pipeline, str, list[ModelMetrics], int, int]:
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
        model = build_model(model_name, random_state=random_state)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        trained_models[model_name] = model
        metrics.append(evaluate_model(model_name, y_test, predictions))

    best_metrics = min(metrics, key=lambda item: item.rmse)
    return trained_models[best_metrics.model_name], best_metrics.model_name, metrics, len(X_train), len(X_test)


def train(
    *,
    dataset_path: Path,
    output_dir: Path,
    model_names: list[str],
    test_size: float,
    random_state: int,
    allow_small_dataset: bool = False,
) -> TrainingResult:
    """Train candidate models, persist the best pipeline, and write metrics."""
    df = load_dataset(dataset_path)
    validate_dataset(df, allow_small_dataset=allow_small_dataset)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "load_predictor.joblib"
    metrics_path = output_dir / "metrics.json"

    best_model, selected_model, metrics, train_rows, test_rows = train_candidates(
        df,
        model_names=model_names,
        test_size=test_size,
        random_state=random_state,
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
    )

    metrics_payload = asdict(result)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, allow_nan=False), encoding="utf-8")

    return result


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
    )

    print(f"Selected model: {result.selected_model}")
    for metric in result.metrics:
        print(
            f"{metric.model_name}: "
            f"MAE={metric.mae:.4f} RMSE={metric.rmse:.4f} R2={format_optional_metric(metric.r2)}"
        )
    print(f"Saved model: {result.model_path}")


def format_optional_metric(value: float | None) -> str:
    """Format a metric that may be unavailable for very small validation sets."""
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
