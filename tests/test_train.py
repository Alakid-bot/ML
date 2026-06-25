import json
from pathlib import Path

import joblib
import pytest

from factories import make_dataset, write_dataset
from ml_project.train import (
    DEFAULT_MODELS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ModelMetrics,
    TrainingResult,
    build_model,
    train,
    validate_dataset,
)


def test_training_facade_preserves_public_imports() -> None:
    from ml_project import train as train_facade

    assert train_facade.FEATURE_COLUMNS is FEATURE_COLUMNS
    assert train_facade.DEFAULT_MODELS is DEFAULT_MODELS
    assert train_facade.ModelMetrics is ModelMetrics
    assert train_facade.TrainingResult is TrainingResult
    assert train_facade.build_model is build_model
    assert train_facade.train is train
    assert callable(train_facade.parse_args)
    assert callable(train_facade.main)


def test_training_split_modules_expose_public_helpers() -> None:
    from ml_project.dataset import load_dataset, validate_dataset as validate_from_dataset
    from ml_project.evaluation import evaluate_model, finite_mean, finite_std
    from ml_project.model_factory import build_model as build_from_factory
    from ml_project.pipeline import train as train_from_pipeline
    from ml_project.schema import FEATURE_COLUMNS as schema_features
    from ml_project.training_types import ModelMetrics as SplitModelMetrics

    assert schema_features is FEATURE_COLUMNS
    assert SplitModelMetrics is ModelMetrics
    assert build_from_factory is build_model
    assert validate_from_dataset is validate_dataset
    assert train_from_pipeline is train
    assert callable(load_dataset)
    assert callable(evaluate_model)
    assert finite_mean([1.0, 2.0]) == 1.5
    assert finite_std([1.0]) == 0.0


def test_validate_dataset_accepts_valid_rows() -> None:
    validate_dataset(make_dataset(include_metadata=True))


def test_validate_dataset_rejects_missing_required_column() -> None:
    df = make_dataset(include_metadata=True).drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(df)


def test_validate_dataset_rejects_null_required_value() -> None:
    df = make_dataset(include_metadata=True)
    df.loc[0, TARGET_COLUMN] = None

    with pytest.raises(ValueError, match="missing values"):
        validate_dataset(df)


def test_validate_dataset_rejects_small_dataset_without_override() -> None:
    with pytest.raises(ValueError, match="at least 20 rows"):
        validate_dataset(make_dataset(row_count=3, include_metadata=True))


def test_build_model_supports_candidate_models() -> None:
    for model_name in DEFAULT_MODELS:
        model = build_model(model_name, random_state=42)

        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


def test_build_model_disables_mlp_early_stopping_for_tiny_demo_data() -> None:
    model = build_model("mlp", random_state=42, row_count=3)

    assert model[-1].early_stopping is False


def test_train_writes_model_and_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_dir = tmp_path / "artifacts"
    write_dataset(dataset_path, make_dataset(include_metadata=True))

    result = train(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_names=["dummy_mean", "ridge"],
        test_size=0.2,
        random_state=42,
    )

    model_path = output_dir / "load_predictor.joblib"
    metrics_path = output_dir / "metrics.json"

    assert result.selected_model in {"dummy_mean", "ridge"}
    assert model_path.exists()
    assert metrics_path.exists()
    assert len(result.metrics) == 2
    assert result.cross_validation == []
    assert set(result.candidate_model_metadata) == {"dummy_mean", "ridge"}

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["cross_validation"] == []
    assert set(metrics_payload["candidate_model_metadata"]) == {"dummy_mean", "ridge"}

    model = joblib.load(model_path)
    predictions = model.predict(make_dataset(row_count=2, include_metadata=True)[FEATURE_COLUMNS])

    assert predictions.shape == (2,)


def test_train_supports_adaptive_hybrid_model(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_dir = tmp_path / "artifacts"
    write_dataset(dataset_path, make_dataset(row_count=80, include_metadata=True))

    result = train(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_names=["adaptive_hybrid"],
        test_size=0.2,
        random_state=42,
    )

    assert result.selected_model == "adaptive_hybrid"
    assert result.model_metadata["model_type"] == "AdaptiveLoadPredictor"
    assert "strategy" in result.model_metadata
    assert "residual_gate_reason" in result.model_metadata
    assert "residual_improvement" in result.model_metadata
    assert "min_improvement" in result.model_metadata
    assert "validation_size" in result.model_metadata
    assert "training_rows" in result.model_metadata
    assert "validation_rows" in result.model_metadata
    assert result.candidate_model_metadata["adaptive_hybrid"] == result.model_metadata


def test_train_writes_non_selected_adaptive_hybrid_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_dir = tmp_path / "artifacts"
    write_dataset(dataset_path, make_dataset(row_count=80, include_metadata=True))

    result = train(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_names=["ridge", "adaptive_hybrid"],
        test_size=0.2,
        random_state=42,
    )

    assert set(result.candidate_model_metadata) == {"ridge", "adaptive_hybrid"}
    adaptive_metadata = result.candidate_model_metadata["adaptive_hybrid"]
    assert adaptive_metadata["model_type"] == "AdaptiveLoadPredictor"
    assert "residual_gate_reason" in adaptive_metadata
    assert "residual_improvement" in adaptive_metadata

    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    written_metadata = metrics_payload["candidate_model_metadata"]["adaptive_hybrid"]
    assert written_metadata["model_type"] == "AdaptiveLoadPredictor"
    assert "residual_gate_reason" in written_metadata


def test_train_writes_report_only_cross_validation_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_dir = tmp_path / "artifacts"
    write_dataset(dataset_path, make_dataset(row_count=30, include_metadata=True))

    result = train(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_names=["dummy_mean", "ridge"],
        test_size=0.2,
        random_state=42,
        cv_folds=3,
    )

    assert result.selected_model in {"dummy_mean", "ridge"}
    assert len(result.cross_validation) == 2
    assert {metric.model_name for metric in result.cross_validation} == {"dummy_mean", "ridge"}
    assert all(metric.folds == 3 for metric in result.cross_validation)
    assert all(metric.rmse_mean >= 0 for metric in result.cross_validation)
    assert all(metric.rmse_std >= 0 for metric in result.cross_validation)
    assert all(metric.mae_mean >= 0 for metric in result.cross_validation)
    assert all(metric.mae_std >= 0 for metric in result.cross_validation)
    assert all(metric.r2_mean is not None for metric in result.cross_validation)
    assert all(metric.r2_std is not None for metric in result.cross_validation)

    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics_payload["cross_validation"]) == 2
    first_cv_metric = metrics_payload["cross_validation"][0]
    assert set(first_cv_metric) == {
        "model_name",
        "folds",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "r2_mean",
        "r2_std",
    }
    assert first_cv_metric["folds"] == 3
    assert first_cv_metric["r2_mean"] is not None
    assert first_cv_metric["r2_std"] is not None


def test_train_rejects_invalid_cross_validation_folds(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_dir = tmp_path / "artifacts"
    write_dataset(dataset_path, make_dataset(row_count=20, include_metadata=True))

    with pytest.raises(ValueError, match="at least 2"):
        train(
            dataset_path=dataset_path,
            output_dir=output_dir,
            model_names=["ridge"],
            test_size=0.2,
            random_state=42,
            cv_folds=1,
        )

    with pytest.raises(ValueError, match="row count"):
        train(
            dataset_path=dataset_path,
            output_dir=output_dir,
            model_names=["ridge"],
            test_size=0.2,
            random_state=42,
            cv_folds=21,
        )


def test_train_default_models_support_tiny_demo_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_dir = tmp_path / "artifacts"
    write_dataset(dataset_path, make_dataset(row_count=3, include_metadata=True))

    result = train(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_names=DEFAULT_MODELS,
        test_size=0.2,
        random_state=42,
        allow_small_dataset=True,
    )

    assert {metric.model_name for metric in result.metrics} == set(DEFAULT_MODELS)
    assert Path(result.model_path).exists()
