import csv
from pathlib import Path

import joblib
import pandas as pd
import pytest

from ml_project.train import (
    DEFAULT_MODELS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_model,
    train,
    validate_dataset,
)


def make_dataset(row_count: int = 30) -> pd.DataFrame:
    rows = []
    for index in range(row_count):
        traffic = 100 + index * 10
        cpu_cores = 1 + index % 4
        ram_gb = 2 + (index % 3) * 2
        link_capacity = 1000
        cpu_utilization = min(95.0, 35 + traffic / 10 - cpu_cores * 4)
        memory_utilization = min(90.0, 30 + ram_gb * 3 + index % 5)
        latency = 10 + traffic / 50 + max(0, cpu_utilization - 70) * 0.8
        throughput = traffic * (1 - min(0.1, index / 1000))
        packet_loss = max(0.0, (latency - 35) / 100)
        max_supported_load = traffic + cpu_cores * 45 + ram_gb * 8 - latency * 1.5

        rows.append(
            {
                "sample_id": index + 1,
                "service_name": "snort",
                "traffic_input_mbps": traffic,
                "cpu_cores": cpu_cores,
                "ram_gb": ram_gb,
                "link_capacity_mbps": link_capacity,
                "cpu_utilization_percent": cpu_utilization,
                "memory_utilization_percent": memory_utilization,
                "latency_ms": latency,
                "throughput_mbps": throughput,
                "packet_loss_percent": packet_loss,
                "max_supported_load_mbps": max_supported_load,
            }
        )

    return pd.DataFrame(rows)


def write_dataset(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def test_validate_dataset_accepts_valid_rows() -> None:
    validate_dataset(make_dataset())


def test_validate_dataset_rejects_missing_required_column() -> None:
    df = make_dataset().drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(df)


def test_validate_dataset_rejects_null_required_value() -> None:
    df = make_dataset()
    df.loc[0, TARGET_COLUMN] = None

    with pytest.raises(ValueError, match="missing values"):
        validate_dataset(df)


def test_validate_dataset_rejects_small_dataset_without_override() -> None:
    with pytest.raises(ValueError, match="at least 20 rows"):
        validate_dataset(make_dataset(row_count=3))


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
    write_dataset(dataset_path, make_dataset())

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

    model = joblib.load(model_path)
    predictions = model.predict(make_dataset(row_count=2)[FEATURE_COLUMNS])

    assert predictions.shape == (2,)


def test_train_supports_adaptive_hybrid_model(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_dir = tmp_path / "artifacts"
    write_dataset(dataset_path, make_dataset(row_count=80))

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


def test_train_default_models_support_tiny_demo_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_dir = tmp_path / "artifacts"
    write_dataset(dataset_path, make_dataset(row_count=3))

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
