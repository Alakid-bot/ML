from pathlib import Path

import pandas as pd

from ml_project.i18n import Language
from ml_project.streamlit_helpers import (
    feature_correlations,
    format_metric_rows,
    parse_csv,
    prepare_downloads,
    summarize_dataset,
    target_distribution,
    validate_frontend_dataset,
)
from ml_project.train import FEATURE_COLUMNS, TARGET_COLUMN, ModelMetrics, TrainingResult


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


def test_parse_csv_with_valid_bytes() -> None:
    df = make_dataset(5)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    parsed, error = parse_csv(csv_bytes)
    assert parsed is not None
    assert error == ""
    assert len(parsed) == 5


def test_parse_csv_with_none() -> None:
    parsed, error = parse_csv(None)
    assert parsed is None
    assert error == ""


def test_parse_csv_with_invalid_bytes() -> None:
    parsed, error = parse_csv(b"\xff\xfe")
    assert parsed is None
    assert error != ""


def test_summarize_dataset_with_valid_data() -> None:
    df = make_dataset(10)
    summary = summarize_dataset(df)
    assert summary["row_count"] == 10
    assert summary["column_count"] == 10
    assert summary["feature_count"] == len(FEATURE_COLUMNS)
    assert summary["target_present"] is True
    assert summary["missing_required"] == 0


def test_summarize_dataset_with_missing_target() -> None:
    df = make_dataset(10).drop(columns=[TARGET_COLUMN])
    summary = summarize_dataset(df)
    assert summary["target_present"] is False
    assert summary["target_unique"] == 0


def test_validate_frontend_dataset_accepts_valid_rows() -> None:
    df = make_dataset(30)
    ok, message = validate_frontend_dataset(df, allow_small_dataset=False)
    assert ok is True
    assert message == ""


def test_validate_frontend_dataset_rejects_small_dataset() -> None:
    df = make_dataset(3)
    ok, message = validate_frontend_dataset(df, allow_small_dataset=False)
    assert ok is False
    assert "at least 20 rows" in message


def test_validate_frontend_dataset_allows_small_dataset_when_flagged() -> None:
    df = make_dataset(3)
    df["max_supported_load_mbps"] = [100, 200, 300]
    ok, message = validate_frontend_dataset(df, allow_small_dataset=True)
    assert ok is True
    assert message == ""


def test_format_metric_rows_localizes_headers() -> None:
    metrics = [ModelMetrics(model_name="ridge", mae=1.0, rmse=2.0, r2=0.5)]
    rows = format_metric_rows(metrics, Language.ZH)
    assert rows[0]["模型"] == "ridge"
    assert rows[0]["平均绝对误差"] == "1.0000"
    assert rows[0]["决定系数"] == "0.5000"


def test_format_metric_rows_handles_none_r2() -> None:
    metrics = [ModelMetrics(model_name="dummy_mean", mae=1.0, rmse=2.0, r2=None)]
    rows = format_metric_rows(metrics, Language.EN)
    assert rows[0]["R²"] == "n/a"


def test_target_distribution_returns_target_column() -> None:
    df = make_dataset(10)
    target_df = target_distribution(df)
    assert list(target_df.columns) == [TARGET_COLUMN]
    assert len(target_df) == 10


def test_target_distribution_returns_empty_without_target() -> None:
    df = make_dataset(10).drop(columns=[TARGET_COLUMN])
    target_df = target_distribution(df)
    assert target_df.empty


def test_feature_correlations_returns_numeric_features() -> None:
    df = make_dataset(30)
    corr_df = feature_correlations(df)
    assert "feature" in corr_df.columns
    assert "correlation" in corr_df.columns
    assert len(corr_df) == len(FEATURE_COLUMNS)


def test_feature_correlations_returns_empty_without_target() -> None:
    df = make_dataset(10).drop(columns=[TARGET_COLUMN])
    corr_df = feature_correlations(df)
    assert corr_df.empty


def test_prepare_downloads_reads_existing_artifacts(tmp_path: Path) -> None:
    model_path = tmp_path / "load_predictor.joblib"
    metrics_path = tmp_path / "metrics.json"
    model_path.write_bytes(b"model-data")
    metrics_path.write_bytes(b"metrics-data")

    result = TrainingResult(
        selected_model="ridge",
        metrics=[],
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        row_count=10,
        train_rows=8,
        test_rows=2,
        test_size=0.2,
        random_state=42,
        created_at="2026-01-01T00:00:00Z",
        sklearn_version="1.5.0",
        model_path=str(model_path),
    )
    model_bytes, metrics_bytes = prepare_downloads(result)
    assert model_bytes == b"model-data"
    assert metrics_bytes == b"metrics-data"


def test_prepare_downloads_serializes_result_when_metrics_missing(tmp_path: Path) -> None:
    model_path = tmp_path / "load_predictor.joblib"
    result = TrainingResult(
        selected_model="ridge",
        metrics=[ModelMetrics(model_name="ridge", mae=1.0, rmse=2.0, r2=0.5)],
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        row_count=10,
        train_rows=8,
        test_rows=2,
        test_size=0.2,
        random_state=42,
        created_at="2026-01-01T00:00:00Z",
        sklearn_version="1.5.0",
        model_path=str(model_path),
    )
    model_bytes, metrics_bytes = prepare_downloads(result)
    assert model_bytes == b""
    assert b"ridge" in metrics_bytes
