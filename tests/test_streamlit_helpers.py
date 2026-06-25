from pathlib import Path

from factories import make_dataset
from ml_project.i18n import Language
from ml_project.streamlit_helpers import (
    adaptive_candidate_metadata,
    adaptive_diagnostic_rows,
    feature_correlations,
    format_metric_rows,
    metric_chart_frame,
    metric_rank_frame,
    parse_csv,
    prepare_downloads,
    selected_model_names,
    selection_rationale,
    summarize_dataset,
    target_distribution,
    validate_frontend_dataset,
)
from ml_project.train import DEFAULT_MODELS, FEATURE_COLUMNS, TARGET_COLUMN, ModelMetrics, TrainingResult


def test_streamlit_helpers_facade_preserves_public_imports() -> None:
    from ml_project import streamlit_helpers as facade

    assert facade.parse_csv is parse_csv
    assert facade.summarize_dataset is summarize_dataset
    assert facade.validate_frontend_dataset is validate_frontend_dataset
    assert facade.target_distribution is target_distribution
    assert facade.feature_correlations is feature_correlations
    assert facade.format_metric_rows is format_metric_rows
    assert facade.selected_model_names is selected_model_names
    assert facade.metric_chart_frame is metric_chart_frame
    assert facade.metric_rank_frame is metric_rank_frame
    assert facade.selection_rationale is selection_rationale
    assert facade.adaptive_diagnostic_rows is adaptive_diagnostic_rows
    assert facade.adaptive_candidate_metadata is adaptive_candidate_metadata
    assert facade.prepare_downloads is prepare_downloads


def test_streamlit_helper_split_modules_expose_public_helpers() -> None:
    from ml_project.ui.dataframe_helpers import parse_csv as parse_csv_from_module
    from ml_project.ui.downloads import prepare_downloads as prepare_downloads_from_module
    from ml_project.ui.formatting import model_label as model_label_from_module
    from ml_project.ui.formatting import selected_model_names as selected_names_from_module

    assert parse_csv_from_module is parse_csv
    assert prepare_downloads_from_module is prepare_downloads
    assert selected_names_from_module is selected_model_names
    assert model_label_from_module("ridge", Language.EN) == "Ridge regression"


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
    assert rows[0]["模型"] == "Ridge 岭回归模型"
    assert rows[0]["平均绝对误差"] == "1.0000"
    assert rows[0]["决定系数"] == "0.5000"


def test_format_metric_rows_handles_none_r2() -> None:
    metrics = [ModelMetrics(model_name="dummy_mean", mae=1.0, rmse=2.0, r2=None)]
    rows = format_metric_rows(metrics, Language.EN)
    assert rows[0]["R²"] == "n/a"


def test_adaptive_diagnostic_rows_formats_metadata() -> None:
    metadata = {
        "residual_gate_reason": "insufficient_improvement",
        "backbone_rmse": 2.0,
        "hybrid_rmse": 1.9,
        "residual_improvement": 0.05,
        "min_improvement": 0.01,
        "training_rows": 64,
        "validation_rows": 16,
    }

    rows = adaptive_diagnostic_rows(metadata, Language.ZH)

    assert rows[0]["诊断项"] == "启用判断结果"
    assert rows[0]["值"] == "insufficient_improvement"
    assert {row["诊断项"] for row in rows} >= {"仅使用 Ridge 的 RMSE", "组合模型验证 RMSE"}
    assert {row["值"] for row in rows} >= {"2.0000", "1.9000"}


def test_adaptive_candidate_metadata_uses_non_selected_candidate() -> None:
    result = TrainingResult(
        selected_model="ridge",
        metrics=[ModelMetrics(model_name="ridge", mae=1.0, rmse=2.0, r2=0.5)],
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        row_count=80,
        train_rows=64,
        test_rows=16,
        test_size=0.2,
        random_state=42,
        created_at="2026-01-01T00:00:00Z",
        sklearn_version="1.5.0",
        model_path="load_predictor.joblib",
        model_metadata={"model_type": "Ridge"},
        candidate_model_metadata={
            "adaptive_hybrid": {
                "model_type": "AdaptiveLoadPredictor",
                "residual_gate_reason": "insufficient_improvement",
            }
        },
    )

    metadata = adaptive_candidate_metadata(result)

    assert metadata["model_type"] == "AdaptiveLoadPredictor"
    assert metadata["residual_gate_reason"] == "insufficient_improvement"


def test_selected_model_names_defaults_when_empty() -> None:
    assert selected_model_names([]) == DEFAULT_MODELS
    assert selected_model_names(None) == DEFAULT_MODELS


def test_selected_model_names_filters_unknown_values() -> None:
    assert selected_model_names(["ridge", "unknown", "adaptive_hybrid"]) == [
        "ridge",
        "adaptive_hybrid",
    ]


def test_metric_chart_frame_uses_friendly_model_names() -> None:
    metrics = [ModelMetrics(model_name="ridge", mae=1.0, rmse=2.0, r2=0.5)]
    chart = metric_chart_frame(metrics, Language.EN)
    assert chart.loc[0, "model"] == "Ridge regression"
    assert chart.loc[0, "RMSE"] == 2.0


def test_metric_rank_frame_sorts_by_rmse() -> None:
    metrics = [
        ModelMetrics(model_name="mlp", mae=3.0, rmse=5.0, r2=0.1),
        ModelMetrics(model_name="ridge", mae=1.0, rmse=2.0, r2=0.5),
    ]
    ranked = metric_rank_frame(metrics, Language.EN)
    assert ranked.loc[0, "Model"] == "Ridge regression"
    assert ranked.loc[0, "Rank"] == 1


def test_selection_rationale_explains_rmse_choice() -> None:
    result = TrainingResult(
        selected_model="ridge",
        metrics=[
            ModelMetrics(model_name="dummy_mean", mae=3.0, rmse=5.0, r2=0.1),
            ModelMetrics(model_name="ridge", mae=1.0, rmse=2.0, r2=0.5),
        ],
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        row_count=10,
        train_rows=8,
        test_rows=2,
        test_size=0.2,
        random_state=42,
        created_at="2026-01-01T00:00:00Z",
        sklearn_version="1.5.0",
        model_path="load_predictor.joblib",
    )

    rationale = selection_rationale(result, Language.EN)
    assert "lowest validation RMSE" in rationale
    assert "2.0000" in rationale


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
