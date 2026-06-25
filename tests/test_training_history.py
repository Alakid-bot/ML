import tempfile
from pathlib import Path

import joblib
import pytest

from factories import make_dataset
from ml_project.artifacts import artifact_paths, delete_run_artifacts, make_run_dir, resolve_artifacts_from_result
from ml_project.train import FEATURE_COLUMNS, TARGET_COLUMN, ModelMetrics, TrainingResult, train
from ml_project.training_history import HistoryStore, RunSummary


@pytest.fixture
def isolated_artifact_root(monkeypatch: pytest.MonkeyPatch) -> Path:
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("ARTIFACT_ROOT", tmp)
    return Path(tmp)


@pytest.fixture
def history_store(isolated_artifact_root: Path) -> HistoryStore:
    db_path = isolated_artifact_root / "test_history.sqlite3"
    url = f"sqlite:///{db_path}"
    return HistoryStore(url)


def test_default_history_store_uses_sqlite_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ml_project.training_history import HistoryStore, make_history_url

    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert make_history_url() == f"sqlite:///{tmp_path / 'ml_history.sqlite3'}"
    store = HistoryStore()
    assert store.list_runs() == []
    store.close()


def test_postgres_url_is_normalized_for_psycopg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ml_project.training_history import make_history_url

    monkeypatch.setenv(
        "DATABASE_URL", "postgres://user:pass@host:5432/dbname"
    )
    assert make_history_url() == "postgresql+psycopg://user:pass@host:5432/dbname"
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://user:pass@host:5432/dbname"
    )
    assert make_history_url() == "postgresql+psycopg://user:pass@host:5432/dbname"


def test_create_from_training_result_records_metadata_and_paths(
    isolated_artifact_root: Path,
    history_store: HistoryStore,
) -> None:
    df = make_dataset(30)
    run_dir = make_run_dir()
    paths = artifact_paths(run_dir)
    df.to_csv(paths["dataset"], index=False)

    result = train(
        dataset_path=paths["dataset"],
        output_dir=run_dir,
        model_names=["dummy_mean", "ridge"],
        test_size=0.2,
        random_state=42,
    )

    run = history_store.create_from_result(
        result,
        artifact_dir=run_dir,
        dataset_filename="dataset.csv",
        dataset_path=paths["dataset"],
    )

    assert run is not None
    assert run.id == run_dir.name
    assert run.artifact_dir == str(run_dir)
    assert run.model_path == str(paths["model"])
    assert run.metrics_path == str(paths["metrics"])
    assert run.selected_model == result.selected_model
    assert run.row_count == result.row_count
    assert Path(run.model_path).exists()
    assert Path(run.metrics_path).exists()

    loaded = history_store.get(run.id)
    assert loaded is not None
    assert loaded.dataset_filename == "dataset.csv"


def test_resolve_artifacts_preserves_result_model_path() -> None:
    run_dir = Path("artifacts") / "run-1"
    custom_model_path = run_dir / "custom-model.joblib"
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
        model_path=str(custom_model_path),
    )

    resolved = resolve_artifacts_from_result(result)

    assert resolved["model"] == custom_model_path
    assert resolved["dataset"] == artifact_paths(run_dir)["dataset"]
    assert resolved["metrics"] == artifact_paths(run_dir)["metrics"]
    assert resolved["run_dir"] == run_dir
    assert resolved["run_id"] == "run-1"


def test_delete_run_artifacts_refuses_path_outside_artifact_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="outside artifact root"):
        delete_run_artifacts(outside)


def test_report_reads_full_metrics_json(
    isolated_artifact_root: Path,
    history_store: HistoryStore,
) -> None:
    df = make_dataset(80)
    run_dir = make_run_dir()
    paths = artifact_paths(run_dir)
    df.to_csv(paths["dataset"], index=False)

    result = train(
        dataset_path=paths["dataset"],
        output_dir=run_dir,
        model_names=["ridge", "adaptive_hybrid"],
        test_size=0.2,
        random_state=42,
        cv_folds=2,
    )

    run = history_store.create_from_result(
        result,
        artifact_dir=run_dir,
        dataset_filename="dataset.csv",
        dataset_path=paths["dataset"],
    )
    report = history_store.report(run)

    assert report["selected_model"] == result.selected_model
    assert "model_metadata" in report
    assert "candidate_model_metadata" in report
    assert "adaptive_hybrid" in report["candidate_model_metadata"]
    assert report["candidate_model_metadata"]["adaptive_hybrid"]["model_type"] == "AdaptiveLoadPredictor"
    assert len(report["cross_validation"]) == 2


def test_report_falls_back_to_stored_metrics(history_store: HistoryStore) -> None:
    run_dir = make_run_dir()
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
        model_path=str(run_dir / "load_predictor.joblib"),
    )
    joblib.dump("model", result.model_path)
    run = history_store.create_from_result(result, artifact_dir=run_dir)

    assert history_store.report(run) == {"metrics": history_store.metrics(run)}


def test_list_runs_returns_newest_first(history_store: HistoryStore) -> None:
    for index in range(3):
        run_dir = make_run_dir()
        result = TrainingResult(
            selected_model="ridge",
            metrics=[],
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            row_count=10 + index,
            train_rows=8,
            test_rows=2,
            test_size=0.2,
            random_state=42,
            created_at=f"2026-01-0{index + 1}T00:00:00Z",
            sklearn_version="1.5.0",
            model_path=str(run_dir / "load_predictor.joblib"),
        )
        joblib.dump("model", result.model_path)
        history_store.create_from_result(result, artifact_dir=run_dir)

    runs = history_store.list_runs()
    assert len(runs) == 3
    counts = [run.row_count for run in runs]
    assert counts == sorted(counts, reverse=True)


def test_set_current_marks_one_run_and_unmarks_others(history_store: HistoryStore) -> None:
    run_ids = []
    for _ in range(2):
        run_dir = make_run_dir()
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
            model_path=str(run_dir / "load_predictor.joblib"),
        )
        joblib.dump("model", result.model_path)
        run = history_store.create_from_result(result, artifact_dir=run_dir)
        run_ids.append(run.id)

    history_store.set_current(run_ids[0])
    history_store.set_current(run_ids[1])

    first = history_store.get(run_ids[0])
    second = history_store.get(run_ids[1])
    assert first is not None
    assert second is not None
    assert first.current is False
    assert second.current is True

    reloaded = history_store.get(run_ids[0])
    assert reloaded is not None
    assert reloaded.current is False


def test_delete_removes_metadata_and_artifacts(history_store: HistoryStore) -> None:
    run_dir = make_run_dir()
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
        model_path=str(run_dir / "load_predictor.joblib"),
    )
    joblib.dump("model", result.model_path)
    run = history_store.create_from_result(result, artifact_dir=run_dir)

    assert history_store.get(run.id) is not None
    assert run_dir.exists()

    deleted = history_store.delete(run.id, remove_artifacts=True)
    assert deleted is True
    assert history_store.get(run.id) is None
    assert not run_dir.exists()


def test_delete_without_artifacts_keeps_files(history_store: HistoryStore) -> None:
    run_dir = make_run_dir()
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
        model_path=str(run_dir / "load_predictor.joblib"),
    )
    joblib.dump("model", result.model_path)
    run = history_store.create_from_result(result, artifact_dir=run_dir)

    history_store.delete(run.id, remove_artifacts=False)
    assert run_dir.exists()


def test_run_summary_finds_best_metric_and_builds_label(history_store: HistoryStore) -> None:
    run_dir = make_run_dir()
    result = TrainingResult(
        selected_model="ridge",
        metrics=[
            ModelMetrics(model_name="dummy_mean", mae=2.0, rmse=3.0, r2=None),
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
        model_path=str(run_dir / "load_predictor.joblib"),
    )
    joblib.dump("model", result.model_path)
    run = history_store.create_from_result(result, artifact_dir=run_dir)

    summary = RunSummary(run, history_store.metrics(run))
    best = summary.best_metric()
    assert best is not None
    assert best["model_name"] == "ridge"
    assert "RMSE=2.0000" in summary.rmse_mae_label()
    assert "MAE=1.0000" in summary.rmse_mae_label()
