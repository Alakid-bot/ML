import tempfile
from pathlib import Path

import joblib
import pandas as pd
import pytest

from ml_project.artifacts import artifact_paths, make_run_dir
from ml_project.train import FEATURE_COLUMNS, TARGET_COLUMN, TrainingResult, train
from ml_project.training_history import HistoryStore, RunSummary


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
            {"model_name": "dummy_mean", "mae": 2.0, "rmse": 3.0, "r2": None},
            {"model_name": "ridge", "mae": 1.0, "rmse": 2.0, "r2": 0.5},
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
