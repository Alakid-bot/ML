from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session, mapped_column

from ml_project.artifacts import artifact_root, delete_run_artifacts, resolve_artifacts_from_result, run_id_from_path
from ml_project.train import TrainingResult

DEFAULT_SQLITE_NAME = "ml_history.sqlite3"


def make_history_url() -> str:
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)
        elif db_url.startswith("postgresql://") and not db_url.startswith("postgresql+"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
        return db_url
    root = artifact_root()
    root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{root / DEFAULT_SQLITE_NAME}"


class Base(DeclarativeBase):
    pass


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = mapped_column(String(36), primary_key=True)
    dataset_filename = mapped_column(String(255), nullable=True)
    dataset_path = mapped_column(String(512), nullable=True)
    artifact_dir = mapped_column(String(512), nullable=False)
    model_path = mapped_column(String(512), nullable=False)
    metrics_path = mapped_column(String(512), nullable=False)
    selected_model = mapped_column(String(64), nullable=False)
    row_count = mapped_column(Integer, nullable=False)
    train_rows = mapped_column(Integer, nullable=False)
    test_rows = mapped_column(Integer, nullable=False)
    test_size = mapped_column(Float, nullable=False)
    random_state = mapped_column(Integer, nullable=False)
    sklearn_version = mapped_column(String(32), nullable=False)
    created_at = mapped_column(DateTime(timezone=True), nullable=False)
    metrics_json = mapped_column(String, nullable=False)
    current = mapped_column(Boolean, default=False, nullable=False)


class HistoryStore:
    def __init__(self, url: str | None = None) -> None:
        self.url = url or make_history_url()
        self.engine = create_engine(self.url, echo=False, future=True)
        Base.metadata.create_all(self.engine)

    def create_from_result(
        self,
        result: TrainingResult,
        *,
        artifact_dir: Path,
        dataset_filename: str | None = None,
        dataset_path: Path | None = None,
    ) -> TrainingRun:
        run_id = run_id_from_path(artifact_dir)
        resolved = resolve_artifacts_from_result(result)
        run = TrainingRun(
            id=run_id,
            dataset_filename=dataset_filename,
            dataset_path=str(dataset_path) if dataset_path else None,
            artifact_dir=str(artifact_dir),
            model_path=str(resolved["model"]),
            metrics_path=str(resolved["metrics"]),
            selected_model=result.selected_model,
            row_count=result.row_count,
            train_rows=result.train_rows,
            test_rows=result.test_rows,
            test_size=float(result.test_size),
            random_state=result.random_state,
            sklearn_version=result.sklearn_version,
            created_at=datetime.now(UTC),
            metrics_json=json.dumps(
                [metric.__dict__ if hasattr(metric, "__dict__") else metric for metric in result.metrics],
                allow_nan=False,
            ),
            current=False,
        )
        with Session(self.engine) as session:
            session.add(run)
            session.commit()
            return self.get(run_id)

    def list_runs(self) -> list[TrainingRun]:
        with Session(self.engine) as session:
            statement = select(TrainingRun).order_by(TrainingRun.created_at.desc())
            return list(session.scalars(statement).all())

    def get(self, run_id: str) -> TrainingRun | None:
        with Session(self.engine) as session:
            return session.get(TrainingRun, run_id)

    def set_current(self, run_id: str) -> TrainingRun | None:
        with Session(self.engine) as session:
            run = session.get(TrainingRun, run_id)
            if run is None:
                return None
            for existing in session.scalars(select(TrainingRun)).all():
                existing.current = False
            run.current = True
            session.commit()
            return self.get(run_id)

    def delete(self, run_id: str, *, remove_artifacts: bool = True) -> bool:
        with Session(self.engine) as session:
            run = session.get(TrainingRun, run_id)
            if run is None:
                return False
            if remove_artifacts:
                delete_run_artifacts(Path(run.artifact_dir))
            session.delete(run)
            session.commit()
            return True

    def metrics(self, run: TrainingRun) -> list[dict[str, Any]]:
        return json.loads(run.metrics_json)

    def close(self) -> None:
        self.engine.dispose()

    def __enter__(self) -> "HistoryStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass
class RunSummary:
    run: TrainingRun
    metrics: list[dict[str, Any]]

    def best_metric(self) -> dict[str, Any] | None:
        for metric in self.metrics:
            if metric.get("model_name") == self.run.selected_model:
                return metric
        return None

    def rmse_mae_label(self) -> str:
        best = self.best_metric()
        if best is None:
            return ""
        rmse = best.get("rmse")
        mae = best.get("mae")
        if rmse is None or mae is None:
            return ""
        return f"RMSE={rmse:.4f}, MAE={mae:.4f}"
