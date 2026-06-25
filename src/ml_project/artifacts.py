from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from ml_project.train import TrainingResult

DEFAULT_ARTIFACT_ROOT = Path("artifacts/streamlit_runs")
DATASET_FILENAME = "uploaded_dataset.csv"
MODEL_FILENAME = "load_predictor.joblib"
METRICS_FILENAME = "metrics.json"


def artifact_root() -> Path:
    root = os.environ.get("ARTIFACT_ROOT")
    if root:
        return Path(root)
    return DEFAULT_ARTIFACT_ROOT


def make_run_dir(run_id: str | None = None) -> Path:
    run_id = run_id or uuid4().hex
    run_dir = artifact_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def artifact_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "dataset": run_dir / DATASET_FILENAME,
        "model": run_dir / MODEL_FILENAME,
        "metrics": run_dir / METRICS_FILENAME,
    }


def run_id_from_path(path: Path) -> str:
    return path.name


def delete_run_artifacts(run_dir: Path) -> None:
    root = artifact_root().resolve()
    target = run_dir.resolve()
    if not target.exists():
        return
    if target == root or root not in target.parents:
        raise ValueError(f"Refusing to delete artifact directory outside artifact root: {target}")
    for child in target.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            delete_run_artifacts(child)
    target.rmdir()


def resolve_artifacts_from_result(result: TrainingResult) -> dict[str, Path | str]:
    model_path = Path(result.model_path)
    run_dir = model_path.parent
    paths = artifact_paths(run_dir)
    return {
        "dataset": paths["dataset"],
        "model": model_path,
        "metrics": paths["metrics"],
        "run_dir": run_dir,
        "run_id": run_dir.name,
    }
