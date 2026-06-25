"""Download payload helpers for trained artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ml_project.training_types import TrainingResult


def prepare_downloads(result: TrainingResult) -> tuple[bytes, bytes]:
    model_path = Path(result.model_path)
    metrics_path = model_path.with_name("metrics.json")
    model_bytes = model_path.read_bytes() if model_path.exists() else b""
    metrics_bytes = (
        metrics_path.read_bytes()
        if metrics_path.exists()
        else json.dumps(asdict(result), indent=2).encode("utf-8")
    )
    return model_bytes, metrics_bytes


__all__ = ["prepare_downloads"]
