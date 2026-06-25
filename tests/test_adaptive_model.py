from pathlib import Path
from typing import cast

import joblib
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_project.adaptive_model import AdaptiveLoadPredictor


def make_regression_frame(row_count: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    targets = []
    for index in range(row_count):
        traffic = 100 + index * 8
        cpu = 1 + index % 6
        latency = 12 + traffic / 80 + max(0, traffic - 450) ** 2 / 8000
        loss = max(0.0, latency - 30) / 100
        rows.append(
            {
                "traffic_input_mbps": traffic,
                "cpu_cores": cpu,
                "ram_gb": 4 + index % 4,
                "link_capacity_mbps": 1000,
                "cpu_utilization_percent": 30 + traffic / 20 - cpu * 2,
                "memory_utilization_percent": 45 + index % 10,
                "latency_ms": latency,
                "throughput_mbps": traffic * (1 - loss),
                "packet_loss_percent": loss,
            }
        )
        targets.append(traffic + cpu * 45 - latency * 2 + max(0, traffic - 500) * 0.25)

    return pd.DataFrame(rows), pd.Series(targets)


def test_adaptive_model_fits_and_predicts() -> None:
    X, y = make_regression_frame()
    model = AdaptiveLoadPredictor(random_state=42, min_samples_for_residual=30)

    model.fit(X, y)
    predictions = model.predict(X.head(5))

    assert predictions.shape == (5,)
    metadata = model.model_metadata()
    assert metadata["model_type"] == "AdaptiveLoadPredictor"
    assert metadata["residual_gate_reason"] in {
        "improvement_met",
        "insufficient_improvement",
        "non_finite_metric",
    }
    assert metadata["min_improvement"] == 0.01
    assert metadata["validation_size"] == 0.2
    training_rows = cast(int, metadata["training_rows"])
    validation_rows = cast(int, metadata["validation_rows"])
    assert training_rows + validation_rows == len(X)
    assert validation_rows > 0
    if metadata["residual_enabled"]:
        assert cast(float, metadata["residual_improvement"]) >= cast(float, metadata["min_improvement"])


def test_adaptive_model_falls_back_on_small_dataset() -> None:
    X, y = make_regression_frame(row_count=10)
    model = AdaptiveLoadPredictor(random_state=42, min_samples_for_residual=30)

    model.fit(X, y)

    metadata = model.model_metadata()
    assert metadata["residual_enabled"] is False
    assert metadata["strategy"] == "ridge_backbone_small_dataset"
    assert metadata["residual_gate_reason"] == "insufficient_samples"
    assert metadata["residual_improvement"] is None
    assert metadata["training_rows"] == len(X)
    assert metadata["validation_rows"] == 0


def test_adaptive_model_disables_residual_when_improvement_below_gate() -> None:
    X, y = make_regression_frame()
    model = AdaptiveLoadPredictor(
        random_state=42,
        min_samples_for_residual=30,
        min_improvement=1.0,
    )

    model.fit(X, y)

    metadata = model.model_metadata()
    assert metadata["residual_enabled"] is False
    assert metadata["residual_gate_reason"] == "insufficient_improvement"
    assert metadata["residual_improvement"] is not None


def test_adaptive_model_is_joblib_serializable(tmp_path: Path) -> None:
    X, y = make_regression_frame()
    model = AdaptiveLoadPredictor(random_state=42, min_samples_for_residual=30)
    model.fit(X, y)

    model_path = tmp_path / "adaptive.joblib"
    joblib.dump(model, model_path)
    loaded = joblib.load(model_path)

    predictions = loaded.predict(X.head(3))
    assert predictions.shape == (3,)


def test_adaptive_model_pipeline_is_joblib_serializable(tmp_path: Path) -> None:
    X, y = make_regression_frame()
    pipeline = make_pipeline(StandardScaler(), AdaptiveLoadPredictor(random_state=42, min_samples_for_residual=30))
    pipeline.fit(X, y)

    model_path = tmp_path / "adaptive_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    loaded = joblib.load(model_path)

    predictions = loaded.predict(X.head(3))
    assert predictions.shape == (3,)
