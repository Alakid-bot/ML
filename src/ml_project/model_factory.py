"""Candidate model factory."""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_project.adaptive_model import AdaptiveLoadPredictor
from ml_project.schema import DEFAULT_MODELS, MIN_ROWS


def build_model(model_name: str, *, random_state: int, row_count: int | None = None) -> Pipeline:
    """Build a candidate model pipeline by name."""
    if model_name == "dummy_mean":
        return make_pipeline(DummyRegressor(strategy="mean"))

    if model_name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=random_state))

    if model_name == "mlp":
        use_early_stopping = row_count is None or row_count >= MIN_ROWS
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                early_stopping=use_early_stopping,
                max_iter=1000,
                random_state=random_state,
            ),
        )

    if model_name == "adaptive_hybrid":
        return make_pipeline(
            StandardScaler(),
            AdaptiveLoadPredictor(random_state=random_state),
        )

    raise ValueError(f"Unsupported model '{model_name}'. Choose from: {DEFAULT_MODELS}")


__all__ = ["build_model"]
