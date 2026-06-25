"""Compatibility facade for Streamlit helper functions."""

from __future__ import annotations

from ml_project.ui.dataframe_helpers import (
    feature_correlations,
    parse_csv,
    summarize_dataset,
    target_distribution,
    validate_frontend_dataset,
)
from ml_project.ui.downloads import prepare_downloads
from ml_project.ui.formatting import (
    adaptive_candidate_metadata,
    adaptive_diagnostic_rows,
    format_metric_rows,
    metric_chart_frame,
    metric_rank_frame,
    model_label,
    selected_model_names,
    selection_rationale,
)

__all__ = [
    "adaptive_candidate_metadata",
    "adaptive_diagnostic_rows",
    "feature_correlations",
    "format_metric_rows",
    "metric_chart_frame",
    "metric_rank_frame",
    "model_label",
    "parse_csv",
    "prepare_downloads",
    "selected_model_names",
    "selection_rationale",
    "summarize_dataset",
    "target_distribution",
    "validate_frontend_dataset",
]
