from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ml_project.artifacts import artifact_paths, make_run_dir
from ml_project.i18n import Language, supported_languages, t
from ml_project.streamlit_helpers import (
    feature_correlations,
    format_metric_rows,
    parse_csv,
    prepare_downloads,
    summarize_dataset,
    target_distribution,
    validate_frontend_dataset,
    model_label,
)
from ml_project.train import DEFAULT_MODELS, TARGET_COLUMN, train
from ml_project.training_history import HistoryStore, RunSummary
from ml_project.ui_config import inject_custom_css

TRAINING_OUTPUT_DIR = Path("artifacts/streamlit")


def initialize_session() -> None:
    defaults: dict[str, Any] = {
        "language": Language.EN.value,
        "uploaded_df": None,
        "dataset_valid": False,
        "validation_message": "",
        "training_result": None,
        "training_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_language() -> None:
    selected = st.session_state.get("language_select", Language.EN.value)
    st.session_state["language"] = selected


def current_language() -> str:
    return st.session_state.get("language", Language.EN.value)


def run_validation(df: pd.DataFrame, allow_small: bool) -> None:
    ok, message = validate_frontend_dataset(df, allow_small_dataset=allow_small)
    st.session_state["dataset_valid"] = ok
    st.session_state["validation_message"] = message


def run_training(df: pd.DataFrame, allow_small: bool) -> None:
    from ml_project.artifacts import delete_run_artifacts

    st.session_state["training_error"] = ""
    st.session_state["training_result"] = None

    run_dir: Path | None = None
    try:
        run_dir = make_run_dir()
        paths = artifact_paths(run_dir)
        df.to_csv(paths["dataset"], index=False)
        dataset_path = paths["dataset"]

        result = train(
            dataset_path=dataset_path,
            output_dir=run_dir,
            model_names=DEFAULT_MODELS,
            test_size=0.2,
            random_state=42,
            allow_small_dataset=allow_small,
        )
        st.session_state["training_result"] = result

        with HistoryStore() as store:
            uploaded_name = ""
            uploaded_file = st.session_state.get("csv_uploader")
            if uploaded_file is not None:
                uploaded_name = getattr(uploaded_file, "name", "")
            run = store.create_from_result(
                result,
                artifact_dir=run_dir,
                dataset_filename=uploaded_name,
                dataset_path=dataset_path,
            )
            store.set_current(run.id)
    except Exception as exc:
        st.session_state["training_error"] = str(exc)
        if run_dir is not None:
            try:
                delete_run_artifacts(run_dir)
            except Exception:
                pass


def render_sidebar() -> None:
    lang = current_language()
    st.sidebar.title(t("app_title", lang))
    st.sidebar.selectbox(
        t("language_label", lang),
        options=[value for _, value in supported_languages()],
        format_func=lambda value: next(
            label for label, val in supported_languages() if val == value
        ),
        index=0 if lang == Language.EN.value else 1,
        key="language_select",
        on_change=set_language,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{t('overview_header', lang)}**")
    st.sidebar.markdown(
        f"- {t('upload_header', lang)}\n- {t('preview_header', lang)}\n- {t('train_header', lang)}\n- {t('results_header', lang)}\n- {t('download_header', lang)}\n- {t('history_header', lang)}"
    )


def render_header() -> None:
    lang = current_language()
    st.markdown(inject_custom_css(), unsafe_allow_html=True)
    st.title(t("app_title", lang))
    st.caption(t("app_subtitle", lang))


def render_overview() -> None:
    lang = current_language()
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.header(t("overview_header", lang))
    st.markdown(t("overview_text", lang))


def render_upload() -> None:
    lang = current_language()
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.header(t("upload_header", lang))

    uploaded = st.file_uploader(
        t("upload_label", lang),
        type=["csv"],
        help=t("upload_help", lang),
        key="csv_uploader",
    )

    allow_small = st.checkbox(
        t("smoke_mode_label", lang),
        help=t("smoke_mode_help", lang),
        key="smoke_mode",
    )

    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        df, error = parse_csv(raw_bytes)

        if df is None:
            st.error(f"{t('error_upload', lang)} {error}")
            st.session_state["uploaded_df"] = None
            st.session_state["dataset_valid"] = False
            return

        st.session_state["uploaded_df"] = df

        if st.button(t("validate_button", lang), key="validate_button", type="primary"):
            run_validation(df, allow_small)

        if st.session_state["dataset_valid"]:
            st.success(t("dataset_valid", lang))
        elif st.session_state["validation_message"]:
            st.error(f"{t('dataset_invalid', lang)} {st.session_state['validation_message']}")


def render_preview() -> None:
    lang = current_language()
    df = st.session_state.get("uploaded_df")
    if df is None:
        return

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.header(t("preview_header", lang))

    summary = summarize_dataset(df)
    cols = st.columns(4)
    metrics = [
        (t("overview_rows", lang), summary["row_count"]),
        (t("overview_columns", lang), summary["column_count"]),
        (t("overview_features", lang), summary["feature_count"]),
        (t("overview_missing", lang), summary["missing_required"]),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(
                f"<div class='metric-card'><small>{label}</small><br><strong>{value}</strong></div>",
                unsafe_allow_html=True,
            )

    st.dataframe(df.head(10), use_container_width=True)

    target_col = summary.get("target_present")
    if target_col:
        target_df = target_distribution(df)
        if not target_df.empty:
            st.subheader(t("target_distribution_header", lang))
            st.line_chart(target_df.rename(columns={TARGET_COLUMN: t("overview_target", lang)}))

        corr_df = feature_correlations(df)
        if not corr_df.empty:
            st.subheader(t("feature_correlation_header", lang))
            st.bar_chart(corr_df.set_index("feature")["correlation"])


def render_training() -> None:
    lang = current_language()
    df = st.session_state.get("uploaded_df")
    if df is None or not st.session_state.get("dataset_valid", False):
        return

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.header(t("train_header", lang))

    if st.button(t("train_button", lang), key="train_button", type="primary"):
        with st.spinner(t("training_running", lang)):
            run_training(df, st.session_state.get("smoke_mode", False))

    if st.session_state.get("training_error"):
        st.error(f"{t('error_train', lang)} {st.session_state['training_error']}")


def render_results() -> None:
    lang = current_language()
    result = st.session_state.get("training_result")
    if result is None:
        return

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.header(t("results_header", lang))
    st.success(f"{t('training_complete', lang)}: {result.created_at}")
    st.metric(t("selected_model", lang), model_label(result.selected_model, lang))

    metadata = result.model_metadata
    if metadata:
        strategy = metadata.get("strategy")
        residual_enabled = metadata.get("residual_enabled")
        if strategy:
            st.caption(f"{t('model_strategy', lang)}: {strategy}")
        if residual_enabled is not None:
            st.caption(
                f"{t('residual_enabled', lang)}: "
                f"{t('yes', lang) if residual_enabled else t('no', lang)}"
            )

    st.subheader(t("metrics_header", lang))
    rows = format_metric_rows(result.metrics, lang)
    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df, use_container_width=True)


def render_downloads() -> None:
    lang = current_language()
    result = st.session_state.get("training_result")
    if result is None:
        return

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.header(t("download_header", lang))

    model_bytes, metrics_bytes = prepare_downloads(result)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label=t("download_model", lang),
            data=model_bytes,
            file_name=f"load_predictor_{timestamp}.joblib",
            mime="application/octet-stream",
        )
    with col2:
        st.download_button(
            label=t("download_metrics", lang),
            data=metrics_bytes,
            file_name=f"metrics_{timestamp}.json",
            mime="application/json",
        )


def _download_button(path: Path, label: str, file_name: str, mime: str) -> None:
    if path.exists():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=file_name,
            mime=mime,
        )


def render_history() -> None:
    lang = current_language()
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.header(t("history_header", lang))

    with HistoryStore() as store:
        runs = store.list_runs()
        if not runs:
            st.info(t("history_empty", lang))
            return

        for run in runs:
            summary = RunSummary(run, store.metrics(run))
            best = summary.best_metric()
            rmse = best.get("rmse") if best else None
            mae = best.get("mae") if best else None
            summary_parts = [f"{t('selected_model', lang)}: {model_label(run.selected_model, lang)}"]
            if run.row_count:
                summary_parts.append(f"{t('overview_rows', lang)}: {run.row_count}")
            if rmse is not None and mae is not None:
                summary_parts.append(f"RMSE={rmse:.4f} / MAE={mae:.4f}")
            if run.current:
                summary_parts.append(f"✅ {t('history_current', lang)}")

            with st.expander(f"{run.created_at:%Y-%m-%d %H:%M:%S} — {' | '.join(summary_parts)}"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button(t("history_view", lang), key=f"view_{run.id}"):
                        st.session_state["history_view_run_id"] = run.id
                with col2:
                    if not run.current and st.button(
                        t("history_set_current", lang), key=f"current_{run.id}"
                    ):
                        store.set_current(run.id)
                        st.rerun()
                with col3:
                    if st.button(t("history_delete", lang), key=f"delete_{run.id}"):
                        store.delete(run.id, remove_artifacts=True)
                        st.rerun()
                with col4:
                    _download_button(
                        Path(run.model_path),
                        t("history_download_model", lang),
                        f"load_predictor_{run.id}.joblib",
                        "application/octet-stream",
                    )

                c1, c2 = st.columns(2)
                with c1:
                    dataset_path = Path(run.dataset_path) if run.dataset_path else None
                    if dataset_path and dataset_path.exists():
                        _download_button(
                            dataset_path,
                            t("history_download_dataset", lang),
                            f"uploaded_dataset_{run.id}.csv",
                            "text/csv",
                        )
                with c2:
                    metrics_path = Path(run.metrics_path)
                    if metrics_path.exists():
                        _download_button(
                            metrics_path,
                            t("history_download_metrics", lang),
                            f"metrics_{run.id}.json",
                            "application/json",
                        )

                view_id = st.session_state.get("history_view_run_id")
                if view_id == run.id:
                    st.subheader(t("history_details", lang))
                    st.json(
                        {
                            "id": run.id,
                            "created_at": run.created_at.isoformat(),
                            "selected_model": run.selected_model,
                            "selected_model_label": model_label(run.selected_model, lang),
                            "dataset_filename": run.dataset_filename,
                            "row_count": run.row_count,
                            "train_rows": run.train_rows,
                            "test_rows": run.test_rows,
                            "test_size": run.test_size,
                            "random_state": run.random_state,
                            "sklearn_version": run.sklearn_version,
                            "current": run.current,
                            "metrics": summary.metrics,
                        }
                    )


def render_footer() -> None:
    lang = current_language()
    st.markdown("---")
    st.caption(t("footer_note", lang))


def main() -> None:
    initialize_session()
    render_sidebar()
    render_header()
    render_overview()
    render_upload()
    render_preview()
    render_training()
    render_results()
    render_downloads()
    render_history()
    render_footer()


if __name__ == "__main__":
    main()
