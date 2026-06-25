from __future__ import annotations

from enum import Enum


class Language(str, Enum):
    EN = "en"
    ZH = "zh"


_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "app_title": "Network Service Load Prediction",
        "app_subtitle": "Train, evaluate, and download regression models for network-service load.",
        "language_label": "Language / 语言",
        "overview_header": "Overview",
        "overview_text": (
            "Upload a profiling CSV, validate the schema, inspect the data, train regression "
            "models, and download the best model plus metrics."
        ),
        "upload_header": "1. Dataset Upload",
        "upload_label": "Upload profiling CSV",
        "upload_help": "Drag and drop a CSV with the required feature and target columns.",
        "smoke_mode_label": "Allow small dataset (smoke/demo mode)",
        "smoke_mode_help": "Enable only when demonstrating with the 3-row template.",
        "validate_button": "Validate dataset",
        "dataset_valid": "Dataset is valid.",
        "dataset_invalid": "Dataset validation failed.",
        "overview_rows": "Rows",
        "overview_columns": "Columns",
        "overview_features": "Features",
        "overview_target": "Target",
        "overview_missing": "Missing values",
        "preview_header": "2. Dataset Preview",
        "target_distribution_header": "Target distribution",
        "feature_correlation_header": "Feature correlation with target",
        "train_header": "3. Training",
        "train_button": "Train models",
        "training_running": "Training in progress...",
        "training_complete": "Training complete",
        "results_header": "4. Results",
        "selected_model": "Selected model",
        "metrics_header": "Model metrics",
        "model_name": "Model",
        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R²",
        "download_header": "5. Downloads",
        "download_model": "Download model (.joblib)",
        "download_metrics": "Download metrics (.json)",
        "footer_note": "Built with Streamlit and scikit-learn.",
        "error_upload": "Could not parse the uploaded file. Please upload a valid CSV.",
        "error_train": "Training failed.",
        "value_na": "n/a",
        "yes": "Yes",
        "no": "No",
        "history_header": "6. Training History",
        "history_empty": "No training runs yet.",
        "history_current": "Current",
        "history_view": "View details",
        "history_set_current": "Set current",
        "history_delete": "Delete",
        "history_download_model": "Download model",
        "history_download_dataset": "Download dataset",
        "history_download_metrics": "Download metrics",
        "history_details": "Run details",
        "model_label_dummy_mean": "Dummy baseline",
        "model_label_ridge": "Ridge regression",
        "model_label_mlp": "MLP neural network",
        "model_label_adaptive_hybrid": "Adaptive hybrid load predictor",
        "model_strategy": "Model strategy",
        "residual_enabled": "Residual correction",
    },
    "zh": {
        "app_title": "网络服务负载预测",
        "app_subtitle": "训练、评估并下载面向网络服务负载的回归模型。",
        "language_label": "Language / 语言",
        "overview_header": "概览",
        "overview_text": (
            "上传分析 CSV，验证数据模式，检查数据，训练回归模型，并下载最佳模型及指标。"
        ),
        "upload_header": "1. 数据集上传",
        "upload_label": "上传分析 CSV",
        "upload_help": "拖放包含所需特征列与目标列的 CSV 文件。",
        "smoke_mode_label": "允许小数据集（冒烟/演示模式）",
        "smoke_mode_help": "仅在使用 3 行模板演示时启用。",
        "validate_button": "验证数据集",
        "dataset_valid": "数据集验证通过。",
        "dataset_invalid": "数据集验证失败。",
        "overview_rows": "行数",
        "overview_columns": "列数",
        "overview_features": "特征数",
        "overview_target": "目标列",
        "overview_missing": "缺失值",
        "preview_header": "2. 数据预览",
        "target_distribution_header": "目标分布",
        "feature_correlation_header": "特征与目标相关性",
        "train_header": "3. 训练",
        "train_button": "训练模型",
        "training_running": "训练中……",
        "training_complete": "训练完成",
        "results_header": "4. 结果",
        "selected_model": "选中的模型",
        "metrics_header": "模型指标",
        "model_name": "模型",
        "mae": "平均绝对误差",
        "rmse": "均方根误差",
        "r2": "决定系数",
        "download_header": "5. 下载",
        "download_model": "下载模型 (.joblib)",
        "download_metrics": "下载指标 (.json)",
        "footer_note": "基于 Streamlit 与 scikit-learn 构建。",
        "error_upload": "无法解析上传的文件，请上传有效的 CSV。",
        "error_train": "训练失败。",
        "value_na": "不适用",
        "yes": "是",
        "no": "否",
        "history_header": "6. 训练历史",
        "history_empty": "暂无训练记录。",
        "history_current": "当前",
        "history_view": "查看详情",
        "history_set_current": "设为当前",
        "history_delete": "删除",
        "history_download_model": "下载模型",
        "history_download_dataset": "下载数据集",
        "history_download_metrics": "下载指标",
        "history_details": "运行详情",
        "model_label_dummy_mean": "均值基线模型",
        "model_label_ridge": "Ridge 岭回归模型",
        "model_label_mlp": "MLP 神经网络模型",
        "model_label_adaptive_hybrid": "自适应混合负载预测模型",
        "model_strategy": "模型策略",
        "residual_enabled": "残差修正",
    },
}


def t(key: str, lang: Language | str) -> str:
    lang_value = lang.value if isinstance(lang, Language) else lang
    labels = _LABELS.get(lang_value, _LABELS["en"])
    fallback = _LABELS["en"].get(key, key)
    return labels.get(key, fallback)


def supported_languages() -> list[tuple[str, str]]:
    return [
        ("English", Language.EN.value),
        ("中文", Language.ZH.value),
    ]


def label_keys() -> set[str]:
    return set(_LABELS["en"].keys())


def translation_parity_issues() -> list[str]:
    issues: list[str] = []
    en_keys = set(_LABELS["en"].keys())
    zh_keys = set(_LABELS["zh"].keys())
    for missing in sorted(en_keys - zh_keys):
        issues.append(f"Missing in zh: {missing}")
    for missing in sorted(zh_keys - en_keys):
        issues.append(f"Missing in en: {missing}")
    return issues
