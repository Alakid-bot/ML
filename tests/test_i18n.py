import pytest

from ml_project.i18n import Language, label_keys, supported_languages, t, translation_parity_issues


def test_supported_languages_returns_pairs() -> None:
    languages = supported_languages()
    assert languages == [("English", "en"), ("中文", "zh")]


@pytest.mark.parametrize(
    ("key", "lang", "expected"),
    [
        ("app_title", Language.EN, "Network Service Load Prediction"),
        ("app_title", Language.ZH, "网络服务负载预测"),
        ("app_title", "en", "Network Service Load Prediction"),
        ("app_title", "zh", "网络服务负载预测"),
    ],
)
def test_t_returns_expected_label(key: str, lang: Language | str, expected: str) -> None:
    assert t(key, lang) == expected


def test_t_falls_back_to_english_for_unknown_language() -> None:
    assert t("app_title", "xx") == "Network Service Load Prediction"


def test_t_falls_back_to_key_for_unknown_label() -> None:
    assert t("unknown_key", Language.EN) == "unknown_key"


def test_label_keys_is_non_empty() -> None:
    keys = label_keys()
    assert "app_title" in keys
    assert "train_button" in keys


def test_translation_parity_has_no_issues() -> None:
    assert translation_parity_issues() == []
