from pathlib import Path

import pytest

from rag_labor_code.config import AppConfig


def test_app_config_uses_chatml_by_default() -> None:
    config = AppConfig(
        saiga_model_path=Path("saiga.gguf"),
    )

    assert config.chat_format == "chatml"


def test_app_config_accepts_custom_chat_format() -> None:
    config = AppConfig(
        saiga_model_path=Path("saiga.gguf"),
        chat_format="llama-2",
    )

    assert config.chat_format == "llama-2"


def test_app_config_accepts_none_chat_format() -> None:
    config = AppConfig(
        saiga_model_path=Path("saiga.gguf"),
        chat_format=None,
    )

    assert config.chat_format is None


def test_app_config_rejects_invalid_chat_format_type() -> None:
    with pytest.raises(
        TypeError,
        match="chat_format должен быть str или None!",
    ):
        AppConfig(
            saiga_model_path=Path("saiga.gguf"),
            chat_format=123,  # type: ignore[arg-type]
        )


def test_app_config_rejects_empty_chat_format() -> None:
    with pytest.raises(
        ValueError,
        match="chat_format не должен быть пустым!",
    ):
        AppConfig(
            saiga_model_path=Path("saiga.gguf"),
            chat_format="   ",
        )


def test_app_config_uses_safe_n_ctx_by_default() -> None:
    config = AppConfig(
        saiga_model_path=Path("saiga.gguf"),
    )

    assert config.n_ctx == 4096
