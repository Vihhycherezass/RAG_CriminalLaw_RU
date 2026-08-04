from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llama_cpp import Llama

from rag_labor_code.generation.saiga_generator import (
    DEFAULT_SYSTEM_PROMPT,
    build_rag_user_prompt,
    create_saiga_llm,
    generate_answer,
)


def create_fake_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "saiga-q4.gguf"
    model_path.write_bytes(b"fake gguf")
    return model_path


@patch("rag_labor_code.generation.saiga_generator.Llama")
def test_create_saiga_llm_uses_default_parameters(
    mock_llama_class: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = create_fake_model(tmp_path)
    expected_llm = MagicMock(spec=Llama)

    mock_llama_class.return_value = expected_llm

    result = create_saiga_llm(
        model_path=model_path,
    )

    mock_llama_class.assert_called_once_with(
        model_path=str(model_path),
        n_ctx=4096,
        n_gpu_layers=-1,
        n_threads=None,
        chat_format=None,
        verbose=False,
    )

    assert result is expected_llm


@patch("rag_labor_code.generation.saiga_generator.Llama")
def test_create_saiga_llm_uses_custom_parameters(
    mock_llama_class: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = create_fake_model(tmp_path)
    expected_llm = MagicMock(spec=Llama)

    mock_llama_class.return_value = expected_llm

    result = create_saiga_llm(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=20,
        n_threads=8,
        chat_format="llama-3",
        verbose=True,
    )

    mock_llama_class.assert_called_once_with(
        model_path=str(model_path),
        n_ctx=4096,
        n_gpu_layers=20,
        n_threads=8,
        chat_format="llama-3",
        verbose=True,
    )

    assert result is expected_llm


def test_create_saiga_llm_rejects_non_path() -> None:
    with pytest.raises(
        TypeError,
        match="model_path должен быть объектом Path!",
    ):
        create_saiga_llm(
            model_path="model.gguf",  # type: ignore[arg-type]
        )


def test_create_saiga_llm_rejects_missing_file(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "missing.gguf"

    with pytest.raises(
        FileNotFoundError,
        match="Файл модели не найден!",
    ):
        create_saiga_llm(
            model_path=model_path,
        )


def test_create_saiga_llm_rejects_directory(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.mkdir()

    with pytest.raises(
        ValueError,
        match="Путь к модели не является файлом!",
    ):
        create_saiga_llm(
            model_path=model_path,
        )


def test_create_saiga_llm_rejects_wrong_extension(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"model")

    with pytest.raises(
        ValueError,
        match="Файл модели должен иметь расширение .gguf!",
    ):
        create_saiga_llm(
            model_path=model_path,
        )


@pytest.mark.parametrize(
    "n_ctx",
    [
        0,
        -1,
    ],
)
def test_create_saiga_llm_rejects_invalid_n_ctx(
    n_ctx: int,
    tmp_path: Path,
) -> None:
    model_path = create_fake_model(tmp_path)

    with pytest.raises(
        ValueError,
        match="n_ctx должен быть больше 0!",
    ):
        create_saiga_llm(
            model_path=model_path,
            n_ctx=n_ctx,
        )


@pytest.mark.parametrize(
    "n_gpu_layers",
    [
        -2,
        -10,
    ],
)
def test_create_saiga_llm_rejects_invalid_gpu_layers(
    n_gpu_layers: int,
    tmp_path: Path,
) -> None:
    model_path = create_fake_model(tmp_path)

    with pytest.raises(
        ValueError,
        match="n_gpu_layers должен быть не меньше -1!",
    ):
        create_saiga_llm(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
        )


@pytest.mark.parametrize(
    "n_threads",
    [
        0,
        -1,
    ],
)
def test_create_saiga_llm_rejects_invalid_threads(
    n_threads: int,
    tmp_path: Path,
) -> None:
    model_path = create_fake_model(tmp_path)

    with pytest.raises(
        ValueError,
        match="n_threads должен быть больше 0!",
    ):
        create_saiga_llm(
            model_path=model_path,
            n_threads=n_threads,
        )


def test_create_saiga_llm_rejects_empty_chat_format(
    tmp_path: Path,
) -> None:
    model_path = create_fake_model(tmp_path)

    with pytest.raises(
        ValueError,
        match="chat_format не должен быть пустым!",
    ):
        create_saiga_llm(
            model_path=model_path,
            chat_format="   ",
        )


@patch("rag_labor_code.generation.saiga_generator.Llama")
def test_create_saiga_llm_rejects_invalid_result(
    mock_llama_class: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = create_fake_model(tmp_path)

    mock_llama_class.return_value = object()

    with pytest.raises(
        TypeError,
        match="Созданный объект не является Llama!",
    ):
        create_saiga_llm(
            model_path=model_path,
        )


def test_build_rag_user_prompt_creates_prompt() -> None:
    result = build_rag_user_prompt(
        question="Какова продолжительность рабочего времени?",
        context="[Источник 1 | статья 91]\nТекст статьи.",
    )

    assert "Вопрос:\n" "Какова продолжительность рабочего времени?" in result

    assert "Контекст:\n" "[Источник 1 | статья 91]\n" "Текст статьи." in result

    assert "Ответь строго на поставленный вопрос." in result

    assert (
        "Игнорируй тематически близкие, "
        "но не отвечающие на вопрос источники." in result
    )

    assert "Не смешивай инициативу работника, " "инициативу работодателя" in result


def test_build_rag_user_prompt_strips_values() -> None:
    result = build_rag_user_prompt(
        question="   Рабочее время?   ",
        context="   Текст контекста.   ",
    )

    assert "Вопрос:\nРабочее время?" in result
    assert "Контекст:\nТекст контекста." in result


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
    ],
)
def test_build_rag_user_prompt_rejects_empty_question(
    question: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Вопрос пустой!",
    ):
        build_rag_user_prompt(
            question=question,
            context="Контекст.",
        )


@pytest.mark.parametrize(
    "context",
    [
        "",
        "   ",
    ],
)
def test_build_rag_user_prompt_rejects_empty_context(
    context: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Контекст пустой!",
    ):
        build_rag_user_prompt(
            question="Вопрос?",
            context=context,
        )


def test_generate_answer_calls_chat_completion() -> None:
    llm = MagicMock(spec=Llama)

    llm.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "  Нормальная продолжительность — 40 часов.  ",
                }
            }
        ]
    }

    result = generate_answer(
        question="Какова продолжительность рабочего времени?",
        context="[Источник 1 | статья 91]\nТекст статьи.",
        llm=llm,
        max_tokens=300,
        temperature=0.2,
        top_p=0.8,
    )

    expected_user_prompt = build_rag_user_prompt(
        question="Какова продолжительность рабочего времени?",
        context="[Источник 1 | статья 91]\nТекст статьи.",
    )

    llm.create_chat_completion.assert_called_once_with(
        messages=[
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": expected_user_prompt,
            },
        ],
        max_tokens=300,
        temperature=0.2,
        top_p=0.8,
        stream=False,
    )

    assert result == "Нормальная продолжительность — 40 часов."


def test_generate_answer_uses_custom_system_prompt() -> None:
    llm = MagicMock(spec=Llama)

    llm.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Ответ.",
                }
            }
        ]
    }

    generate_answer(
        question="Вопрос?",
        context="Контекст.",
        llm=llm,
        system_prompt="Пользовательский системный промпт.",
    )

    messages = llm.create_chat_completion.call_args.kwargs["messages"]

    assert messages[0] == {
        "role": "system",
        "content": "Пользовательский системный промпт.",
    }


def test_generate_answer_rejects_invalid_llm() -> None:
    with pytest.raises(
        TypeError,
        match="llm должен быть объектом Llama!",
    ):
        generate_answer(
            question="Вопрос?",
            context="Контекст.",
            llm="не Llama",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "system_prompt",
    [
        "",
        "   ",
    ],
)
def test_generate_answer_rejects_empty_system_prompt(
    system_prompt: str,
) -> None:
    llm = MagicMock(spec=Llama)

    with pytest.raises(
        ValueError,
        match="Системный промпт пустой!",
    ):
        generate_answer(
            question="Вопрос?",
            context="Контекст.",
            llm=llm,
            system_prompt=system_prompt,
        )


@pytest.mark.parametrize(
    "max_tokens",
    [
        0,
        -1,
    ],
)
def test_generate_answer_rejects_invalid_max_tokens(
    max_tokens: int,
) -> None:
    llm = MagicMock(spec=Llama)

    with pytest.raises(
        ValueError,
        match="max_tokens должен быть больше 0!",
    ):
        generate_answer(
            question="Вопрос?",
            context="Контекст.",
            llm=llm,
            max_tokens=max_tokens,
        )


@pytest.mark.parametrize(
    "temperature",
    [
        -0.1,
        2.1,
    ],
)
def test_generate_answer_rejects_invalid_temperature(
    temperature: float,
) -> None:
    llm = MagicMock(spec=Llama)

    with pytest.raises(
        ValueError,
        match="temperature должна находиться от 0 до 2!",
    ):
        generate_answer(
            question="Вопрос?",
            context="Контекст.",
            llm=llm,
            temperature=temperature,
        )


@pytest.mark.parametrize(
    "top_p",
    [
        0.0,
        -0.1,
        1.1,
    ],
)
def test_generate_answer_rejects_invalid_top_p(
    top_p: float,
) -> None:
    llm = MagicMock(spec=Llama)

    with pytest.raises(
        ValueError,
        match="top_p должен находиться от 0 исключительно до 1 включительно!",
    ):
        generate_answer(
            question="Вопрос?",
            context="Контекст.",
            llm=llm,
            top_p=top_p,
        )


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"choices": []},
        {"choices": "не список"},
    ],
)
def test_generate_answer_rejects_invalid_choices(
    response: object,
) -> None:
    llm = MagicMock(spec=Llama)
    llm.create_chat_completion.return_value = response

    with pytest.raises(
        ValueError,
        match="Ответ LLM не содержит choices!",
    ):
        generate_answer(
            question="Вопрос?",
            context="Контекст.",
            llm=llm,
        )


@pytest.mark.parametrize(
    "response",
    [
        {"choices": [{}]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": "не словарь"}]},
    ],
)
def test_generate_answer_rejects_invalid_message(
    response: object,
) -> None:
    llm = MagicMock(spec=Llama)
    llm.create_chat_completion.return_value = response

    with pytest.raises(
        ValueError,
        match="Ответ LLM не содержит message!",
    ):
        generate_answer(
            question="Вопрос?",
            context="Контекст.",
            llm=llm,
        )


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "   ",
    ],
)
def test_generate_answer_rejects_empty_content(
    content: object,
) -> None:
    llm = MagicMock(spec=Llama)

    llm.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": content,
                }
            }
        ]
    }

    with pytest.raises(
        ValueError,
        match="LLM вернула пустой ответ!",
    ):
        generate_answer(
            question="Вопрос?",
            context="Контекст.",
            llm=llm,
        )
