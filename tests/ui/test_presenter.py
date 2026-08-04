import pytest

from rag_labor_code.generation.context_builder import (
    ContextSource,
)
from rag_labor_code.pipeline.rag_pipeline import (
    RAGPipelineResult,
)
from rag_labor_code.ui.presenter import (
    BLOCKED_FALLBACK_TEXT,
    BLOCKED_STATUS,
    NO_SOURCES_TEXT,
    SUCCESS_STATUS,
    format_sources,
    present_pipeline_result,
)


def make_source(
    *,
    article_num: str = "91",
    title: str = "Понятие рабочего времени",
) -> ContextSource:
    return ContextSource(
        article_num=article_num,
        title=title,
        source="Трудовой кодекс Российской Федерации",
        score=0.95,
    )


def test_format_sources_returns_fallback_for_empty_sources() -> None:
    result = format_sources(())

    assert result == NO_SOURCES_TEXT


def test_format_sources_formats_single_source() -> None:
    source = make_source()

    result = format_sources((source,))

    assert result == (
        "1. **Статья 91 — Понятие рабочего времени**\n"
        "   Трудовой кодекс Российской Федерации"
    )


def test_format_sources_formats_multiple_sources() -> None:
    first_source = make_source()

    second_source = make_source(
        article_num="100",
        title="Режим рабочего времени",
    )

    result = format_sources(
        (
            first_source,
            second_source,
        )
    )

    assert result == (
        "1. **Статья 91 — Понятие рабочего времени**\n"
        "   Трудовой кодекс Российской Федерации\n\n"
        "2. **Статья 100 — Режим рабочего времени**\n"
        "   Трудовой кодекс Российской Федерации"
    )


def test_present_pipeline_result_returns_success() -> None:
    source = make_source()

    pipeline_result = RAGPipelineResult(
        answer=(
            "Нормальная продолжительность рабочего времени "
            "не превышает 40 часов [Источник 1]."
        ),
        sources=(source,),
        blocked=False,
        reason=None,
    )

    result = present_pipeline_result(pipeline_result)

    assert result == (
        (
            "Нормальная продолжительность рабочего времени "
            "не превышает 40 часов [Источник 1]."
        ),
        (
            "1. **Статья 91 — Понятие рабочего времени**\n"
            "   Трудовой кодекс Российской Федерации"
        ),
        SUCCESS_STATUS,
    )


def test_present_pipeline_result_handles_no_sources() -> None:
    pipeline_result = RAGPipelineResult(
        answer=(
            "Не удалось найти релевантные положения " "Трудового кодекса для ответа."
        ),
        sources=(),
        blocked=False,
        reason=None,
    )

    result = present_pipeline_result(pipeline_result)

    assert result == (
        ("Не удалось найти релевантные положения " "Трудового кодекса для ответа."),
        NO_SOURCES_TEXT,
        SUCCESS_STATUS,
    )


def test_present_pipeline_result_returns_block_reason() -> None:
    source = make_source()

    pipeline_result = RAGPipelineResult(
        answer="",
        sources=(source,),
        blocked=True,
        reason="Запрос не относится к трудовому праву.",
    )

    result = present_pipeline_result(pipeline_result)

    assert result == (
        "Запрос не относится к трудовому праву.",
        "",
        BLOCKED_STATUS,
    )


def test_present_pipeline_result_uses_block_fallback() -> None:
    pipeline_result = RAGPipelineResult(
        answer="",
        sources=(),
        blocked=True,
        reason=None,
    )

    result = present_pipeline_result(pipeline_result)

    assert result == (
        BLOCKED_FALLBACK_TEXT,
        "",
        BLOCKED_STATUS,
    )


def test_format_sources_rejects_invalid_sources_type() -> None:
    with pytest.raises(
        TypeError,
        match="sources должен быть tuple!",
    ):
        format_sources(
            [],  # type: ignore[arg-type]
        )


def test_format_sources_rejects_invalid_source_item() -> None:
    with pytest.raises(
        TypeError,
        match=("Каждый элемент sources " "должен быть ContextSource!"),
    ):
        format_sources((object(),))  # type: ignore[arg-type]


def test_present_pipeline_result_rejects_invalid_result() -> None:
    with pytest.raises(
        TypeError,
        match=("result должен быть объектом " "RAGPipelineResult!"),
    ):
        present_pipeline_result(
            object(),  # type: ignore[arg-type]
        )
