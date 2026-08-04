from rag_labor_code.generation.citation_postprocessor import (
    normalize_source_citations,
)

from rag_labor_code.generation.context_builder import (
    ContextSource,
)


def make_source(
    article_num: str,
) -> ContextSource:
    return ContextSource(
        article_num=article_num,
        title=f"Статья {article_num}",
        source="ТК РФ",
        score=1.0,
    )


def test_adds_source_reference_for_article() -> None:
    sources = [
        make_source("91"),
        make_source("92"),
    ]

    answer = (
        "Нормальная продолжительность "
        "рабочего времени составляет "
        "40 часов согласно статье 91 "
        "Трудового кодекса РФ."
    )

    result = normalize_source_citations(
        answer=answer,
        sources=sources,
    )

    assert "статье 91 [Источник 1]" in result


def test_uses_correct_source_number() -> None:
    sources = [
        make_source("92"),
        make_source("91"),
    ]

    answer = "Это установлено статьёй 91 " "Трудового кодекса РФ."

    result = normalize_source_citations(
        answer=answer,
        sources=sources,
    )

    assert "статьёй 91 [Источник 2]" in result


def test_preserves_correct_existing_reference() -> None:
    sources = [
        make_source("91"),
    ]

    answer = (
        "Согласно статье 91 " "[Источник 1] рабочее время " "не превышает 40 часов."
    )

    result = normalize_source_citations(
        answer=answer,
        sources=sources,
    )

    assert result.count("[Источник 1]") == 1


def test_corrects_wrong_existing_reference() -> None:
    sources = [
        make_source("91"),
        make_source("92"),
    ]

    answer = "Согласно статье 91 " "[Источник 2] применяется " "норма рабочего времени."

    result = normalize_source_citations(
        answer=answer,
        sources=sources,
    )

    assert "статье 91 [Источник 1]" in result

    assert "статье 91 [Источник 2]" not in result


def test_does_not_add_reference_for_unknown_article() -> None:
    sources = [
        make_source("91"),
    ]

    answer = "Также существует статья 999 " "Трудового кодекса РФ."

    result = normalize_source_citations(
        answer=answer,
        sources=sources,
    )

    assert result == answer


def test_supports_article_abbreviation() -> None:
    sources = [
        make_source("91"),
    ]

    answer = "Согласно ст. 91 ТК РФ " "применяется данное правило."

    result = normalize_source_citations(
        answer=answer,
        sources=sources,
    )

    assert "ст. 91 [Источник 1]" in result


def test_supports_hyphenated_article_number() -> None:
    sources = [
        make_source("341.1-1"),
    ]

    answer = "Правило предусмотрено " "статьёй 341.1-1 ТК РФ."

    result = normalize_source_citations(
        answer=answer,
        sources=sources,
    )

    assert "статьёй 341.1-1 " "[Источник 1]" in result


def test_returns_answer_unchanged_without_sources() -> None:
    answer = "Ответ без доступных источников."

    result = normalize_source_citations(
        answer=answer,
        sources=[],
    )

    assert result == answer


def test_rejects_empty_answer() -> None:
    try:
        normalize_source_citations(
            answer="   ",
            sources=[],
        )

    except ValueError as error:
        assert "answer не должен быть пустым" in str(error)

    else:
        raise AssertionError("Ожидался ValueError.")


def test_rejects_non_string_answer() -> None:
    try:
        normalize_source_citations(
            answer=123,  # type: ignore[arg-type]
            sources=[],
        )

    except TypeError as error:
        assert "answer должен быть строкой" in str(error)

    else:
        raise AssertionError("Ожидался TypeError.")
