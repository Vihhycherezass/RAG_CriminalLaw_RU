import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from rag_labor_code.generation.context_builder import (
    ContextResult,
    ContextSource,
    build_context,
)

CONTEXT_SEPARATOR = "\n\n---\n\n"


def make_candidate(
    article_num: str = "91",
    title: str = "Понятие рабочего времени",
    text: str = "Статья 91. Понятие рабочего времени.",
    source: str = "Трудовой кодекс Российской Федерации",
    score: float | None = 0.9,
) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            text=text,
            metadata={
                "article_num": article_num,
                "title": title,
                "source": source,
            },
        ),
        score=score,
    )


def test_build_context_creates_context_and_sources() -> None:
    candidates = [
        make_candidate(
            article_num="92",
            title="Нормальная продолжительность рабочего времени",
            text=("Статья 92. Нормальная продолжительность " "рабочего времени."),
            score=0.95,
        ),
        make_candidate(
            article_num="91",
            title="Понятие рабочего времени",
            text="Статья 91. Понятие рабочего времени.",
            score=0.80,
        ),
    ]

    result = build_context(
        candidates=candidates,
        max_sources=5,
        max_chars=12_000,
    )

    expected_context = (
        "[Источник 1 | статья 92]\n"
        "Статья 92. Нормальная продолжительность "
        "рабочего времени."
        f"{CONTEXT_SEPARATOR}"
        "[Источник 2 | статья 91]\n"
        "Статья 91. Понятие рабочего времени."
    )

    assert isinstance(result, ContextResult)
    assert result.context == expected_context

    assert result.sources == [
        ContextSource(
            article_num="92",
            title="Нормальная продолжительность рабочего времени",
            source="Трудовой кодекс Российской Федерации",
            score=0.95,
        ),
        ContextSource(
            article_num="91",
            title="Понятие рабочего времени",
            source="Трудовой кодекс Российской Федерации",
            score=0.80,
        ),
    ]


def test_build_context_preserves_reranker_order() -> None:
    candidates = [
        make_candidate(
            article_num="93",
            title="Сокращённое рабочее время",
            text="Текст статьи 93.",
            score=0.99,
        ),
        make_candidate(
            article_num="91",
            title="Понятие рабочего времени",
            text="Текст статьи 91.",
            score=0.70,
        ),
        make_candidate(
            article_num="92",
            title="Продолжительность рабочего времени",
            text="Текст статьи 92.",
            score=0.60,
        ),
    ]

    result = build_context(candidates)

    assert [source.article_num for source in result.sources] == ["93", "91", "92"]


def test_build_context_deduplicates_articles() -> None:
    candidates = [
        make_candidate(
            article_num="91",
            text="Первый и наиболее релевантный чанк статьи 91.",
            score=0.95,
        ),
        make_candidate(
            article_num="92",
            text="Текст статьи 92.",
            score=0.90,
        ),
        make_candidate(
            article_num="91",
            text="Второй чанк статьи 91.",
            score=0.80,
        ),
    ]

    result = build_context(candidates)

    assert [source.article_num for source in result.sources] == ["91", "92"]

    assert "Первый и наиболее релевантный чанк статьи 91." in result.context
    assert "Второй чанк статьи 91." not in result.context


def test_build_context_limits_source_count() -> None:
    candidates = [
        make_candidate(
            article_num="91",
            text="Текст статьи 91.",
        ),
        make_candidate(
            article_num="92",
            text="Текст статьи 92.",
        ),
        make_candidate(
            article_num="93",
            text="Текст статьи 93.",
        ),
    ]

    result = build_context(
        candidates=candidates,
        max_sources=2,
    )

    assert len(result.sources) == 2

    assert [source.article_num for source in result.sources] == ["91", "92"]

    assert "Текст статьи 93." not in result.context


def test_build_context_respects_character_limit() -> None:
    first_candidate = make_candidate(
        article_num="91",
        text="Текст статьи 91.",
    )
    second_candidate = make_candidate(
        article_num="92",
        text="Текст статьи 92.",
    )

    first_block = "[Источник 1 | статья 91]\n" "Текст статьи 91."

    result = build_context(
        candidates=[
            first_candidate,
            second_candidate,
        ],
        max_chars=len(first_block),
    )

    assert result.context == first_block

    assert [source.article_num for source in result.sources] == ["91"]


def test_build_context_skips_too_long_candidate() -> None:
    short_block = "[Источник 1 | статья 92]\n" "Короткий текст."

    candidates = [
        make_candidate(
            article_num="91",
            text="Очень длинный текст. " * 100,
            score=0.95,
        ),
        make_candidate(
            article_num="92",
            text="Короткий текст.",
            score=0.80,
        ),
    ]

    result = build_context(
        candidates=candidates,
        max_chars=len(short_block),
    )

    assert result.context == short_block

    assert [source.article_num for source in result.sources] == ["92"]


def test_build_context_strips_node_text() -> None:
    candidate = make_candidate(
        text="   Статья 91. Рабочее время.   ",
    )

    result = build_context([candidate])

    assert result.context == ("[Источник 1 | статья 91]\n" "Статья 91. Рабочее время.")


def test_build_context_returns_empty_result_for_empty_candidates() -> None:
    result = build_context([])

    assert result == ContextResult(
        context="",
        sources=[],
    )


def test_build_context_rejects_invalid_candidate_type() -> None:
    with pytest.raises(
        TypeError,
        match=("Кандидат не является объектом " "NodeWithScore!"),
    ):
        build_context(
            candidates=[
                TextNode(text="Обычный узел."),
            ],  # type: ignore[list-item]
        )


def test_build_context_rejects_empty_node_text() -> None:
    candidate = make_candidate(
        text="   ",
    )

    with pytest.raises(
        ValueError,
        match="Узел-кандидат не содержит текста!",
    ):
        build_context([candidate])


@pytest.mark.parametrize(
    "metadata_key",
    [
        "article_num",
        "title",
        "source",
    ],
)
def test_build_context_rejects_missing_metadata(
    metadata_key: str,
) -> None:
    metadata = {
        "article_num": "91",
        "title": "Понятие рабочего времени",
        "source": "Трудовой кодекс Российской Федерации",
    }
    metadata.pop(metadata_key)

    candidate = NodeWithScore(
        node=TextNode(
            text="Статья 91. Понятие рабочего времени.",
            metadata=metadata,
        ),
        score=0.9,
    )

    with pytest.raises(
        ValueError,
        match=(f"В metadata узла отсутствует " f"обязательное поле '{metadata_key}'!"),
    ):
        build_context([candidate])


@pytest.mark.parametrize(
    "metadata_key",
    [
        "article_num",
        "title",
        "source",
    ],
)
def test_build_context_rejects_empty_metadata(
    metadata_key: str,
) -> None:
    metadata = {
        "article_num": "91",
        "title": "Понятие рабочего времени",
        "source": "Трудовой кодекс Российской Федерации",
    }
    metadata[metadata_key] = "   "

    candidate = NodeWithScore(
        node=TextNode(
            text="Статья 91. Понятие рабочего времени.",
            metadata=metadata,
        ),
        score=0.9,
    )

    with pytest.raises(
        ValueError,
        match=(f"Поле metadata '{metadata_key}' " f"не должно быть пустым!"),
    ):
        build_context([candidate])


def test_build_context_rejects_missing_score() -> None:
    candidate = make_candidate(
        score=None,
    )

    with pytest.raises(
        ValueError,
        match="У кандидата отсутствует score!",
    ):
        build_context([candidate])


@pytest.mark.parametrize(
    "max_chars",
    [
        0,
        -1,
    ],
)
def test_build_context_rejects_invalid_max_chars(
    max_chars: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_chars должен быть больше 0!",
    ):
        build_context(
            candidates=[],
            max_chars=max_chars,
        )


@pytest.mark.parametrize(
    "max_sources",
    [
        0,
        -1,
    ],
)
def test_build_context_rejects_invalid_max_sources(
    max_sources: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_sources должен быть больше 0!",
    ):
        build_context(
            candidates=[],
            max_sources=max_sources,
        )
