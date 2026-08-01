from unittest.mock import MagicMock, patch

import pytest

from rag_labor_code.generation.context_builder import (
    ContextResult,
    ContextSource,
)
from rag_labor_code.guardrails.rules import GuardrailDecision
from rag_labor_code.pipeline.rag_pipeline import (
    NO_CONTEXT_ANSWER,
    RAGPipeline,
    RAGPipelineConfig,
    RAGPipelineResult,
)


def make_source() -> ContextSource:
    return ContextSource(
        article_num="91",
        title="Понятие рабочего времени",
        source="Трудовой кодекс Российской Федерации",
        score=0.95,
    )


def make_pipeline(
    config: RAGPipelineConfig | None = None,
) -> tuple[
    RAGPipeline,
    object,
    object,
    object,
    object,
]:
    index = object()
    bm25_retriever = object()
    reranker = object()
    llm = object()

    pipeline = RAGPipeline(
        index=index,  # type: ignore[arg-type]
        bm25_retriever=bm25_retriever,  # type: ignore[arg-type]
        reranker=reranker,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        config=config,
    )

    return (
        pipeline,
        index,
        bm25_retriever,
        reranker,
        llm,
    )


def test_pipeline_returns_successful_answer() -> None:
    (
        pipeline,
        index,
        bm25_retriever,
        reranker,
        llm,
    ) = make_pipeline()

    hybrid_candidates = [MagicMock()]
    reranked_candidates = [MagicMock()]
    source = make_source()

    with (
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "check_query_guardrails"
        ) as mock_query_guardrail,
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "retrieve_hybrid_nodes"
        ) as mock_retrieve,
        patch("rag_labor_code.pipeline.rag_pipeline." "rerank_nodes") as mock_rerank,
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "build_context"
        ) as mock_build_context,
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "generate_answer"
        ) as mock_generate,
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "check_answer_guardrails"
        ) as mock_answer_guardrail,
    ):
        mock_query_guardrail.return_value = GuardrailDecision(
            allowed=True,
            reason=None,
        )

        mock_retrieve.return_value = hybrid_candidates
        mock_rerank.return_value = reranked_candidates

        mock_build_context.return_value = ContextResult(
            context="[Источник 1 | статья 91]\nТекст статьи.",
            sources=[source],
        )

        mock_generate.return_value = (
            "Нормальная продолжительность рабочего времени "
            "не превышает 40 часов [Источник 1]."
        )

        mock_answer_guardrail.return_value = GuardrailDecision(
            allowed=True,
            reason=None,
        )

        result = pipeline.answer("Какова продолжительность рабочего времени?")

    assert result == RAGPipelineResult(
        answer=(
            "Нормальная продолжительность рабочего времени "
            "не превышает 40 часов [Источник 1]."
        ),
        sources=(source,),
        blocked=False,
        reason=None,
    )

    mock_query_guardrail.assert_called_once_with(
        query="Какова продолжительность рабочего времени?",
        max_chars=2_000,
    )

    mock_retrieve.assert_called_once_with(
        index=index,
        bm25_retriever=bm25_retriever,
        query="Какова продолжительность рабочего времени?",
        vector_top_k=10,
        final_top_k=10,
        rrf_k=60,
    )

    mock_rerank.assert_called_once_with(
        query="Какова продолжительность рабочего времени?",
        candidates=hybrid_candidates,
        reranker=reranker,
        top_k=5,
        batch_size=8,
    )

    mock_build_context.assert_called_once_with(
        candidates=reranked_candidates,
        max_chars=12_000,
        max_sources=5,
    )

    mock_generate.assert_called_once_with(
        question="Какова продолжительность рабочего времени?",
        context="[Источник 1 | статья 91]\nТекст статьи.",
        llm=llm,
        max_tokens=512,
        temperature=0.1,
        top_p=0.9,
    )

    mock_answer_guardrail.assert_called_once_with(
        answer=(
            "Нормальная продолжительность рабочего времени "
            "не превышает 40 часов [Источник 1]."
        ),
        source_count=1,
        require_sources=True,
    )


def test_pipeline_stops_when_query_is_blocked() -> None:
    pipeline, *_ = make_pipeline()

    with (
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "check_query_guardrails"
        ) as mock_query_guardrail,
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "retrieve_hybrid_nodes"
        ) as mock_retrieve,
    ):
        mock_query_guardrail.return_value = GuardrailDecision(
            allowed=False,
            reason="Запрос не относится к трудовому праву.",
        )

        result = pipeline.answer("Как приготовить борщ?")

    assert result == RAGPipelineResult(
        answer="",
        sources=(),
        blocked=True,
        reason="Запрос не относится к трудовому праву.",
    )

    mock_retrieve.assert_not_called()


def test_pipeline_returns_fallback_for_empty_context() -> None:
    pipeline, *_ = make_pipeline()

    with (
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "check_query_guardrails"
        ) as mock_query_guardrail,
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "retrieve_hybrid_nodes"
        ) as mock_retrieve,
        patch("rag_labor_code.pipeline.rag_pipeline." "rerank_nodes") as mock_rerank,
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "build_context"
        ) as mock_build_context,
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "generate_answer"
        ) as mock_generate,
    ):
        mock_query_guardrail.return_value = GuardrailDecision(
            allowed=True,
            reason=None,
        )

        mock_retrieve.return_value = []
        mock_rerank.return_value = []

        mock_build_context.return_value = ContextResult(
            context="",
            sources=[],
        )

        result = pipeline.answer("Как регулируется редкая трудовая ситуация?")

    assert result == RAGPipelineResult(
        answer=NO_CONTEXT_ANSWER,
        sources=(),
        blocked=False,
        reason=None,
    )

    mock_generate.assert_not_called()


def test_pipeline_blocks_invalid_generated_answer() -> None:
    pipeline, *_ = make_pipeline()
    source = make_source()

    with (
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "check_query_guardrails"
        ) as mock_query_guardrail,
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "retrieve_hybrid_nodes"
        ) as mock_retrieve,
        patch("rag_labor_code.pipeline.rag_pipeline." "rerank_nodes") as mock_rerank,
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "build_context"
        ) as mock_build_context,
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "generate_answer"
        ) as mock_generate,
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "check_answer_guardrails"
        ) as mock_answer_guardrail,
    ):
        mock_query_guardrail.return_value = GuardrailDecision(
            allowed=True,
            reason=None,
        )

        mock_retrieve.return_value = [MagicMock()]
        mock_rerank.return_value = [MagicMock()]

        mock_build_context.return_value = ContextResult(
            context="Контекст.",
            sources=[source],
        )

        mock_generate.return_value = "Ответ без ссылки."

        mock_answer_guardrail.return_value = GuardrailDecision(
            allowed=False,
            reason="Ответ не содержит ссылок на источники.",
        )

        result = pipeline.answer("Вопрос про рабочее время?")

    assert result == RAGPipelineResult(
        answer="",
        sources=(source,),
        blocked=True,
        reason="Ответ не содержит ссылок на источники.",
    )


def test_pipeline_passes_custom_configuration() -> None:
    config = RAGPipelineConfig(
        max_query_chars=1_000,
        retrieval_top_k=12,
        rerank_top_k=4,
        rrf_k=80,
        reranker_batch_size=2,
        max_context_chars=6_000,
        max_sources=4,
        max_tokens=300,
        temperature=0.2,
        top_p=0.8,
        require_sources=False,
    )

    (
        pipeline,
        index,
        bm25_retriever,
        reranker,
        llm,
    ) = make_pipeline(config)

    source = make_source()

    with (
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "check_query_guardrails"
        ) as mock_query_guardrail,
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "retrieve_hybrid_nodes"
        ) as mock_retrieve,
        patch("rag_labor_code.pipeline.rag_pipeline." "rerank_nodes") as mock_rerank,
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "build_context"
        ) as mock_build_context,
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "generate_answer"
        ) as mock_generate,
        patch(
            "rag_labor_code.pipeline.rag_pipeline." "check_answer_guardrails"
        ) as mock_answer_guardrail,
    ):
        mock_query_guardrail.return_value = GuardrailDecision(True)
        mock_retrieve.return_value = []
        mock_rerank.return_value = []

        mock_build_context.return_value = ContextResult(
            context="Контекст.",
            sources=[source],
        )

        mock_generate.return_value = "Ответ."
        mock_answer_guardrail.return_value = GuardrailDecision(True)

        pipeline.answer("Вопрос про отпуск?")

    mock_query_guardrail.assert_called_once_with(
        query="Вопрос про отпуск?",
        max_chars=1_000,
    )

    mock_retrieve.assert_called_once_with(
        index=index,
        bm25_retriever=bm25_retriever,
        query="Вопрос про отпуск?",
        vector_top_k=12,
        final_top_k=12,
        rrf_k=80,
    )

    mock_rerank.assert_called_once_with(
        query="Вопрос про отпуск?",
        candidates=[],
        reranker=reranker,
        top_k=4,
        batch_size=2,
    )

    mock_build_context.assert_called_once_with(
        candidates=[],
        max_chars=6_000,
        max_sources=4,
    )

    mock_generate.assert_called_once_with(
        question="Вопрос про отпуск?",
        context="Контекст.",
        llm=llm,
        max_tokens=300,
        temperature=0.2,
        top_p=0.8,
    )

    mock_answer_guardrail.assert_called_once_with(
        answer="Ответ.",
        source_count=1,
        require_sources=False,
    )


def test_pipeline_rejects_invalid_config_type() -> None:
    with pytest.raises(
        TypeError,
        match="config должен быть объектом RAGPipelineConfig!",
    ):
        RAGPipeline(
            index=object(),  # type: ignore[arg-type]
            bm25_retriever=object(),  # type: ignore[arg-type]
            reranker=object(),  # type: ignore[arg-type]
            llm=object(),  # type: ignore[arg-type]
            config="config",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "max_query_chars",
        "retrieval_top_k",
        "rerank_top_k",
        "rrf_k",
        "reranker_batch_size",
        "max_context_chars",
        "max_sources",
        "max_tokens",
    ],
)
def test_config_rejects_non_positive_integer(
    field_name: str,
) -> None:
    values = {
        field_name: 0,
    }

    with pytest.raises(
        ValueError,
        match=f"{field_name} должен быть больше 0!",
    ):
        RAGPipelineConfig(**values)


@pytest.mark.parametrize(
    "temperature",
    [
        -0.1,
        2.1,
    ],
)
def test_config_rejects_invalid_temperature(
    temperature: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="temperature должна находиться от 0 до 2!",
    ):
        RAGPipelineConfig(
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
def test_config_rejects_invalid_top_p(
    top_p: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_p должен быть больше 0 и не больше 1!",
    ):
        RAGPipelineConfig(
            top_p=top_p,
        )


def test_config_rejects_invalid_require_sources_type() -> None:
    with pytest.raises(
        TypeError,
        match="require_sources должен быть bool!",
    ):
        RAGPipelineConfig(
            require_sources=1,  # type: ignore[arg-type]
        )
