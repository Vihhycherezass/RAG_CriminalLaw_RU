import gradio as gr
import pytest

from rag_labor_code.pipeline.rag_pipeline import (
    RAGPipelineResult,
)
from rag_labor_code.ui import gradio_app
from rag_labor_code.ui.gradio_app import (
    EMPTY_QUESTION_MESSAGE,
    ERROR_STATUS,
    PROCESSING_ERROR_MESSAGE,
    create_gradio_app,
    handle_question,
)


class FakePipeline:
    def __init__(
        self,
        result: RAGPipelineResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[str] = []

    def answer(
        self,
        question: str,
    ) -> RAGPipelineResult:
        self.calls.append(question)

        if self.error is not None:
            raise self.error

        if self.result is None:
            raise RuntimeError("FakePipeline не получил result.")

        return self.result


@pytest.fixture
def patch_pipeline_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gradio_app,
        "RAG_PIPELINE_TYPE",
        FakePipeline,
    )


def test_handle_question_returns_presented_result(
    patch_pipeline_type: None,
) -> None:
    pipeline = FakePipeline(
        result=RAGPipelineResult(
            answer="Ответ.",
            sources=(),
            blocked=False,
            reason=None,
        )
    )

    result = handle_question(
        question="   Вопрос?   ",
        pipeline=pipeline,  # type: ignore[arg-type]
    )

    assert result == (
        "Ответ.",
        "Источники не найдены.",
        "Готово",
    )

    assert pipeline.calls == ["Вопрос?"]


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
    ],
)
def test_handle_question_rejects_empty_question(
    patch_pipeline_type: None,
    question: str,
) -> None:
    pipeline = FakePipeline(
        result=RAGPipelineResult(
            answer="Ответ.",
            sources=(),
            blocked=False,
        )
    )

    result = handle_question(
        question=question,
        pipeline=pipeline,  # type: ignore[arg-type]
    )

    assert result == (
        EMPTY_QUESTION_MESSAGE,
        "",
        ERROR_STATUS,
    )

    assert pipeline.calls == []


def test_handle_question_rejects_invalid_question_type(
    patch_pipeline_type: None,
) -> None:
    pipeline = FakePipeline()

    with pytest.raises(
        TypeError,
        match="question должен быть строкой!",
    ):
        handle_question(
            question=123,  # type: ignore[arg-type]
            pipeline=pipeline,  # type: ignore[arg-type]
        )


def test_handle_question_rejects_invalid_pipeline(
    patch_pipeline_type: None,
) -> None:
    with pytest.raises(
        TypeError,
        match=("pipeline должен быть объектом " "RAGPipeline!"),
    ):
        handle_question(
            question="Вопрос?",
            pipeline=object(),  # type: ignore[arg-type]
        )


def test_handle_question_handles_pipeline_error(
    patch_pipeline_type: None,
) -> None:
    pipeline = FakePipeline(error=RuntimeError("Внутренняя ошибка."))

    result = handle_question(
        question="Вопрос?",
        pipeline=pipeline,  # type: ignore[arg-type]
    )

    assert result == (
        PROCESSING_ERROR_MESSAGE,
        "",
        ERROR_STATUS,
    )


def test_create_gradio_app_returns_blocks(
    patch_pipeline_type: None,
) -> None:
    pipeline = FakePipeline(
        result=RAGPipelineResult(
            answer="Ответ.",
            sources=(),
            blocked=False,
        )
    )

    app = create_gradio_app(
        pipeline=pipeline,  # type: ignore[arg-type]
    )

    assert isinstance(
        app,
        gr.Blocks,
    )


def test_create_gradio_app_rejects_invalid_pipeline(
    patch_pipeline_type: None,
) -> None:
    with pytest.raises(
        TypeError,
        match=("pipeline должен быть объектом " "RAGPipeline!"),
    ):
        create_gradio_app(
            pipeline=object(),  # type: ignore[arg-type]
        )
