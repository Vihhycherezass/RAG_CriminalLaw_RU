import asyncio
from types import SimpleNamespace

import rag_labor_code.pipeline.rag_pipeline as pipeline_module
from rag_labor_code.pipeline.rag_pipeline import (
    RAGPipeline,
    RAGPipelineConfig,
)


class FakeNemoGuardrails:
    def __init__(self) -> None:
        self.input_calls: list[str] = []
        self.output_calls: list[tuple[str, str]] = []

    async def check_input_async(
        self,
        question: str,
    ):
        self.input_calls.append(question)

        return SimpleNamespace(
            allowed=True,
            content=question,
            modified=False,
            reason=None,
        )

    async def check_output_async(
        self,
        question: str,
        answer: str,
    ):
        self.output_calls.append(
            (
                question,
                answer,
            )
        )

        return SimpleNamespace(
            allowed=True,
            content=answer,
            modified=False,
            reason=None,
        )


def test_answer_async_uses_async_nemo_checks(
    monkeypatch,
) -> None:
    pipeline = object.__new__(RAGPipeline)

    pipeline._index = object()
    pipeline._bm25_retriever = object()
    pipeline._reranker = object()
    pipeline._llm = object()
    pipeline._config = RAGPipelineConfig(
        require_sources=False,
    )

    nemo = FakeNemoGuardrails()
    pipeline._nemo_guardrails = nemo

    monkeypatch.setattr(
        pipeline_module,
        "check_query_guardrails",
        lambda **kwargs: SimpleNamespace(
            allowed=True,
            reason=None,
        ),
    )

    monkeypatch.setattr(
        pipeline_module,
        "retrieve_hybrid_nodes",
        lambda **kwargs: ["candidate"],
    )

    monkeypatch.setattr(
        pipeline_module,
        "rerank_nodes",
        lambda **kwargs: ["reranked"],
    )

    source = SimpleNamespace(
        article_num="91",
        title="Понятие рабочего времени",
        source="ТК РФ",
        score=1.0,
    )

    monkeypatch.setattr(
        pipeline_module,
        "build_context",
        lambda **kwargs: SimpleNamespace(
            context="Контекст статьи 91",
            sources=[source],
        ),
    )

    monkeypatch.setattr(
        pipeline_module,
        "generate_answer",
        lambda **kwargs: "Нормальная продолжительность — 40 часов.",
    )

    monkeypatch.setattr(
        pipeline_module,
        "check_answer_guardrails",
        lambda **kwargs: SimpleNamespace(
            allowed=True,
            reason=None,
        ),
    )

    question = "Какова нормальная продолжительность " "рабочего времени в неделю?"

    result = asyncio.run(pipeline.answer_async(question))

    assert result.blocked is False
    assert result.reason is None
    assert result.answer == "Нормальная продолжительность — 40 часов."

    assert nemo.input_calls == [question]

    assert nemo.output_calls == [
        (
            question,
            "Нормальная продолжительность — 40 часов.",
        )
    ]
