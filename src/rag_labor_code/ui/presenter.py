from rag_labor_code.generation.context_builder import ContextSource
from rag_labor_code.pipeline.rag_pipeline import RAGPipelineResult

SUCCESS_STATUS = "Готово"
BLOCKED_STATUS = "Заблокировано"

NO_SOURCES_TEXT = "Источники не найдены."

CONTEXT_SOURCE_TYPE = ContextSource
RAG_PIPELINE_RESULT_TYPE = RAGPipelineResult

BLOCKED_FALLBACK_TEXT = "Запрос или ответ был заблокирован системой безопасности."


def format_sources(
    sources: tuple[ContextSource, ...],
) -> str:

    if not isinstance(sources, tuple):
        raise TypeError("sources должен быть tuple!")

    if not sources:
        return NO_SOURCES_TEXT

    parts: list[str] = []

    for index, source in enumerate(sources, start=1):
        if not isinstance(source, CONTEXT_SOURCE_TYPE):
            raise TypeError("Каждый элемент sources должен быть ContextSource!")

        part = (
            f"{index}. **Статья {source.article_num} — {source.title}**\n"
            f"   {source.source}"
        )

        parts.append(part)

    return "\n\n".join(parts)


def present_pipeline_result(
    result: RAGPipelineResult,
) -> tuple[str, str, str]:
    if not isinstance(result, RAG_PIPELINE_RESULT_TYPE):
        raise TypeError("result должен быть объектом RAGPipelineResult!")

    if result.blocked:
        message = result.reason or BLOCKED_FALLBACK_TEXT

        return (
            message,
            "",
            BLOCKED_STATUS,
        )

    sources_text = format_sources(result.sources)

    return (
        result.answer,
        sources_text,
        SUCCESS_STATUS,
    )
