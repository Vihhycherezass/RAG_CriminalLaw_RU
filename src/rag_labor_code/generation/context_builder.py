from dataclasses import dataclass

from llama_index.core.schema import NodeWithScore

CONTEXT_SEPARATOR = "\n\n---\n\n"


@dataclass(frozen=True)
class ContextSource:
    article_num: str
    title: str
    source: str
    score: float


@dataclass(frozen=True)
class ContextResult:
    context: str
    sources: list[ContextSource]


def build_context(
    candidates: list[NodeWithScore],
    max_chars: int = 12_000,
    max_sources: int = 5,
) -> ContextResult:
    """Формирует контекст и список источников для LLM."""

    if max_chars <= 0:
        raise ValueError("max_chars должен быть больше 0!")

    if max_sources <= 0:
        raise ValueError("max_sources должен быть больше 0!")

    if not candidates:
        return ContextResult(
            context="",
            sources=[],
        )

    selected_blocks: list[str] = []
    sources: list[ContextSource] = []
    seen_article_numbers: set[str] = set()

    required_metadata = (
        "article_num",
        "title",
        "source",
    )

    for candidate in candidates:
        if len(sources) >= max_sources:
            break

        if not isinstance(candidate, NodeWithScore):
            raise TypeError("Кандидат не является объектом NodeWithScore!")

        node_text = candidate.node.get_content().strip()

        if not node_text:
            raise ValueError("Узел-кандидат не содержит текста!")

        metadata = candidate.node.metadata

        for metadata_key in required_metadata:
            if metadata_key not in metadata:
                raise ValueError(
                    "В metadata узла отсутствует "
                    f"обязательное поле '{metadata_key}'!"
                )

            metadata_value = metadata[metadata_key]

            if not isinstance(metadata_value, str) or not metadata_value.strip():
                raise ValueError(
                    f"Поле metadata '{metadata_key}' " "не должно быть пустым!"
                )

        article_num = candidate.node.metadata["article_num"].strip()
        title = candidate.node.metadata["title"].strip()
        source = candidate.node.metadata["source"].strip()

        if candidate.score is None:
            raise ValueError("У кандидата отсутствует score!")

        if article_num in seen_article_numbers:
            continue

        source_number = len(sources) + 1

        block = f"[Источник {source_number} | статья {article_num}]\n" f"{node_text}"

        possible_context = CONTEXT_SEPARATOR.join(
            [
                *selected_blocks,
                block,
            ]
        )

        if len(possible_context) > max_chars:
            continue

        selected_blocks.append(block)

        sources.append(
            ContextSource(
                article_num=article_num,
                title=title,
                source=source,
                score=float(candidate.score),
            )
        )

        seen_article_numbers.add(article_num)

    context = CONTEXT_SEPARATOR.join(selected_blocks)

    return ContextResult(
        context=context,
        sources=sources,
    )
