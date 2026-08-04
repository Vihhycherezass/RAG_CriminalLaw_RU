import re

from rag_labor_code.generation.context_builder import (
    ContextSource,
)

ARTICLE_REFERENCE_PATTERN = re.compile(
    r"(?P<prefix>"
    r"\b(?:"
    r"статья|"
    r"статьи|"
    r"статье|"
    r"статью|"
    r"статьёй|"
    r"статьей|"
    r"ст\."
    r")"
    r"\s*"
    r"(?:№\s*)?"
    r")"
    r"(?P<number>"
    r"\d+(?:\.\d+)*(?:-\d+)?"
    r")"
    r"(?P<citation>"
    r"\s*\[Источник\s+\d+\]"
    r")?",
    flags=re.IGNORECASE,
)


SOURCE_REFERENCE_PATTERN = re.compile(
    r"\[Источник\s+\d+\]",
    flags=re.IGNORECASE,
)


def normalize_source_citations(
    answer: str,
    sources: list[ContextSource] | tuple[ContextSource, ...],
) -> str:
    """
    Нормализует ссылки на источники в ответе LLM.

    Если модель пишет естественную ссылку:

        "согласно статье 91 ТК РФ"

    она преобразуется в:

        "согласно статье 91 [Источник 1] ТК РФ"

    Если модель вообще не указала источник,
    добавляется ссылка на первый источник,
    имеющий наивысший reranking score.
    """

    if not isinstance(answer, str):
        raise TypeError("answer должен быть строкой!")

    answer = answer.strip()

    if not answer:
        raise ValueError("answer не должен быть пустым!")

    if not isinstance(
        sources,
        (list, tuple),
    ):
        raise TypeError("sources должен быть " "списком или кортежем!")

    if not sources:
        return answer

    source_numbers_by_article: dict[
        str,
        int,
    ] = {}

    for source_number, source in enumerate(
        sources,
        start=1,
    ):
        if not isinstance(
            source,
            ContextSource,
        ):
            raise TypeError("Каждый source должен быть " "объектом ContextSource!")

        article_num = source.article_num.strip()

        if not article_num:
            raise ValueError("article_num источника " "не должен быть пустым!")

        source_numbers_by_article.setdefault(
            article_num,
            source_number,
        )

    def replace_article_reference(
        match: re.Match[str],
    ) -> str:
        prefix = match.group("prefix")

        article_num = match.group("number")

        source_number = source_numbers_by_article.get(article_num)

        if source_number is None:
            return match.group(0)

        return f"{prefix}" f"{article_num} " f"[Источник {source_number}]"

    normalized_answer = ARTICLE_REFERENCE_PATTERN.sub(
        replace_article_reference,
        answer,
    )

    if SOURCE_REFERENCE_PATTERN.search(normalized_answer):
        return normalized_answer.strip()

    return f"{normalized_answer.rstrip()}\n\n" "[Источник 1]"
