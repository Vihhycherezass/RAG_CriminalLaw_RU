import pytest
from llama_index.core import Document

from rag_labor_code.ingestion.models import Article
from rag_labor_code.ingestion.document_builder import articles_to_document


def test_articles_to_documents_creates_document_for_each_article() -> None:
    articles = [
        Article(
            article_num="91",
            title="Понятие рабочего времени",
            content="Текст статьи.",
            source="ТК РФ",
        ),
        Article(
            article_num="92",
            title="Понятие рабочего времени",
            content="Текст статьи.",
            source="ТК РФ",
        ),
    ]

    assert len(articles) == 2

    res = articles_to_document(articles)

    assert len(res) == 2
    for document in res:
        assert isinstance(document, Document)


def test_articles_to_documents_builds_document_text() -> None:
    articles = [
        Article(
            article_num="91",
            title="Понятие рабочего времени",
            content="Текст статьи.",
            source="ТК РФ",
        ),
    ]

    res = articles_to_document(articles)

    assert res[0].text == "Статья 91. Понятие рабочего времени\n\nТекст статьи."


def test_articles_to_documents_preserves_metadata() -> None:
    articles = [
        Article(
            article_num="91",
            title="Понятие рабочего времени",
            content="Текст статьи.",
            source="ТК РФ",
        ),
    ]

    res = articles_to_document(articles)

    assert res[0].metadata["article_num"] == "91"
    assert res[0].metadata["title"] == "Понятие рабочего времени"
    assert res[0].metadata["source"] == "ТК РФ"
    assert res[0].metadata["document_type"] == "labor_code_article"


def test_articles_to_documents_preserves_order() -> None:
    articles = [
        Article(
            article_num="91",
            title="Понятие рабочего времени",
            content="Текст статьи.",
            source="ТК РФ",
        ),
        Article(
            article_num="92",
            title="Понятие рабочего времени",
            content="Текст статьи.",
            source="ТК РФ",
        ),
    ]

    res = articles_to_document(articles)

    assert (
        f"{res[0].metadata["article_num"]}, {res[1].metadata["article_num"]}"
        == "91, 92"
    )


def test_articles_to_documents_rejects_invalid_articles() -> None:
    with pytest.raises(ValueError):
        articles_to_document([])
