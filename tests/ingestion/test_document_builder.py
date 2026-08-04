import pytest
from llama_index.core import Document

from rag_labor_code.ingestion.models import Article
from rag_labor_code.ingestion.document_builder import articles_to_documents


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

    documents = articles_to_documents(articles)

    assert len(documents) == 2
    for document in documents:
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

    documents = articles_to_documents(articles)

    assert documents[0].text == "Статья 91. Понятие рабочего времени\n\nТекст статьи."


def test_articles_to_documents_preserves_metadata() -> None:
    articles = [
        Article(
            article_num="91",
            title="Понятие рабочего времени",
            content="Текст статьи.",
            source="ТК РФ",
        ),
    ]

    documents = articles_to_documents(articles)

    assert documents[0].metadata["article_num"] == "91"
    assert documents[0].metadata["title"] == "Понятие рабочего времени"
    assert documents[0].metadata["source"] == "ТК РФ"
    assert documents[0].metadata["document_type"] == "labor_code_article"


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

    documents = articles_to_documents(articles)

    assert [document.metadata["article_num"] for document in documents] == ["91", "92"]


def test_articles_to_documents_rejects_invalid_articles() -> None:
    with pytest.raises(ValueError):
        articles_to_documents([])
