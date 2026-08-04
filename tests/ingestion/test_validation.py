from rag_labor_code.ingestion.validation import validate_articles
from rag_labor_code.ingestion.models import Article

import pytest


def test_validate_articles_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="Список статей не должен быть пустым!"):
        validate_articles(articles=[])


def test_validate_articles_rejects_non_article_element() -> None:
    with pytest.raises(ValueError) as error:
        validate_articles([1])

    message = str(error.value)

    assert "не является объектом Article!" in message


def test_validate_articles_rejects_invalid_number() -> None:
    article = Article(
        article_num="91a",
        title="Понятие рабочего времени",
        content="Текст статьи.",
        source="ТК РФ",
    )
    with pytest.raises(ValueError) as error:
        validate_articles([article])

    message = str(error.value)

    assert "не совпадает с паттерном!" in message


def test_validate_articles_rejects_duplicate_numbers() -> None:
    article1 = Article(
        article_num="91",
        title="Понятие рабочего времени",
        content="Текст статьи.",
        source="ТК РФ",
    )

    article2 = Article(
        article_num="91",
        title="Понятие рабочего времени",
        content="Текст статьи.",
        source="ТК РФ",
    )

    with pytest.raises(ValueError) as error:
        validate_articles([article1, article2])

    message = str(error.value)

    assert "повторяющиеся номера статей:" in message


def test_validate_articles_collects_multiple_errors() -> None:
    article = Article(
        article_num="91a",
        title="",
        content="",
        source="",
    )
    with pytest.raises(ValueError) as error:
        validate_articles([article])

    message = str(error.value)

    assert "не совпадает с паттерном!" in message
    assert "отсутствует заголовок!" in message
    assert "отсутствует содержание!" in message
    assert "отсутствует источник!" in message


def test_validate_articles_rejects_whitespace_only_fields() -> None:
    article = Article(
        article_num="   ",
        title="   ",
        content="   ",
        source="   ",
    )
    with pytest.raises(ValueError) as error:
        validate_articles([article])

    message = str(error.value)

    assert "отсутствует номер!" in message
    assert "отсутствует заголовок!" in message
    assert "отсутствует содержание!" in message
    assert "отсутствует источник!" in message


def test_validate_articles_accepts_valid_articles() -> None:
    article = Article(
        article_num="91",
        title="Понятие рабочего времени",
        content="Текст статьи.",
        source="ТК РФ",
    )

    result = validate_articles([article])

    assert result is None
