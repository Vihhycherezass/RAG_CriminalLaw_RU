import re

from .models import Article

ARTICLE_NUMBER_PATTERN = re.compile(r"^\d+(?:\.\d+)*$")


def validate_articles(articles: list[Article]) -> None:
    """Проверяет корректность списка распарсенных статей."""
    if not articles:
        raise ValueError("Список статей не должен быть пустым!")

    errors: list[str] = []

    seen_numbers: set[str] = set()
    duplicate_numbers: set[str] = set()

    for i, article in enumerate(articles):
        if not isinstance(article, Article):
            errors.append(f"Элемент с индексом {i} не является объектом Article!")
            continue

        article.article_num.strip()
        article.title.strip()
        article.content.strip()
        article.source.strip()

        if not article.article_num:
            errors.append(f"У статьи с индексом {i} отсутствует номер!")

        if not ARTICLE_NUMBER_PATTERN.fullmatch(article.article_num):
            errors.append(f"Номер элемента с индексом {i} не совпадает с паттерном!")

        if not article.title:
            errors.append(f"У статьи {article.article_num} отсутствует заголовок!")

        if not article.content:
            errors.append(f"У статьи {article.article_num} отсутствует содержание!")

        if not article.source:
            errors.append(f"У статьи {article.article_num} отсутствует источник!")

        if article.article_num not in seen_numbers:
            seen_numbers.add(article.article_num)
        else:
            duplicate_numbers.add(article.article_num)

    if duplicate_numbers:
        errors.append(
            f"Обнаружены повторяющиеся номера статей: {', '.join(duplicate_numbers)}"
        )

    if errors:
        raise ValueError(
            "Проверка статей завершилась с ошибками:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
