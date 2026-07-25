from pathlib import Path

import pytest

from rag_labor_code.ingestion.pdf_parser import (
    normalize_text,
    parse_articles,
    extract_text_from_pdf,
)


def test_normalize_text_normalizes_whitespace() -> None:
    raw_text = (
        "  Статья 91.\xa0Рабочее   время\r\n" "\r\n" "\r\n" "\tТекст статьи.\u00ad"
    )

    result = normalize_text(raw_text)

    assert result == "Статья 91. Рабочее время\n\nТекст статьи."


def test_normalize_text_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="Текст не должен быть пустым!"):
        normalize_text("   ")


def test_normalize_text_rejects_soft_hyphen_only_text() -> None:
    with pytest.raises(ValueError, match="После нормализации текст оказался пустым!"):
        normalize_text("\u00ad")


def test_parse_articles_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="Текст для парсинга не должен быть пустым!"):
        parse_articles("   ")


def test_parse_articles_rejects_empty_source() -> None:
    text = (
        "Статья 1. Первая статья\n"
        "Содержание первой статьи.\n\n"
        "Статья 2.1. Вторая статья\n"
        "Содержание второй статьи."
    )

    with pytest.raises(ValueError, match="Источник не должен быть пустым!"):
        parse_articles(text, source="  ")


def test_parse_articles_rejects_text_without_headers() -> None:
    with pytest.raises(ValueError, match="В тексте не найдены заголовки статей"):
        parse_articles("Нет заголовка")


def test_parse_articles_parses_multiple_articles() -> None:
    text = (
        "Статья 1. Первая статья\n"
        "Содержание первой статьи.\n\n"
        "Статья 2.1. Вторая статья\n"
        "Содержание второй статьи."
    )

    articles = parse_articles(text)

    assert len(articles) == 2
    assert articles[0].article_num == "1"
    assert articles[0].title == "Первая статья"
    assert articles[0].content == "Содержание первой статьи."

    assert articles[1].article_num == "2.1"
    assert articles[1].title == "Вторая статья"


def test_parse_articles_keeps_content_separate_from_empty_title() -> None:
    text = "Статья 1.\n" "Первый абзац содержания."
    articles = parse_articles(text)

    assert articles[0].title == ""
    assert articles[0].content == "Первый абзац содержания."


def test_parse_articles_supports_compound_article_numbers() -> None:
    text = "Статья 53.1. Особенности регулирования\n" "Содержание статьи."

    articles = parse_articles(text)

    assert len(articles) == 1
    assert articles[0].article_num == "53.1"
    assert articles[0].title == "Особенности регулирования"
    assert articles[0].content == "Содержание статьи."


def test_extract_text_from_pdf_rejects_missing_file(tmp_path: Path) -> None:
    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="Файл не найден!"):
        extract_text_from_pdf(file_path)


def test_extract_text_from_pdf_rejects_directory(tmp_path: Path) -> None:
    directory_path = tmp_path / "directory"
    directory_path.mkdir()

    with pytest.raises(ValueError, match="Указанный путь не является файлом!"):
        extract_text_from_pdf(directory_path)


def test_extract_text_from_pdf_rejects_non_pdf_file(tmp_path: Path) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("Текст", encoding="utf-8")

    with pytest.raises(ValueError, match="Расширение файла не .pdf"):
        extract_text_from_pdf(file_path)
