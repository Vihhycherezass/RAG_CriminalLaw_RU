from pathlib import Path

import pytest

from rag_labor_code.ingestion.pdf_parser import (
    normalize_text,
    parse_articles,
    extract_text_from_pdf,
)


from rag_labor_code.ingestion.validation import validate_articles


def test_parse_articles_supports_hyphenated_article_numbers() -> None:
    text = """
Статья 341. Обычная статья
Содержание обычной статьи.

Статья 341.1-1. Организации, имеющие право на осуществление деятельности
Содержание статьи 341.1-1.

Статья 341.1-2. Договор о предоставлении труда работников
Содержание статьи 341.1-2.

Статья 348.11-1. Дополнительные основания прекращения трудового договора
Содержание статьи 348.11-1.
""".strip()

    articles = parse_articles(text)

    assert [article.article_num for article in articles] == [
        "341",
        "341.1-1",
        "341.1-2",
        "348.11-1",
    ]

    validate_articles(articles)


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
    file_path = tmp_path / "missing.pdf"

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


def test_parse_articles_collects_two_line_title() -> None:
    text = (
        "Статья 322. Порядок предоставления и соединения ежегодных\n"
        "оплачиваемых отпусков\n"
        "Ежегодный дополнительный оплачиваемый отпуск предоставляется работникам."
    )

    articles = parse_articles(text)

    assert len(articles) == 1
    assert articles[0].title == (
        "Порядок предоставления и соединения ежегодных " "оплачиваемых отпусков"
    )
    assert articles[0].content == (
        "Ежегодный дополнительный оплачиваемый отпуск " "предоставляется работникам."
    )


def test_parse_articles_collects_three_line_title() -> None:
    text = (
        "Статья 327.2. Особенности заключения трудового договора с\n"
        "работником, являющимся иностранным гражданином или\n"
        "лицом без гражданства\n"
        "Наряду со сведениями, предусмотренными статьей 57 настоящего Кодекса."
    )

    articles = parse_articles(text)

    assert articles[0].title == (
        "Особенности заключения трудового договора с "
        "работником, являющимся иностранным гражданином или "
        "лицом без гражданства"
    )
    assert articles[0].content == (
        "Наряду со сведениями, предусмотренными статьей 57 " "настоящего Кодекса."
    )


def test_parse_articles_keeps_editorial_note_in_content() -> None:
    text = (
        "Статья 324. Заключение трудового договора с лицами,\n"
        "привлекаемыми на работу в районы Крайнего Севера и\n"
        "приравненные к ним местности из других местностей (в ред.\n"
        "Федерального закона от 30.06.2006 N 90-ФЗ)\n"
        "Заключение трудового договора допускается при наличии документов."
    )

    articles = parse_articles(text)

    assert articles[0].title == (
        "Заключение трудового договора с лицами, "
        "привлекаемыми на работу в районы Крайнего Севера и "
        "приравненные к ним местности из других местностей"
    )
    assert articles[0].content == (
        "(в ред.\n"
        "Федерального закона от 30.06.2006 N 90-ФЗ)\n"
        "Заключение трудового договора допускается при наличии документов."
    )


def test_parse_articles_does_not_absorb_body_into_title() -> None:
    text = (
        "Статья 91. Понятие рабочего времени\n"
        "Рабочее время — время, в течение которого работник исполняет обязанности."
    )

    articles = parse_articles(text)

    assert articles[0].title == "Понятие рабочего времени"
    assert articles[0].content == (
        "Рабочее время — время, в течение которого " "работник исполняет обязанности."
    )
