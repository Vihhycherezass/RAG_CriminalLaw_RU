import re
from pathlib import Path

import pdfplumber

from .models import Article

ARTICLE_HEADER_PATTERN = re.compile(
    r"^Статья[ \t]+" r"(?P<number>\d+(?:\.\d+)*)\." r"[ \t]*(?P<title>[^\n]*)$",
    flags=re.MULTILINE,
)

EDITORIAL_NOTE_PATTERN = re.compile(
    r"\(в\s+ред\.",
    flags=re.IGNORECASE,
)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Извлекает текст из всех страниц PDF."""
    if not pdf_path.exists():
        raise FileNotFoundError("Файл не найден!")

    if not pdf_path.is_file():
        raise ValueError("Указанный путь не является файлом!")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Расширение файла не .pdf")

    page_texts: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                page_texts.append(page_text)
        if not page_texts:
            raise ValueError("Не удалось извлечь текст из файла!")

    return "\n\n".join(page_texts)


def normalize_text(text: str) -> str:
    """Нормализует текст, извлеченный из PDF"""
    if not text or not text.strip():
        raise ValueError("Текст не должен быть пустым!")

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = text.replace("\xa0", " ")

    text = text.replace("\u00ad", "")

    normalized_lines: list[str] = []

    for line in text.splitlines():
        normalized_line = re.sub(r"[ \t]+", " ", line).strip()
        normalized_lines.append(normalized_line)

    normalized_text = "\n".join(normalized_lines)
    normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text).strip()

    if not normalized_text:
        raise ValueError("После нормализации текст оказался пустым!")

    return normalized_text


def parse_articles(
    text: str,
    source: str = "Трудовой кодекс Российской Федерации",
) -> list[Article]:
    """Разделяет нормализованный текст на статьи."""
    if not text or not text.strip():
        raise ValueError("Текст для парсинга не должен быть пустым!")

    if not source or not source.strip():
        raise ValueError("Источник не должен быть пустым!")

    source = source.strip()

    matches = list(ARTICLE_HEADER_PATTERN.finditer(text))

    if not matches:
        raise ValueError("В тексте не найдены заголовки статей!")

    articles: list[Article] = []

    for i, match in enumerate(matches):
        article_num = match.group("number").strip()
        initial_title = match.group("title").strip()

        content_start = match.end()

        if i + 1 < len(matches):
            content_end = matches[i + 1].start()
        else:
            content_end = len(text)

        article_block = text[content_start:content_end].strip()

        title, content = split_title_and_content(
            initial_title=initial_title,
            article_block=article_block,
        )

        article = Article(
            article_num=article_num,
            title=title,
            content=content,
            source=source,
        )

        articles.append(article)

    return articles


def get_first_letter(text: str) -> str:
    """Возвращает первую букву строки."""
    for char in text:
        if char.isalpha():
            return char

    return ""


def is_title_continuation(line: str) -> bool:
    """Проверяет, является ли строка продолжением заголовка."""
    first_letter = get_first_letter(line)

    return bool(first_letter and first_letter.islower())


def split_title_and_content(
    initial_title: str,
    article_block: str,
) -> tuple[str, str]:
    """Отделяет многострочный заголовок статьи от ее содержания."""
    initial_title = initial_title.strip()
    article_block = article_block.strip()

    if not initial_title:
        return "", article_block

    initial_note_match = EDITORIAL_NOTE_PATTERN.search(initial_title)

    if initial_note_match:
        title = initial_title[: initial_note_match.start()].strip()
        editorial_note = initial_title[initial_note_match.start() :].strip()

        content_parts = [part for part in (editorial_note, article_block) if part]

        content = "\n".join(content_parts)

        return title, content

    title_parts = [initial_title]

    note_match = EDITORIAL_NOTE_PATTERN.search(article_block)

    if note_match:
        text_before_note = article_block[: note_match.start()].strip()

        lines_before_note = [
            line.strip() for line in text_before_note.splitlines() if line.strip()
        ]

        if all(is_title_continuation(line) for line in lines_before_note):
            title_parts.extend(lines_before_note)

            title = " ".join(title_parts).strip()
            content = article_block[note_match.start() :].strip()

            return title, content

    lines = article_block.splitlines()
    content_start = len(lines)

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()

        if not line:
            content_start = index + 1
            break

        if is_title_continuation(line):
            title_parts.append(line)
            continue

        content_start = index
        break

    title = " ".join(title_parts).strip()

    content = "\n".join(line.strip() for line in lines[content_start:]).strip()

    return title, content
