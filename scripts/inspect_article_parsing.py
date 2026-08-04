from argparse import ArgumentParser
from pathlib import Path

from rag_labor_code.ingestion.pdf_parser import (
    extract_text_from_pdf,
    normalize_text,
    parse_articles,
)


def get_first_content_line(content: str) -> str:
    """Возвращает первую непустую строку содержания статьи."""
    for line in content.splitlines():
        line = line.strip()

        if line:
            return line

    return ""


def find_suspicious_articles(pdf_path: Path) -> None:
    """Выводит статьи с возможным переносом заголовка в content."""
    raw_text = extract_text_from_pdf(pdf_path)
    normalized_text = normalize_text(raw_text)
    articles = parse_articles(normalized_text)

    article_numbers_to_check = {
        "322",
        "324",
        "325",
        "327.2",
        "327.3",
        "327.4",
    }

    for article in articles:
        if article.article_num in article_numbers_to_check:
            print("=" * 80)
            print(f"Статья: {article.article_num}")
            print(f"TITLE: {article.title}")
            print(f"CONTENT START: {article.content[:500]}")

    print(f"Всего найдено статей: {len(articles)}")

    if articles:
        print(f"Первая статья: {articles[0].article_num}")
        print(f"Последняя статья: {articles[-1].article_num}")

    print("\nПодозрительные статьи:\n")

    suspicious_count = 0

    for article in articles:
        first_content_line = get_first_content_line(article.content)
        reasons: list[str] = []

        if not article.title.strip():
            reasons.append("пустой заголовок")

        if len(article.title.strip()) < 5:
            reasons.append("слишком короткий заголовок")

        if first_content_line and first_content_line[0].islower():
            reasons.append("первая строка content начинается со строчной буквы")

        if not reasons:
            continue

        suspicious_count += 1

        print("=" * 80)
        print(f"Статья: {article.article_num}")
        print(f"Причины: {', '.join(reasons)}")
        print(f"TITLE: {article.title!r}")
        print(f"Первые строки CONTENT: {article.content[:400]!r}")
        print()

    print("=" * 80)
    print(f"Подозрительных статей найдено: {suspicious_count}")


def main() -> None:
    parser = ArgumentParser(description="Проверка качества парсинга заголовков статей.")
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Путь к PDF-файлу Трудового кодекса.",
    )

    args = parser.parse_args()

    find_suspicious_articles(args.pdf_path)


if __name__ == "__main__":
    main()
