from pathlib import Path
from dataclasses import asdict

import json

from rag_labor_code.ingestion.validation import validate_articles
from rag_labor_code.ingestion.models import Article


def save_articles_to_json(articles: list[Article], output_path: Path) -> None:
    """Сохраняет распарсенные статьи в JSON-файл"""
    validate_articles(articles)

    if output_path.suffix.lower() != ".json":
        raise ValueError("Расширение файла не .json!")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data: list[dict] = []

    for article in articles:
        article = asdict(article)
        data.append(article)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return None
