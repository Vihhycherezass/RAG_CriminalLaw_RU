from pathlib import Path
import json

import pytest

from rag_labor_code.ingestion.serialization import save_articles_to_json
from rag_labor_code.ingestion.models import Article


def test_save_articles_to_json_saves_articles(tmp_path: Path) -> None:
    articles = [
        Article(
            article_num="91",
            title="Заголовок статьи 91",
            content="Текст статьи.",
            source="ТК РФ",
        ),
    ]

    output_path = tmp_path / "data" / "processed" / "articles.json"

    save_articles_to_json(articles=articles, output_path=output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert len(data) == 1
    assert data[0]["article_num"] == "91"
    assert data[0]["title"] == "Заголовок статьи 91"
    assert data[0]["content"] == "Текст статьи."
    assert data[0]["source"] == "ТК РФ"


def test_save_articles_to_json_creates_parent_directories(tmp_path: Path) -> None:
    articles = [
        Article(
            article_num="91",
            title="Заголовок статьи 91",
            content="Текст статьи.",
            source="ТК РФ",
        ),
    ]

    output_path = tmp_path / "data" / "processed" / "articles.json"

    assert not output_path.parent.exists()

    save_articles_to_json(articles, output_path)

    assert output_path.parent.is_dir()
    assert output_path.is_file()


def test_save_articles_to_json_rejects_non_json_path(tmp_path: Path) -> None:
    articles = [
        Article(
            article_num="91",
            title="Заголовок статьи 91",
            content="Текст статьи.",
            source="ТК РФ",
        ),
    ]

    output_path = tmp_path / "articles.txt"
    with pytest.raises(ValueError, match="Расширение файла не .json!"):
        save_articles_to_json(articles, output_path)


def test_save_articles_to_json_rejects_invalid_articles(tmp_path: Path) -> None:
    articles = [
        Article(
            article_num="",
            title="",
            content="",
            source="",
        ),
    ]

    output_path = tmp_path / "articles.json"

    with pytest.raises(
        ValueError,
        match="Проверка статей завершилась с ошибками:",
    ):
        save_articles_to_json(articles, output_path)

    assert not output_path.exists()
