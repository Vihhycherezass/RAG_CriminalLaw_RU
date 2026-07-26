from unittest.mock import MagicMock

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode

from rag_labor_code.embeddings.e5_embedder import (
    prepare_nodes,
    prepare_passage,
    prepare_query,
)


def test_prepare_passage_adds_prefix() -> None:
    text = prepare_passage("текст")

    assert text == "passage: текст"


def test_prepare_passage_does_not_duplicate_prefix() -> None:
    text = prepare_passage("passage: текст")

    assert text == "passage: текст"


def test_prepare_passage_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="Текст пустой!"):
        prepare_passage("   ")


def test_prepare_query_adds_prefix() -> None:
    text = prepare_query("текст")

    assert text == "query: текст"


def test_prepare_query_does_not_duplicate_prefix() -> None:
    text = prepare_query("query: текст")

    assert text == "query: текст"


def test_prepare_query_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="Запрос пустой!"):
        prepare_query("   ")


def test_embed_nodes_assigns_embedding_to_each_node() -> None:
    nodes = [
        TextNode(text="Текст статьи 91."),
        TextNode(text="Текст статьи 92."),
    ]

    expected_embeddings = [
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
    ]

    embed_model = MagicMock(spec=BaseEmbedding)
    embed_model.get_text_embedding_batch.return_value = expected_embeddings

    prepared_nodes = prepare_nodes(
        nodes=nodes,
        embed_model=embed_model,
    )

    assert prepared_nodes[0].embedding == expected_embeddings[0]
    assert prepared_nodes[1].embedding == expected_embeddings[1]


def test_embed_nodes_passes_prefixed_texts_to_model() -> None:
    nodes = [
        TextNode(text="Текст статьи 91."),
        TextNode(text="Текст статьи 92."),
    ]

    embed_model = MagicMock(spec=BaseEmbedding)
    embed_model.get_text_embedding_batch.return_value = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    prepare_nodes(
        nodes=nodes,
        embed_model=embed_model,
    )

    embed_model.get_text_embedding_batch.assert_called_once_with(
        texts=[
            "passage: Текст статьи 91.",
            "passage: Текст статьи 92.",
        ],
        show_progress=False,
    )


def test_embed_nodes_preserves_metadata() -> None:
    nodes = [
        TextNode(
            text="Текст статьи 91.",
            metadata={
                "article_num": "91",
                "title": "Понятие рабочего времени",
                "source": "ТК РФ",
                "document_type": "labor_code_article",
            },
        )
    ]

    embed_model = MagicMock(spec=BaseEmbedding)
    embed_model.get_text_embedding_batch.return_value = [[1.0, 0.0, 0.5]]

    prepared_nodes = prepare_nodes(
        nodes=nodes,
        embed_model=embed_model,
    )

    assert prepared_nodes[0].metadata == {
        "article_num": "91",
        "title": "Понятие рабочего времени",
        "source": "ТК РФ",
        "document_type": "labor_code_article",
    }


def test_embed_nodes_preserves_order() -> None:
    nodes = [
        TextNode(
            text="Текст статьи 91.",
            metadata={"article_num": "91"},
        ),
        TextNode(
            text="Текст статьи 92.",
            metadata={"article_num": "92"},
        ),
    ]

    embed_model = MagicMock(spec=BaseEmbedding)
    embed_model.get_text_embedding_batch.return_value = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    prepared_nodes = prepare_nodes(
        nodes=nodes,
        embed_model=embed_model,
    )

    assert [node.metadata["article_num"] for node in prepared_nodes] == ["91", "92"]

    assert prepared_nodes[0].embedding == [1.0, 0.0]
    assert prepared_nodes[1].embedding == [0.0, 1.0]


def test_embed_nodes_rejects_empty_nodes() -> None:
    embed_model = MagicMock(spec=BaseEmbedding)

    with pytest.raises(
        ValueError,
        match="Список с узлами пуст!",
    ):
        prepare_nodes(
            nodes=[],
            embed_model=embed_model,
        )


def test_embed_nodes_rejects_non_node_element() -> None:
    embed_model = MagicMock(spec=BaseEmbedding)

    with pytest.raises(
        TypeError,
        match="Список содержит элемент, не являющийся объектом BaseNode!",
    ):
        prepare_nodes(
            nodes=["не узел"],  # type: ignore[list-item]
            embed_model=embed_model,
        )


def test_embed_nodes_rejects_empty_node_text() -> None:
    nodes = [
        TextNode(text="   "),
    ]

    embed_model = MagicMock(spec=BaseEmbedding)

    with pytest.raises(
        ValueError,
        match="Узел не содержит текста!",
    ):
        prepare_nodes(
            nodes=nodes,
            embed_model=embed_model,
        )


def test_embed_nodes_rejects_embedding_count_mismatch() -> None:
    nodes = [
        TextNode(text="Текст статьи 91."),
        TextNode(text="Текст статьи 92."),
    ]

    embed_model = MagicMock(spec=BaseEmbedding)

    # Передали два узла, но модель вернула только один embedding.
    embed_model.get_text_embedding_batch.return_value = [
        [1.0, 0.0],
    ]

    with pytest.raises(
        ValueError,
        match="Количество embeddings не совпадает с количеством узлов!",
    ):
        prepare_nodes(
            nodes=nodes,
            embed_model=embed_model,
        )
