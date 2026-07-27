from unittest.mock import MagicMock

import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode

from rag_labor_code.retrieval.vector_retriever import (
    retrieve_vector_nodes,
)


def create_retrieved_nodes() -> list[NodeWithScore]:
    return [
        NodeWithScore(
            node=TextNode(
                text="Статья 91. Понятие рабочего времени.",
                metadata={"article_num": "91"},
            ),
            score=0.95,
        ),
        NodeWithScore(
            node=TextNode(
                text="Статья 92. Нормальная продолжительность рабочего времени.",
                metadata={"article_num": "92"},
            ),
            score=0.87,
        ),
    ]


def test_retrieve_vector_nodes_retrieves_top_k_nodes() -> None:
    index = MagicMock(spec=VectorStoreIndex)
    retriever = MagicMock()

    expected_nodes = create_retrieved_nodes()

    index.as_retriever.return_value = retriever
    retriever.retrieve.return_value = expected_nodes

    result = retrieve_vector_nodes(
        index=index,
        query="рабочее время",
        top_k=2,
    )

    index.as_retriever.assert_called_once_with(
        similarity_top_k=2,
    )

    retriever.retrieve.assert_called_once_with(
        "query: рабочее время",
    )

    assert result is expected_nodes


def test_retrieve_vector_nodes_does_not_duplicate_query_prefix() -> None:
    index = MagicMock(spec=VectorStoreIndex)
    retriever = MagicMock()

    index.as_retriever.return_value = retriever
    retriever.retrieve.return_value = []

    retrieve_vector_nodes(
        index=index,
        query="query: рабочее время",
        top_k=3,
    )

    retriever.retrieve.assert_called_once_with(
        "query: рабочее время",
    )


def test_retrieve_vector_nodes_preserves_order_and_scores() -> None:
    index = MagicMock(spec=VectorStoreIndex)
    retriever = MagicMock()

    expected_nodes = create_retrieved_nodes()

    index.as_retriever.return_value = retriever
    retriever.retrieve.return_value = expected_nodes

    result = retrieve_vector_nodes(
        index=index,
        query="продолжительность рабочего времени",
        top_k=2,
    )

    assert [item.node.metadata["article_num"] for item in result] == ["91", "92"]

    assert [item.score for item in result] == [
        0.95,
        0.87,
    ]


def test_retrieve_vector_nodes_returns_empty_list() -> None:
    index = MagicMock(spec=VectorStoreIndex)
    retriever = MagicMock()

    index.as_retriever.return_value = retriever
    retriever.retrieve.return_value = []

    result = retrieve_vector_nodes(
        index=index,
        query="несуществующий запрос",
    )

    assert result == []


def test_retrieve_vector_nodes_rejects_invalid_index() -> None:
    with pytest.raises(
        TypeError,
        match="index должен быть объектом VectorStoreIndex!",
    ):
        retrieve_vector_nodes(
            index="не индекс",  # type: ignore[arg-type]
            query="рабочее время",
        )


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
    ],
)
def test_retrieve_vector_nodes_rejects_empty_query(
    query: str,
) -> None:
    index = MagicMock(spec=VectorStoreIndex)

    with pytest.raises(
        ValueError,
        match="Запрос пустой!",
    ):
        retrieve_vector_nodes(
            index=index,
            query=query,
        )


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_retrieve_vector_nodes_rejects_invalid_top_k(
    top_k: int,
) -> None:
    index = MagicMock(spec=VectorStoreIndex)

    with pytest.raises(
        ValueError,
        match="top_k должен быть больше 0!",
    ):
        retrieve_vector_nodes(
            index=index,
            query="рабочее время",
            top_k=top_k,
        )


def test_retrieve_vector_nodes_rejects_invalid_result_type() -> None:
    index = MagicMock(spec=VectorStoreIndex)
    retriever = MagicMock()

    index.as_retriever.return_value = retriever
    retriever.retrieve.return_value = [
        TextNode(text="Обычный узел без NodeWithScore."),
    ]

    with pytest.raises(
        TypeError,
        match=(
            "Результат поиска содержит элемент, "
            "не являющийся объектом NodeWithScore!"
        ),
    ):
        retrieve_vector_nodes(
            index=index,
            query="рабочее время",
        )
