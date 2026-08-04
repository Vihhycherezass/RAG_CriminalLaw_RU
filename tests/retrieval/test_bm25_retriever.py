from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from rag_labor_code.retrieval.bm25_retriever import (
    build_bm25_retriever,
    retrieve_bm25_nodes,
)


def create_nodes() -> list[TextNode]:
    return [
        TextNode(
            text="Статья 91. Понятие рабочего времени.",
            metadata={"article_num": "91"},
        ),
        TextNode(
            text=("Статья 92. Нормальная продолжительность " "рабочего времени."),
            metadata={"article_num": "92"},
        ),
    ]


def create_retrieved_nodes() -> list[NodeWithScore]:
    return [
        NodeWithScore(
            node=TextNode(
                text="Статья 91. Понятие рабочего времени.",
                metadata={"article_num": "91"},
            ),
            score=2.4,
        ),
        NodeWithScore(
            node=TextNode(
                text="Статья 92. Нормальная продолжительность рабочего времени.",
                metadata={"article_num": "92"},
            ),
            score=1.7,
        ),
    ]


@patch("rag_labor_code.retrieval.bm25_retriever." "BM25Retriever.from_defaults")
@patch("rag_labor_code.retrieval.bm25_retriever.Stemmer.Stemmer")
def test_build_bm25_retriever_creates_russian_retriever(
    mock_stemmer_factory: MagicMock,
    mock_from_defaults: MagicMock,
) -> None:
    nodes = create_nodes()
    russian_stemmer = MagicMock()
    expected_retriever = MagicMock(spec=BM25Retriever)

    mock_stemmer_factory.return_value = russian_stemmer
    mock_from_defaults.return_value = expected_retriever

    retriever = build_bm25_retriever(
        nodes=nodes,
        top_k=2,
    )

    mock_stemmer_factory.assert_called_once_with(
        "russian",
    )

    mock_from_defaults.assert_called_once_with(
        nodes=nodes,
        stemmer=russian_stemmer,
        language="ru",
        similarity_top_k=2,
        verbose=False,
    )

    assert retriever is expected_retriever


@patch("rag_labor_code.retrieval.bm25_retriever." "BM25Retriever.from_defaults")
@patch("rag_labor_code.retrieval.bm25_retriever.Stemmer.Stemmer")
def test_build_bm25_retriever_limits_top_k_to_node_count(
    mock_stemmer_factory: MagicMock,
    mock_from_defaults: MagicMock,
) -> None:
    nodes = create_nodes()
    russian_stemmer = MagicMock()
    expected_retriever = MagicMock(spec=BM25Retriever)

    mock_stemmer_factory.return_value = russian_stemmer
    mock_from_defaults.return_value = expected_retriever

    build_bm25_retriever(
        nodes=nodes,
        top_k=10,
    )

    mock_from_defaults.assert_called_once_with(
        nodes=nodes,
        stemmer=russian_stemmer,
        language="ru",
        similarity_top_k=2,
        verbose=False,
    )


def test_build_bm25_retriever_rejects_empty_nodes() -> None:
    with pytest.raises(
        ValueError,
        match="Список узлов пуст!",
    ):
        build_bm25_retriever(nodes=[])


def test_build_bm25_retriever_rejects_non_node_element() -> None:
    with pytest.raises(
        TypeError,
        match=("Список содержит элемент, " "не являющийся объектом BaseNode!"),
    ):
        build_bm25_retriever(
            nodes=["не узел"],  # type: ignore[list-item]
        )


def test_build_bm25_retriever_rejects_empty_node_text() -> None:
    nodes = [
        TextNode(text="   "),
    ]

    with pytest.raises(
        ValueError,
        match="Узел не содержит текста!",
    ):
        build_bm25_retriever(nodes=nodes)


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_build_bm25_retriever_rejects_invalid_top_k(
    top_k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_k должен быть больше 0!",
    ):
        build_bm25_retriever(
            nodes=create_nodes(),
            top_k=top_k,
        )


@patch("rag_labor_code.retrieval.bm25_retriever." "BM25Retriever.from_defaults")
@patch("rag_labor_code.retrieval.bm25_retriever.Stemmer.Stemmer")
def test_build_bm25_retriever_rejects_invalid_result(
    mock_stemmer_factory: MagicMock,
    mock_from_defaults: MagicMock,
) -> None:
    mock_stemmer_factory.return_value = MagicMock()
    mock_from_defaults.return_value = object()

    with pytest.raises(
        TypeError,
        match="Созданный объект не является BM25Retriever!",
    ):
        build_bm25_retriever(
            nodes=create_nodes(),
        )


def test_retrieve_bm25_nodes_retrieves_nodes() -> None:
    retriever = MagicMock(spec=BM25Retriever)
    expected_nodes = create_retrieved_nodes()

    retriever.retrieve.return_value = expected_nodes

    result = retrieve_bm25_nodes(
        retriever=retriever,
        query="рабочее время",
    )

    retriever.retrieve.assert_called_once_with(
        "рабочее время",
    )

    assert result is expected_nodes


def test_retrieve_bm25_nodes_strips_query() -> None:
    retriever = MagicMock(spec=BM25Retriever)
    retriever.retrieve.return_value = []

    retrieve_bm25_nodes(
        retriever=retriever,
        query="   рабочее время   ",
    )

    retriever.retrieve.assert_called_once_with(
        "рабочее время",
    )


def test_retrieve_bm25_nodes_does_not_add_e5_prefix() -> None:
    retriever = MagicMock(spec=BM25Retriever)
    retriever.retrieve.return_value = []

    retrieve_bm25_nodes(
        retriever=retriever,
        query="продолжительность отдыха",
    )

    retriever.retrieve.assert_called_once_with(
        "продолжительность отдыха",
    )


def test_retrieve_bm25_nodes_preserves_order_and_scores() -> None:
    retriever = MagicMock(spec=BM25Retriever)
    expected_nodes = create_retrieved_nodes()

    retriever.retrieve.return_value = expected_nodes

    result = retrieve_bm25_nodes(
        retriever=retriever,
        query="рабочее время",
    )

    assert [item.node.metadata["article_num"] for item in result] == ["91", "92"]

    assert [item.score for item in result] == [2.4, 1.7]


def test_retrieve_bm25_nodes_returns_empty_list() -> None:
    retriever = MagicMock(spec=BM25Retriever)
    retriever.retrieve.return_value = []

    result = retrieve_bm25_nodes(
        retriever=retriever,
        query="несуществующая формулировка",
    )

    assert result == []


def test_retrieve_bm25_nodes_rejects_invalid_retriever() -> None:
    with pytest.raises(
        TypeError,
        match="retriever должен быть объектом BM25Retriever!",
    ):
        retrieve_bm25_nodes(
            retriever="не retriever",  # type: ignore[arg-type]
            query="рабочее время",
        )


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
    ],
)
def test_retrieve_bm25_nodes_rejects_empty_query(
    query: str,
) -> None:
    retriever = MagicMock(spec=BM25Retriever)

    with pytest.raises(
        ValueError,
        match="Запрос пустой!",
    ):
        retrieve_bm25_nodes(
            retriever=retriever,
            query=query,
        )


def test_retrieve_bm25_nodes_rejects_invalid_result_type() -> None:
    retriever = MagicMock(spec=BM25Retriever)

    retriever.retrieve.return_value = [
        TextNode(text="Узел без NodeWithScore."),
    ]

    with pytest.raises(
        TypeError,
        match=(
            "Результат поиска содержит элемент, "
            "не являющийся объектом NodeWithScore!"
        ),
    ):
        retrieve_bm25_nodes(
            retriever=retriever,
            query="рабочее время",
        )
