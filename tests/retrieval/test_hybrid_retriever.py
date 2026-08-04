from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from rag_labor_code.retrieval.hybrid_retriever import (
    reciprocal_rank_fusion,
    retrieve_hybrid_nodes,
)


def create_nodes() -> dict[str, TextNode]:
    return {
        "91": TextNode(
            text="Статья 91. Понятие рабочего времени.",
            metadata={"article_num": "91"},
        ),
        "92": TextNode(
            text=("Статья 92. Нормальная продолжительность " "рабочего времени."),
            metadata={"article_num": "92"},
        ),
        "93": TextNode(
            text="Статья 93. Сокращённая продолжительность рабочего времени.",
            metadata={"article_num": "93"},
        ),
        "94": TextNode(
            text="Статья 94. Продолжительность ежедневной работы.",
            metadata={"article_num": "94"},
        ),
    }


def make_result(
    node: TextNode,
    score: float,
) -> NodeWithScore:
    return NodeWithScore(
        node=node,
        score=score,
    )


def test_reciprocal_rank_fusion_combines_rankings() -> None:
    nodes = create_nodes()

    vector_results = [
        make_result(nodes["91"], 0.98),
        make_result(nodes["92"], 0.91),
        make_result(nodes["93"], 0.84),
    ]

    bm25_results = [
        make_result(nodes["92"], 4.8),
        make_result(nodes["94"], 3.7),
        make_result(nodes["91"], 2.9),
    ]

    result = reciprocal_rank_fusion(
        result_lists=[
            vector_results,
            bm25_results,
        ],
        top_k=4,
        rrf_k=60,
    )

    assert [item.node.metadata["article_num"] for item in result] == [
        "92",
        "91",
        "94",
        "93",
    ]


def test_reciprocal_rank_fusion_calculates_correct_scores() -> None:
    nodes = create_nodes()

    vector_results = [
        make_result(nodes["91"], 0.98),
        make_result(nodes["92"], 0.91),
        make_result(nodes["93"], 0.84),
    ]

    bm25_results = [
        make_result(nodes["92"], 4.8),
        make_result(nodes["94"], 3.7),
        make_result(nodes["91"], 2.9),
    ]

    result = reciprocal_rank_fusion(
        result_lists=[
            vector_results,
            bm25_results,
        ],
        top_k=4,
        rrf_k=60,
    )

    scores_by_article = {
        item.node.metadata["article_num"]: item.score for item in result
    }

    assert scores_by_article["91"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores_by_article["92"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores_by_article["93"] == pytest.approx(1 / 63)
    assert scores_by_article["94"] == pytest.approx(1 / 62)


def test_reciprocal_rank_fusion_ignores_original_scores() -> None:
    nodes = create_nodes()

    first_results = [
        make_result(nodes["91"], 0.01),
        make_result(nodes["92"], 1000.0),
    ]

    second_results = [
        make_result(nodes["91"], -500.0),
        make_result(nodes["92"], 99999.0),
    ]

    result = reciprocal_rank_fusion(
        result_lists=[
            first_results,
            second_results,
        ],
        top_k=2,
        rrf_k=60,
    )

    assert result[0].node.metadata["article_num"] == "91"
    assert result[0].score == pytest.approx(1 / 61 + 1 / 61)

    assert result[1].node.metadata["article_num"] == "92"
    assert result[1].score == pytest.approx(1 / 62 + 1 / 62)


def test_reciprocal_rank_fusion_deduplicates_nodes() -> None:
    nodes = create_nodes()

    vector_results = [
        make_result(nodes["91"], 0.98),
        make_result(nodes["91"], 0.95),
        make_result(nodes["92"], 0.90),
    ]

    result = reciprocal_rank_fusion(
        result_lists=[vector_results],
        top_k=5,
        rrf_k=60,
    )

    assert [item.node.metadata["article_num"] for item in result] == ["91", "92"]

    assert result[0].score == pytest.approx(1 / 61)
    assert result[1].score == pytest.approx(1 / 62)


def test_reciprocal_rank_fusion_limits_result_count() -> None:
    nodes = create_nodes()

    results = [
        make_result(nodes["91"], 1.0),
        make_result(nodes["92"], 0.9),
        make_result(nodes["93"], 0.8),
    ]

    fused_results = reciprocal_rank_fusion(
        result_lists=[results],
        top_k=2,
    )

    assert len(fused_results) == 2


def test_reciprocal_rank_fusion_handles_empty_inner_lists() -> None:
    nodes = create_nodes()

    vector_results = [
        make_result(nodes["91"], 0.9),
    ]

    result = reciprocal_rank_fusion(
        result_lists=[
            vector_results,
            [],
        ],
    )

    assert len(result) == 1
    assert result[0].node.metadata["article_num"] == "91"


def test_reciprocal_rank_fusion_returns_empty_list_when_all_results_empty() -> None:
    result = reciprocal_rank_fusion(
        result_lists=[
            [],
            [],
        ],
    )

    assert result == []


def test_reciprocal_rank_fusion_rejects_empty_result_lists() -> None:
    with pytest.raises(
        ValueError,
        match="Список поисковых выдач пуст!",
    ):
        reciprocal_rank_fusion(
            result_lists=[],
        )


def test_reciprocal_rank_fusion_rejects_invalid_result_type() -> None:
    invalid_results = [
        [
            TextNode(text="Обычный узел."),
        ]
    ]

    with pytest.raises(
        TypeError,
        match=(
            "Поисковая выдача содержит элемент, "
            "не являющийся объектом NodeWithScore!"
        ),
    ):
        reciprocal_rank_fusion(
            result_lists=invalid_results,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_reciprocal_rank_fusion_rejects_invalid_top_k(
    top_k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_k должен быть больше 0!",
    ):
        reciprocal_rank_fusion(
            result_lists=[[]],
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "rrf_k",
    [
        0,
        -1,
    ],
)
def test_reciprocal_rank_fusion_rejects_invalid_rrf_k(
    rrf_k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="rrf_k должен быть больше 0!",
    ):
        reciprocal_rank_fusion(
            result_lists=[[]],
            rrf_k=rrf_k,
        )


@patch("rag_labor_code.retrieval.hybrid_retriever." "reciprocal_rank_fusion")
@patch("rag_labor_code.retrieval.hybrid_retriever." "retrieve_bm25_nodes")
@patch("rag_labor_code.retrieval.hybrid_retriever." "retrieve_vector_nodes")
def test_retrieve_hybrid_nodes_combines_retrieval_results(
    mock_retrieve_vector_nodes: MagicMock,
    mock_retrieve_bm25_nodes: MagicMock,
    mock_rrf: MagicMock,
) -> None:
    nodes = create_nodes()

    vector_results = [
        make_result(nodes["91"], 0.95),
    ]
    bm25_results = [
        make_result(nodes["92"], 3.5),
    ]
    expected_results = [
        make_result(nodes["91"], 0.03),
        make_result(nodes["92"], 0.02),
    ]

    index = MagicMock(spec=VectorStoreIndex)
    bm25_retriever = MagicMock(spec=BM25Retriever)

    mock_retrieve_vector_nodes.return_value = vector_results
    mock_retrieve_bm25_nodes.return_value = bm25_results
    mock_rrf.return_value = expected_results

    result = retrieve_hybrid_nodes(
        index=index,
        bm25_retriever=bm25_retriever,
        query="рабочее время",
        vector_top_k=10,
        final_top_k=5,
        rrf_k=60,
    )

    mock_retrieve_vector_nodes.assert_called_once_with(
        index=index,
        query="рабочее время",
        top_k=10,
    )

    mock_retrieve_bm25_nodes.assert_called_once_with(
        retriever=bm25_retriever,
        query="рабочее время",
    )

    mock_rrf.assert_called_once_with(
        result_lists=[
            vector_results,
            bm25_results,
        ],
        top_k=5,
        rrf_k=60,
    )

    assert result is expected_results


@patch("rag_labor_code.retrieval.hybrid_retriever." "reciprocal_rank_fusion")
@patch("rag_labor_code.retrieval.hybrid_retriever." "retrieve_bm25_nodes")
@patch("rag_labor_code.retrieval.hybrid_retriever." "retrieve_vector_nodes")
def test_retrieve_hybrid_nodes_strips_query(
    mock_retrieve_vector_nodes: MagicMock,
    mock_retrieve_bm25_nodes: MagicMock,
    mock_rrf: MagicMock,
) -> None:
    index = MagicMock(spec=VectorStoreIndex)
    bm25_retriever = MagicMock(spec=BM25Retriever)

    mock_retrieve_vector_nodes.return_value = []
    mock_retrieve_bm25_nodes.return_value = []
    mock_rrf.return_value = []

    retrieve_hybrid_nodes(
        index=index,
        bm25_retriever=bm25_retriever,
        query="   рабочее время   ",
    )

    mock_retrieve_vector_nodes.assert_called_once_with(
        index=index,
        query="рабочее время",
        top_k=10,
    )

    mock_retrieve_bm25_nodes.assert_called_once_with(
        retriever=bm25_retriever,
        query="рабочее время",
    )


def test_retrieve_hybrid_nodes_rejects_invalid_index() -> None:
    bm25_retriever = MagicMock(spec=BM25Retriever)

    with pytest.raises(
        TypeError,
        match="index должен быть объектом VectorStoreIndex!",
    ):
        retrieve_hybrid_nodes(
            index="не индекс",  # type: ignore[arg-type]
            bm25_retriever=bm25_retriever,
            query="рабочее время",
        )


def test_retrieve_hybrid_nodes_rejects_invalid_bm25_retriever() -> None:
    index = MagicMock(spec=VectorStoreIndex)

    with pytest.raises(
        TypeError,
        match=("bm25_retriever должен быть объектом " "BM25Retriever!"),
    ):
        retrieve_hybrid_nodes(
            index=index,
            bm25_retriever="не retriever",  # type: ignore[arg-type]
            query="рабочее время",
        )


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
    ],
)
def test_retrieve_hybrid_nodes_rejects_empty_query(
    query: str,
) -> None:
    index = MagicMock(spec=VectorStoreIndex)
    bm25_retriever = MagicMock(spec=BM25Retriever)

    with pytest.raises(
        ValueError,
        match="Запрос пустой!",
    ):
        retrieve_hybrid_nodes(
            index=index,
            bm25_retriever=bm25_retriever,
            query=query,
        )


@pytest.mark.parametrize(
    "vector_top_k",
    [
        0,
        -1,
    ],
)
def test_retrieve_hybrid_nodes_rejects_invalid_vector_top_k(
    vector_top_k: int,
) -> None:
    index = MagicMock(spec=VectorStoreIndex)
    bm25_retriever = MagicMock(spec=BM25Retriever)

    with pytest.raises(
        ValueError,
        match="vector_top_k должен быть больше 0!",
    ):
        retrieve_hybrid_nodes(
            index=index,
            bm25_retriever=bm25_retriever,
            query="рабочее время",
            vector_top_k=vector_top_k,
        )


@pytest.mark.parametrize(
    "final_top_k",
    [
        0,
        -1,
    ],
)
def test_retrieve_hybrid_nodes_rejects_invalid_final_top_k(
    final_top_k: int,
) -> None:
    index = MagicMock(spec=VectorStoreIndex)
    bm25_retriever = MagicMock(spec=BM25Retriever)

    with pytest.raises(
        ValueError,
        match="final_top_k должен быть больше 0!",
    ):
        retrieve_hybrid_nodes(
            index=index,
            bm25_retriever=bm25_retriever,
            query="рабочее время",
            final_top_k=final_top_k,
        )


@pytest.mark.parametrize(
    "rrf_k",
    [
        0,
        -1,
    ],
)
def test_retrieve_hybrid_nodes_rejects_invalid_rrf_k(
    rrf_k: int,
) -> None:
    index = MagicMock(spec=VectorStoreIndex)
    bm25_retriever = MagicMock(spec=BM25Retriever)

    with pytest.raises(
        ValueError,
        match="rrf_k должен быть больше 0!",
    ):
        retrieve_hybrid_nodes(
            index=index,
            bm25_retriever=bm25_retriever,
            query="рабочее время",
            rrf_k=rrf_k,
        )
