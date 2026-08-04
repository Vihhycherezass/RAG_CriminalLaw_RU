from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode
from sentence_transformers import CrossEncoder

from rag_labor_code.reranking.cross_encoder import (
    DEFAULT_RERANKER_MODEL_NAME,
    create_cross_encoder,
    rerank_nodes,
)


def create_candidates() -> list[NodeWithScore]:
    return [
        NodeWithScore(
            node=TextNode(
                text="Статья 91. Понятие рабочего времени.",
                metadata={"article_num": "91"},
            ),
            score=0.031,
        ),
        NodeWithScore(
            node=TextNode(
                text=("Статья 92. Нормальная продолжительность " "рабочего времени."),
                metadata={"article_num": "92"},
            ),
            score=0.029,
        ),
        NodeWithScore(
            node=TextNode(
                text="Статья 93. Сокращённое рабочее время.",
                metadata={"article_num": "93"},
            ),
            score=0.018,
        ),
    ]


@patch("rag_labor_code.reranking.cross_encoder.CrossEncoder")
def test_create_cross_encoder_uses_default_parameters(
    mock_cross_encoder_class: MagicMock,
) -> None:
    expected_model = MagicMock(spec=CrossEncoder)
    mock_cross_encoder_class.return_value = expected_model

    model = create_cross_encoder()

    mock_cross_encoder_class.assert_called_once_with(
        model_name_or_path=DEFAULT_RERANKER_MODEL_NAME,
        device=None,
        max_length=512,
    )

    assert model is expected_model


@patch("rag_labor_code.reranking.cross_encoder.CrossEncoder")
def test_create_cross_encoder_uses_custom_parameters(
    mock_cross_encoder_class: MagicMock,
) -> None:
    expected_model = MagicMock(spec=CrossEncoder)
    mock_cross_encoder_class.return_value = expected_model

    model = create_cross_encoder(
        model_name="custom/reranker",
        device="cuda",
        max_length=384,
    )

    mock_cross_encoder_class.assert_called_once_with(
        model_name_or_path="custom/reranker",
        device="cuda",
        max_length=384,
    )

    assert model is expected_model


@pytest.mark.parametrize(
    "model_name",
    [
        "",
        "   ",
    ],
)
def test_create_cross_encoder_rejects_empty_model_name(
    model_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Название reranker-модели не должно быть пустым!",
    ):
        create_cross_encoder(
            model_name=model_name,
        )


@pytest.mark.parametrize(
    "max_length",
    [
        0,
        -1,
    ],
)
def test_create_cross_encoder_rejects_invalid_max_length(
    max_length: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_length должен быть больше 0!",
    ):
        create_cross_encoder(
            max_length=max_length,
        )


def test_create_cross_encoder_rejects_empty_device() -> None:
    with pytest.raises(
        ValueError,
        match="Название устройства не должно быть пустым!",
    ):
        create_cross_encoder(
            device="   ",
        )


def test_rerank_nodes_predicts_query_document_pairs() -> None:
    candidates = create_candidates()
    reranker = MagicMock(spec=CrossEncoder)

    reranker.predict.return_value = [
        0.20,
        0.95,
        0.40,
    ]

    result = rerank_nodes(
        query="нормальная продолжительность рабочего времени",
        candidates=candidates,
        reranker=reranker,
        top_k=3,
        batch_size=4,
    )

    reranker.predict.assert_called_once_with(
        [
            (
                "нормальная продолжительность рабочего времени",
                "Статья 91. Понятие рабочего времени.",
            ),
            (
                "нормальная продолжительность рабочего времени",
                ("Статья 92. Нормальная продолжительность " "рабочего времени."),
            ),
            (
                "нормальная продолжительность рабочего времени",
                "Статья 93. Сокращённое рабочее время.",
            ),
        ],
        batch_size=4,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    assert [item.node.metadata["article_num"] for item in result] == ["92", "93", "91"]


def test_rerank_nodes_replaces_previous_scores() -> None:
    candidates = create_candidates()
    reranker = MagicMock(spec=CrossEncoder)

    original_scores = [candidate.score for candidate in candidates]

    reranker.predict.return_value = [
        0.10,
        0.90,
        0.50,
    ]

    result = rerank_nodes(
        query="рабочее время",
        candidates=candidates,
        reranker=reranker,
        top_k=3,
    )

    assert [item.score for item in result] == [0.90, 0.50, 0.10]

    assert [candidate.score for candidate in candidates] == original_scores


def test_rerank_nodes_preserves_nodes_and_metadata() -> None:
    candidates = create_candidates()
    reranker = MagicMock(spec=CrossEncoder)

    reranker.predict.return_value = [
        0.30,
        0.80,
        0.50,
    ]

    result = rerank_nodes(
        query="рабочее время",
        candidates=candidates,
        reranker=reranker,
        top_k=3,
    )

    nodes_by_article = {
        candidate.node.metadata["article_num"]: candidate.node
        for candidate in candidates
    }

    for item in result:
        article_num = item.node.metadata["article_num"]

        assert item.node is nodes_by_article[article_num]


def test_rerank_nodes_limits_result_count() -> None:
    candidates = create_candidates()
    reranker = MagicMock(spec=CrossEncoder)

    reranker.predict.return_value = [
        0.30,
        0.90,
        0.60,
    ]

    result = rerank_nodes(
        query="рабочее время",
        candidates=candidates,
        reranker=reranker,
        top_k=2,
    )

    assert len(result) == 2

    assert [item.node.metadata["article_num"] for item in result] == ["92", "93"]


def test_rerank_nodes_strips_query() -> None:
    candidates = create_candidates()
    reranker = MagicMock(spec=CrossEncoder)

    reranker.predict.return_value = [
        0.10,
        0.20,
        0.30,
    ]

    rerank_nodes(
        query="   рабочее время   ",
        candidates=candidates,
        reranker=reranker,
    )

    pairs = reranker.predict.call_args.args[0]

    assert all(query == "рабочее время" for query, _ in pairs)


def test_rerank_nodes_strips_candidate_text() -> None:
    candidates = [
        NodeWithScore(
            node=TextNode(
                text="   Статья 91. Рабочее время.   ",
            ),
            score=0.5,
        )
    ]
    reranker = MagicMock(spec=CrossEncoder)

    reranker.predict.return_value = [0.8]

    rerank_nodes(
        query="рабочее время",
        candidates=candidates,
        reranker=reranker,
    )

    reranker.predict.assert_called_once_with(
        [
            (
                "рабочее время",
                "Статья 91. Рабочее время.",
            )
        ],
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def test_rerank_nodes_returns_empty_list_for_empty_candidates() -> None:
    reranker = MagicMock(spec=CrossEncoder)

    result = rerank_nodes(
        query="рабочее время",
        candidates=[],
        reranker=reranker,
    )

    assert result == []
    reranker.predict.assert_not_called()


def test_rerank_nodes_rejects_invalid_reranker() -> None:
    with pytest.raises(
        TypeError,
        match="reranker должен быть объектом CrossEncoder!",
    ):
        rerank_nodes(
            query="рабочее время",
            candidates=create_candidates(),
            reranker="не модель",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
    ],
)
def test_rerank_nodes_rejects_empty_query(
    query: str,
) -> None:
    reranker = MagicMock(spec=CrossEncoder)

    with pytest.raises(
        ValueError,
        match="Запрос пустой!",
    ):
        rerank_nodes(
            query=query,
            candidates=create_candidates(),
            reranker=reranker,
        )


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_rerank_nodes_rejects_invalid_top_k(
    top_k: int,
) -> None:
    reranker = MagicMock(spec=CrossEncoder)

    with pytest.raises(
        ValueError,
        match="top_k должен быть больше 0!",
    ):
        rerank_nodes(
            query="рабочее время",
            candidates=create_candidates(),
            reranker=reranker,
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "batch_size",
    [
        0,
        -1,
    ],
)
def test_rerank_nodes_rejects_invalid_batch_size(
    batch_size: int,
) -> None:
    reranker = MagicMock(spec=CrossEncoder)

    with pytest.raises(
        ValueError,
        match="batch_size должен быть больше 0!",
    ):
        rerank_nodes(
            query="рабочее время",
            candidates=create_candidates(),
            reranker=reranker,
            batch_size=batch_size,
        )


def test_rerank_nodes_rejects_invalid_candidate_type() -> None:
    reranker = MagicMock(spec=CrossEncoder)

    with pytest.raises(
        TypeError,
        match=("Кандидат не является объектом " "NodeWithScore!"),
    ):
        rerank_nodes(
            query="рабочее время",
            candidates=[
                TextNode(text="Обычный узел."),
            ],  # type: ignore[list-item]
            reranker=reranker,
        )


def test_rerank_nodes_rejects_empty_candidate_text() -> None:
    reranker = MagicMock(spec=CrossEncoder)

    candidates = [
        NodeWithScore(
            node=TextNode(text="   "),
            score=0.5,
        )
    ]

    with pytest.raises(
        ValueError,
        match="Узел-кандидат не содержит текста!",
    ):
        rerank_nodes(
            query="рабочее время",
            candidates=candidates,
            reranker=reranker,
        )


def test_rerank_nodes_rejects_score_count_mismatch() -> None:
    candidates = create_candidates()
    reranker = MagicMock(spec=CrossEncoder)

    reranker.predict.return_value = [
        0.50,
        0.40,
    ]

    with pytest.raises(
        ValueError,
        match=("Количество scores не совпадает " "с количеством кандидатов!"),
    ):
        rerank_nodes(
            query="рабочее время",
            candidates=candidates,
            reranker=reranker,
        )
