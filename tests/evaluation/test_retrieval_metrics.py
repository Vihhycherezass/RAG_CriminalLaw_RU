import pytest

from rag_labor_code.evaluation.retrieval_metrics import (
    RetrievalEvaluationCase,
    RetrievalMetrics,
    evaluate_retrieval,
)


def make_case(
    query: str = "Какова продолжительность рабочего времени?",
    relevant_article_nums: tuple[str, ...] = ("91",),
    retrieved_article_nums: tuple[str, ...] = ("91", "92", "93"),
) -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        query=query,
        relevant_article_nums=relevant_article_nums,
        retrieved_article_nums=retrieved_article_nums,
    )


def test_evaluate_retrieval_returns_perfect_metrics() -> None:
    cases = [
        make_case(
            relevant_article_nums=("91",),
            retrieved_article_nums=("91", "92", "93"),
        ),
    ]

    result = evaluate_retrieval(
        cases=cases,
        k=3,
    )

    assert result == RetrievalMetrics(
        evaluated_cases=1,
        k=3,
        hit_rate_at_k=1.0,
        mean_recall_at_k=1.0,
        mrr_at_k=1.0,
    )


def test_evaluate_retrieval_calculates_partial_recall() -> None:
    cases = [
        make_case(
            relevant_article_nums=("91", "92"),
            retrieved_article_nums=("50", "91", "100"),
        ),
    ]

    result = evaluate_retrieval(
        cases=cases,
        k=3,
    )

    assert result.evaluated_cases == 1
    assert result.k == 3
    assert result.hit_rate_at_k == 1.0
    assert result.mean_recall_at_k == 0.5
    assert result.mrr_at_k == 0.5


def test_evaluate_retrieval_returns_zero_for_miss() -> None:
    cases = [
        make_case(
            relevant_article_nums=("91",),
            retrieved_article_nums=("50", "51", "52"),
        ),
    ]

    result = evaluate_retrieval(
        cases=cases,
        k=3,
    )

    assert result == RetrievalMetrics(
        evaluated_cases=1,
        k=3,
        hit_rate_at_k=0.0,
        mean_recall_at_k=0.0,
        mrr_at_k=0.0,
    )


def test_evaluate_retrieval_averages_multiple_cases() -> None:
    cases = [
        make_case(
            query="Первый вопрос",
            relevant_article_nums=("91",),
            retrieved_article_nums=("91", "92"),
        ),
        make_case(
            query="Второй вопрос",
            relevant_article_nums=("81", "82"),
            retrieved_article_nums=("99", "82", "81"),
        ),
        make_case(
            query="Третий вопрос",
            relevant_article_nums=("57",),
            retrieved_article_nums=("60", "61"),
        ),
    ]

    result = evaluate_retrieval(
        cases=cases,
        k=2,
    )

    assert result.evaluated_cases == 3
    assert result.k == 2
    assert result.hit_rate_at_k == pytest.approx(2 / 3)
    assert result.mean_recall_at_k == pytest.approx(0.5)
    assert result.mrr_at_k == pytest.approx(0.5)


def test_evaluate_retrieval_respects_k() -> None:
    cases = [
        make_case(
            relevant_article_nums=("91",),
            retrieved_article_nums=("50", "51", "91"),
        ),
    ]

    result = evaluate_retrieval(
        cases=cases,
        k=2,
    )

    assert result.hit_rate_at_k == 0.0
    assert result.mean_recall_at_k == 0.0
    assert result.mrr_at_k == 0.0


def test_evaluate_retrieval_allows_empty_retrieved_articles() -> None:
    cases = [
        make_case(
            relevant_article_nums=("91",),
            retrieved_article_nums=(),
        ),
    ]

    result = evaluate_retrieval(
        cases=cases,
        k=5,
    )

    assert result == RetrievalMetrics(
        evaluated_cases=1,
        k=5,
        hit_rate_at_k=0.0,
        mean_recall_at_k=0.0,
        mrr_at_k=0.0,
    )


def test_evaluate_retrieval_does_not_inflate_recall_with_duplicates() -> None:
    cases = [
        make_case(
            relevant_article_nums=("91", "92"),
            retrieved_article_nums=("91", "91", "92"),
        ),
    ]

    result = evaluate_retrieval(
        cases=cases,
        k=3,
    )

    assert result.hit_rate_at_k == 1.0
    assert result.mean_recall_at_k == 1.0
    assert result.mrr_at_k == 1.0


def test_evaluate_retrieval_rejects_non_list_cases() -> None:
    with pytest.raises(
        TypeError,
        match="cases должен быть списком!",
    ):
        evaluate_retrieval(
            cases=(make_case(),),  # type: ignore[arg-type]
        )


def test_evaluate_retrieval_rejects_empty_cases() -> None:
    with pytest.raises(
        ValueError,
        match="cases не должен быть пустым!",
    ):
        evaluate_retrieval([])


@pytest.mark.parametrize(
    "k",
    [
        True,
        2.5,
        "5",
    ],
)
def test_evaluate_retrieval_rejects_invalid_k_type(
    k: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="k должен быть целым числом!",
    ):
        evaluate_retrieval(
            cases=[make_case()],
            k=k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "k",
    [
        0,
        -1,
    ],
)
def test_evaluate_retrieval_rejects_invalid_k_value(
    k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="k должен быть больше 0!",
    ):
        evaluate_retrieval(
            cases=[make_case()],
            k=k,
        )


def test_evaluate_retrieval_rejects_invalid_case_type() -> None:
    with pytest.raises(
        TypeError,
        match=("Каждый элемент cases должен быть " "RetrievalEvaluationCase!"),
    ):
        evaluate_retrieval(
            cases=["не evaluation case"],  # type: ignore[list-item]
        )


def test_evaluate_retrieval_rejects_invalid_query_type() -> None:
    case = RetrievalEvaluationCase(
        query=123,  # type: ignore[arg-type]
        relevant_article_nums=("91",),
        retrieved_article_nums=("91",),
    )

    with pytest.raises(
        TypeError,
        match="query должен быть строкой!",
    ):
        evaluate_retrieval([case])


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
    ],
)
def test_evaluate_retrieval_rejects_empty_query(
    query: str,
) -> None:
    case = make_case(query=query)

    with pytest.raises(
        ValueError,
        match="query не должен быть пустым!",
    ):
        evaluate_retrieval([case])


def test_evaluate_retrieval_rejects_invalid_relevant_type() -> None:
    case = RetrievalEvaluationCase(
        query="Вопрос?",
        relevant_article_nums=["91"],  # type: ignore[arg-type]
        retrieved_article_nums=("91",),
    )

    with pytest.raises(
        TypeError,
        match="relevant_article_nums должен быть tuple!",
    ):
        evaluate_retrieval([case])


def test_evaluate_retrieval_rejects_empty_relevant_articles() -> None:
    case = make_case(
        relevant_article_nums=(),
    )

    with pytest.raises(
        ValueError,
        match="relevant_article_nums не должен быть пустым!",
    ):
        evaluate_retrieval([case])


def test_evaluate_retrieval_rejects_invalid_retrieved_type() -> None:
    case = RetrievalEvaluationCase(
        query="Вопрос?",
        relevant_article_nums=("91",),
        retrieved_article_nums=["91"],  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match="retrieved_article_nums должен быть tuple!",
    ):
        evaluate_retrieval([case])


@pytest.mark.parametrize(
    ("relevant_article_nums", "retrieved_article_nums"),
    [
        (
            ("91", ""),
            ("91",),
        ),
        (
            ("91",),
            ("91", "   "),
        ),
        (
            ("91",),
            ("91", 92),
        ),
    ],
)
def test_evaluate_retrieval_rejects_invalid_article_numbers(
    relevant_article_nums: tuple[object, ...],
    retrieved_article_nums: tuple[object, ...],
) -> None:
    case = RetrievalEvaluationCase(
        query="Вопрос?",
        relevant_article_nums=relevant_article_nums,  # type: ignore[arg-type]
        retrieved_article_nums=retrieved_article_nums,  # type: ignore[arg-type]
    )

    with pytest.raises(
        ValueError,
        match="Номера статей должны быть непустыми строками!",
    ):
        evaluate_retrieval([case])
