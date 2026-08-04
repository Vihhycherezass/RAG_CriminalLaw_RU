from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    query: str
    relevant_article_nums: tuple[str, ...]
    retrieved_article_nums: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalMetrics:
    evaluated_cases: int
    k: int
    hit_rate_at_k: float
    mean_recall_at_k: float
    mrr_at_k: float


def _validate_case(case: RetrievalEvaluationCase) -> None:
    if not isinstance(case, RetrievalEvaluationCase):
        raise TypeError("Каждый элемент cases должен быть RetrievalEvaluationCase!")

    if not isinstance(case.query, str):
        raise TypeError("query должен быть строкой!")

    query = case.query.strip()

    if not query:
        raise ValueError("query не должен быть пустым!")

    if not isinstance(case.relevant_article_nums, tuple):
        raise TypeError("relevant_article_nums должен быть tuple!")

    if not case.relevant_article_nums:
        raise ValueError("relevant_article_nums не должен быть пустым!")

    if not isinstance(case.retrieved_article_nums, tuple):
        raise TypeError("retrieved_article_nums должен быть tuple!")

    all_article_nums = case.relevant_article_nums + case.retrieved_article_nums

    for article_num in all_article_nums:
        if not isinstance(article_num, str) or not article_num.strip():
            raise ValueError("Номера статей должны быть непустыми строками!")


def evaluate_retrieval(
    cases: list[RetrievalEvaluationCase],
    k: int = 5,
) -> RetrievalMetrics:
    """Считает offline-метрики retrieval по размеченным запросам."""

    if not isinstance(cases, list):
        raise TypeError("cases должен быть списком!")

    if not cases:
        raise ValueError("cases не должен быть пустым!")

    if type(k) is not int:
        raise TypeError("k должен быть целым числом!")

    if k <= 0:
        raise ValueError("k должен быть больше 0!")

    hits: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []

    for case in cases:
        _validate_case(case)

        top_k = case.retrieved_article_nums[:k]

        relevant_set = set(case.relevant_article_nums)
        retrieved_set = set(top_k)

        matched_articles = relevant_set & retrieved_set

        hit = 1.0 if matched_articles else 0.0

        recall = len(matched_articles) / len(relevant_set)

        reciprocal_rank = 0.0

        for rank, article_num in enumerate(top_k, start=1):
            if article_num in relevant_set:
                reciprocal_rank = 1.0 / rank
                break

        hits.append(hit)
        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)

    mean_hits = sum(hits) / len(hits)
    mean_recalls = sum(recalls) / len(recalls)
    mean_reciprocal_ranks = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return RetrievalMetrics(
        evaluated_cases=len(cases),
        k=k,
        hit_rate_at_k=mean_hits,
        mean_recall_at_k=mean_recalls,
        mrr_at_k=mean_reciprocal_ranks,
    )
