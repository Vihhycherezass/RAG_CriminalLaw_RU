from llama_index.core.schema import NodeWithScore
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def create_cross_encoder(
    model_name: str = DEFAULT_RERANKER_MODEL_NAME,
    device: str | None = None,
    max_length: int = 512,
) -> CrossEncoder:
    """Создаёт multilingual CrossEncoder для reranking."""

    if not model_name or not model_name.strip():
        raise ValueError("Название reranker-модели не должно быть пустым!")

    if max_length <= 0:
        raise ValueError("max_length должен быть больше 0!")

    if device is not None and not device.strip():
        raise ValueError("Название устройства не должно быть пустым!")

    cross_encoder = CrossEncoder(
        model_name_or_path=model_name,
        device=device,
        max_length=max_length,
    )

    return cross_encoder


def rerank_nodes(
    query: str,
    candidates: list[NodeWithScore],
    reranker: CrossEncoder,
    top_k: int = 5,
    batch_size: int = 8,
) -> list[NodeWithScore]:
    """Повторно ранжирует найденные узлы с помощью CrossEncoder."""

    if not isinstance(reranker, CrossEncoder):
        raise TypeError("reranker должен быть объектом CrossEncoder!")

    query = query.strip()

    if not query:
        raise ValueError("Запрос пустой!")

    if top_k <= 0:
        raise ValueError("top_k должен быть больше 0!")

    if batch_size <= 0:
        raise ValueError("batch_size должен быть больше 0!")

    if not candidates:
        return []

    pairs: list[tuple[str, str]] = []

    for candidate in candidates:
        if not isinstance(candidate, NodeWithScore):
            raise TypeError("Кандидат не является объектом NodeWithScore!")

        node_text = candidate.node.get_content().strip()

        if not node_text:
            raise ValueError("Узел-кандидат не содержит текста!")

        pairs.append((query, node_text))

    scores = reranker.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    if len(scores) != len(candidates):
        raise ValueError("Количество scores не совпадает с количеством кандидатов!")

    reranker_results = [
        NodeWithScore(
            node=candidate.node,
            score=float(score),
        )
        for candidate, score in zip(candidates, scores, strict=True)
    ]

    reranker_results.sort(key=lambda item: item.score, reverse=True)

    return reranker_results[:top_k]
