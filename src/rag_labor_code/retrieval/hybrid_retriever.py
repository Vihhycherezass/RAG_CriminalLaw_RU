from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever

from rag_labor_code.retrieval.vector_retriever import retrieve_vector_nodes
from rag_labor_code.retrieval.bm25_retriever import retrieve_bm25_nodes


def reciprocal_rank_fusion(
    result_lists: list[list[NodeWithScore]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[NodeWithScore]:
    """Объединяет несколько ранжированных выдач методом RRF."""

    if not result_lists:
        raise ValueError("Список поисковых выдач пуст!")

    if top_k <= 0:
        raise ValueError("top_k должен быть больше 0!")

    if rrf_k <= 0:
        raise ValueError("rrf_k должен быть больше 0!")

    nodes_by_id: dict[str, BaseNode] = {}
    scores_by_id: dict[str, float] = {}

    for results in result_lists:
        seen_nodes_ids: set[str] = set()
        unique_rank = 0

        for result in results:
            if not isinstance(result, NodeWithScore):
                raise TypeError(
                    "Поисковая выдача содержит элемент, не являющийся объектом NodeWithScore!"
                )

            node_id = result.node.node_id

            if node_id in seen_nodes_ids:
                continue

            seen_nodes_ids.add(node_id)

            unique_rank += 1

            if node_id not in nodes_by_id:
                nodes_by_id[node_id] = result.node
                scores_by_id[node_id] = 0.0

            scores_by_id[node_id] += 1 / (rrf_k + unique_rank)

    fused_results = [
        NodeWithScore(node=nodes_by_id[node_id], score=score)
        for node_id, score in scores_by_id.items()
    ]

    fused_results.sort(key=lambda item: item.score, reverse=True)

    return fused_results[:top_k]


def retrieve_hybrid_nodes(
    index: VectorStoreIndex,
    bm25_retriever: BM25Retriever,
    query: str,
    vector_top_k: int = 10,
    final_top_k: int = 5,
    rrf_k: int = 60,
) -> list[NodeWithScore]:
    """Объединяет результаты векторного и BM25-поиска."""

    if not isinstance(index, VectorStoreIndex):
        raise TypeError("index должен быть объектом VectorStoreIndex!")

    if not isinstance(bm25_retriever, BM25Retriever):
        raise TypeError("bm25_retriever должен быть объектом BM25Retriever!")

    query = query.strip()

    if not query:
        raise ValueError("Запрос пустой!")

    if vector_top_k <= 0:
        raise ValueError("vector_top_k должен быть больше 0!")

    if final_top_k <= 0:
        raise ValueError("final_top_k должен быть больше 0!")

    if rrf_k <= 0:
        raise ValueError("rrf_k должен быть больше 0!")

    vector_results = retrieve_vector_nodes(
        index=index,
        query=query,
        top_k=vector_top_k,
    )

    bm25_results = retrieve_bm25_nodes(
        retriever=bm25_retriever,
        query=query,
    )

    result = reciprocal_rank_fusion(
        result_lists=[vector_results, bm25_results],
        top_k=final_top_k,
        rrf_k=rrf_k,
    )

    return result
