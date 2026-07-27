from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from rag_labor_code.embeddings.e5_embedder import prepare_query


def retrieve_vector_nodes(
    index: VectorStoreIndex,
    query: str,
    top_k: int = 5,
) -> list[NodeWithScore]:
    """Возвращает наиболее релевантные узлы векторного индекса."""

    if not isinstance(index, VectorStoreIndex):
        raise TypeError("index должен быть объектом VectorStoreIndex!")

    prepared_query = prepare_query(query)

    if top_k <= 0:
        raise ValueError("top_k должен быть больше 0!")

    retriever = index.as_retriever(
        similarity_top_k=top_k,
    )

    results = retriever.retrieve(prepared_query)

    for res in results:
        if not isinstance(res, NodeWithScore):
            raise TypeError(
                "Результат поиска содержит элемент, не являющийся объектом NodeWithScore!"
            )

    return results
