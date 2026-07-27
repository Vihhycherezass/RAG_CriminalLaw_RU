from llama_index.core.schema import NodeWithScore, BaseNode
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer


def build_bm25_retriever(
    nodes: list[BaseNode],
    top_k: int = 5,
) -> BM25Retriever:
    """Создает BM25 retriever с русским стеммингом."""

    if not nodes:
        raise ValueError("Список узлов пуст!")

    if top_k <= 0:
        raise ValueError("top_k должен быть больше 0!")

    for node in nodes:
        if not isinstance(node, BaseNode):
            raise TypeError(
                "Список содержит элемент, " "не являющийся объектом BaseNode!"
            )

        if not node.get_content().strip():
            raise ValueError("Узел не содержит текста!")

    effective_top_k = min(top_k, len(nodes))

    stemmer = Stemmer.Stemmer("russian")

    retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        stemmer=stemmer,
        language="ru",
        similarity_top_k=effective_top_k,
        verbose=False,
    )

    if not isinstance(retriever, BM25Retriever):
        raise TypeError("Созданный объект не является BM25Retriever!")

    return retriever


def retrieve_bm25_nodes(
    retriever: BM25Retriever,
    query: str,
) -> list[NodeWithScore]:
    """Возвращает узлы, найденные лексическим поиском BM25."""

    if not isinstance(retriever, BM25Retriever):
        raise TypeError("retriever должен быть объектом BM25Retriever!")

    query = query.strip()

    if not query:
        raise ValueError("Запрос пустой!")

    results = retriever.retrieve(query)

    for result in results:
        if not isinstance(result, NodeWithScore):
            raise TypeError(
                "Результат поиска содержит элемент, не являющийся объектом NodeWithScore!"
            )

    return results
