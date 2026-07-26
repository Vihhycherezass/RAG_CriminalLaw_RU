from llama_index.core.schema import BaseNode
from llama_index.core.base.embeddings.base import BaseEmbedding


def prepare_passage(text: str) -> str:
    if not text.strip():
        raise ValueError("Текст пустой!")

    text = text.strip()

    if text.startswith("passage:"):
        return text
    else:
        return "passage: " + text


def prepare_query(query: str) -> str:
    if not query.strip():
        raise ValueError("Запрос пустой!")

    query = query.strip()

    if query.startswith("query:"):
        return query
    else:
        return "query: " + query


def embed_nodes(
    nodes: list[BaseNode],
    embed_model: BaseEmbedding,
) -> list[BaseNode]:
    if not nodes:
        raise ValueError("Список с узлами пуст!")

    text_list: list[str] = []

    for node in nodes:
        if not isinstance(node, BaseNode):
            raise TypeError("Список содержит элемент, не являющийся объектом BaseNode!")

        node_content = node.get_content().strip()

        if not node_content:
            raise ValueError("Узел не содержит текста!")

        node_content = prepare_passage(node_content)
        text_list.append(node_content)

    embeddings = embed_model.get_text_embedding_batch(
        texts=text_list,
        show_progress=False,
    )

    if len(embeddings) != len(nodes):
        raise ValueError("Количество embeddings не совпадает с количеством узлов!")

    for node, embedding in zip(nodes, embeddings, strict=True):
        node.embedding = embedding

    return nodes
