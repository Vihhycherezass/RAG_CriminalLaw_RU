from llama_index.core import Document
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SentenceSplitter


def documents_to_nodes(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> list[BaseNode]:
    """Разбивает документ LlamaIndex на узлы."""
    if not documents:
        raise ValueError("Список документов пуст!")

    for document in documents:
        if not isinstance(document, Document):
            raise TypeError(
                "Как минимум один из элементов списка documents не является объектом Document!"
            )

        if not document.get_content().strip():
            raise ValueError("Как минимум один из документов не имеет текс статьи!")

    if chunk_size < 0:
        raise ValueError("chunk_size должен быть больше 0!")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap не может быть меньше 0!")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap не может быть больше или равен chunk_size!")

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_metadata=True,
        include_prev_next_rel=True,
    )

    nodes = splitter.get_nodes_from_documents(documents, show_progress=False)

    if not nodes:
        raise ValueError("Список узлов пуст!")

    return nodes
