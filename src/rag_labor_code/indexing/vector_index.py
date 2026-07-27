from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.schema import BaseNode
from llama_index.core.embeddings import BaseEmbedding


def build_vector_index(
    nodes: list[BaseNode],
    embed_model: BaseEmbedding,
) -> VectorStoreIndex:
    """Создает векторный индекс из узлов с embeddings."""

    if not nodes:
        raise ValueError("Список узлов пуст!")

    if not isinstance(embed_model, BaseEmbedding):
        raise TypeError("embed_model должен быть объектом BaseEmbedding!")

    for node in nodes:
        if not isinstance(node, BaseNode):
            raise TypeError("Список содержит элемент, не являющийся объектом BaseNode!")

        if not node.text or not node.text.strip():
            raise ValueError("Узел не содержит текста!")

        if not node.embedding:
            raise ValueError("Узел не содержит embedding!")

    embedding_dimension = len(nodes[0].embedding)

    for node in nodes[1:]:
        if len(node.embedding) != embedding_dimension:
            raise ValueError("Размерность embeddings узлов не совпадает!")

    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=embed_model,
        show_progress=False,
    )

    return index


def save_vector_index(
    index: VectorStoreIndex,
    persist_dir: Path,
) -> None:
    """Сохраняет векторный индекс на диск."""

    if not isinstance(index, VectorStoreIndex):
        raise TypeError("index должен быть объектом VectorStoreIndex!")

    if persist_dir.exists() and persist_dir.is_file():
        raise ValueError("Путь для сохранения индекса указывает на файл!")

    persist_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    index.storage_context.persist(persist_dir=str(persist_dir))


def load_vector_index(
    persist_dir: Path,
    embed_model: BaseEmbedding,
) -> VectorStoreIndex:
    """Загружает векторный индекс с диска."""

    if not isinstance(embed_model, BaseEmbedding):
        raise TypeError("embed_model должен быть объектом BaseEmbedding!")

    if not persist_dir.exists():
        raise FileNotFoundError("Директория с индексом не найдена!")

    if persist_dir.is_file():
        raise ValueError("Путь к индексу не является директорией!")

    if not persist_dir.is_dir():
        ValueError("Путь к индексу не является директорией!")

    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))

    index = load_index_from_storage(
        storage_context=storage_context,
        embed_model=embed_model,
    )

    if not isinstance(index, VectorStoreIndex):
        raise TypeError("Загруженный объект не является VectorStoreIndex!")

    return index
