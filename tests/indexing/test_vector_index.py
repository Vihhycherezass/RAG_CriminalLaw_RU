from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode

from rag_labor_code.indexing.vector_index import (
    build_vector_index,
    load_vector_index,
    save_vector_index,
)


def create_embedded_nodes() -> list[TextNode]:
    return [
        TextNode(
            text="Статья 91. Понятие рабочего времени.",
            metadata={"article_num": "91"},
            embedding=[1.0, 0.0, 0.5],
        ),
        TextNode(
            text="Статья 92. Нормальная продолжительность рабочего времени.",
            metadata={"article_num": "92"},
            embedding=[0.0, 1.0, 0.5],
        ),
    ]


@patch("rag_labor_code.indexing.vector_index.VectorStoreIndex")
def test_build_vector_index_creates_index(
    mock_vector_store_index: MagicMock,
) -> None:
    nodes = create_embedded_nodes()
    embed_model = MagicMock(spec=BaseEmbedding)
    expected_index = MagicMock(spec=VectorStoreIndex)

    mock_vector_store_index.return_value = expected_index

    index = build_vector_index(
        nodes=nodes,
        embed_model=embed_model,
    )

    mock_vector_store_index.assert_called_once_with(
        nodes=nodes,
        embed_model=embed_model,
        show_progress=False,
    )

    assert index is expected_index


def test_build_vector_index_rejects_empty_nodes() -> None:
    embed_model = MagicMock(spec=BaseEmbedding)

    with pytest.raises(
        ValueError,
        match="Список узлов пуст!",
    ):
        build_vector_index(
            nodes=[],
            embed_model=embed_model,
        )


def test_build_vector_index_rejects_non_node_element() -> None:
    embed_model = MagicMock(spec=BaseEmbedding)

    with pytest.raises(
        TypeError,
        match="Список содержит элемент, не являющийся объектом BaseNode!",
    ):
        build_vector_index(
            nodes=["не узел"],  # type: ignore[list-item]
            embed_model=embed_model,
        )


def test_build_vector_index_rejects_empty_node_text() -> None:
    nodes = [
        TextNode(
            text="   ",
            embedding=[1.0, 0.0],
        )
    ]
    embed_model = MagicMock(spec=BaseEmbedding)

    with pytest.raises(
        ValueError,
        match="Узел не содержит текста!",
    ):
        build_vector_index(
            nodes=nodes,
            embed_model=embed_model,
        )


def test_build_vector_index_rejects_node_without_embedding() -> None:
    nodes = [
        TextNode(
            text="Статья 91. Понятие рабочего времени.",
        )
    ]
    embed_model = MagicMock(spec=BaseEmbedding)

    with pytest.raises(
        ValueError,
        match="Узел не содержит embedding!",
    ):
        build_vector_index(
            nodes=nodes,
            embed_model=embed_model,
        )


def test_build_vector_index_rejects_empty_embedding() -> None:
    nodes = [
        TextNode(
            text="Статья 91. Понятие рабочего времени.",
            embedding=[],
        )
    ]
    embed_model = MagicMock(spec=BaseEmbedding)

    with pytest.raises(
        ValueError,
        match="Узел не содержит embedding!",
    ):
        build_vector_index(
            nodes=nodes,
            embed_model=embed_model,
        )


def test_build_vector_index_rejects_inconsistent_embedding_dimensions() -> None:
    nodes = [
        TextNode(
            text="Текст статьи 91.",
            embedding=[1.0, 0.0],
        ),
        TextNode(
            text="Текст статьи 92.",
            embedding=[0.0, 1.0, 0.5],
        ),
    ]
    embed_model = MagicMock(spec=BaseEmbedding)

    with pytest.raises(
        ValueError,
        match="Размерность embeddings узлов не совпадает!",
    ):
        build_vector_index(
            nodes=nodes,
            embed_model=embed_model,
        )


def test_build_vector_index_rejects_invalid_embed_model() -> None:
    nodes = create_embedded_nodes()

    with pytest.raises(
        TypeError,
        match="embed_model должен быть объектом BaseEmbedding!",
    ):
        build_vector_index(
            nodes=nodes,
            embed_model="не модель",  # type: ignore[arg-type]
        )


def test_save_vector_index_creates_directory_and_saves_index(
    tmp_path: Path,
) -> None:
    index = MagicMock(spec=VectorStoreIndex)
    storage_context = MagicMock(spec=StorageContext)
    index.storage_context = storage_context

    persist_dir = tmp_path / "nested" / "vector_index"

    save_vector_index(
        index=index,
        persist_dir=persist_dir,
    )

    assert persist_dir.is_dir()

    storage_context.persist.assert_called_once_with(
        persist_dir=str(persist_dir),
    )


def test_save_vector_index_rejects_invalid_index(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match="index должен быть объектом VectorStoreIndex!",
    ):
        save_vector_index(
            index="не индекс",  # type: ignore[arg-type]
            persist_dir=tmp_path / "vector_index",
        )


def test_save_vector_index_rejects_file_path(
    tmp_path: Path,
) -> None:
    persist_path = tmp_path / "vector_index"
    persist_path.write_text(
        "Это файл, а не директория.",
        encoding="utf-8",
    )

    index = MagicMock(spec=VectorStoreIndex)

    with pytest.raises(
        ValueError,
        match="Путь для сохранения индекса указывает на файл!",
    ):
        save_vector_index(
            index=index,
            persist_dir=persist_path,
        )


@patch("rag_labor_code.indexing.vector_index.load_index_from_storage")
@patch("rag_labor_code.indexing.vector_index.StorageContext")
def test_load_vector_index_loads_persisted_index(
    mock_storage_context_class: MagicMock,
    mock_load_index_from_storage: MagicMock,
    tmp_path: Path,
) -> None:
    persist_dir = tmp_path / "vector_index"
    persist_dir.mkdir()

    embed_model = MagicMock(spec=BaseEmbedding)
    storage_context = MagicMock(spec=StorageContext)
    expected_index = MagicMock(spec=VectorStoreIndex)

    mock_storage_context_class.from_defaults.return_value = storage_context
    mock_load_index_from_storage.return_value = expected_index

    index = load_vector_index(
        persist_dir=persist_dir,
        embed_model=embed_model,
    )

    mock_storage_context_class.from_defaults.assert_called_once_with(
        persist_dir=str(persist_dir),
    )

    mock_load_index_from_storage.assert_called_once_with(
        storage_context=storage_context,
        embed_model=embed_model,
    )

    assert index is expected_index


def test_load_vector_index_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    embed_model = MagicMock(spec=BaseEmbedding)
    persist_dir = tmp_path / "missing_index"

    with pytest.raises(
        FileNotFoundError,
        match="Директория с индексом не найдена!",
    ):
        load_vector_index(
            persist_dir=persist_dir,
            embed_model=embed_model,
        )


def test_load_vector_index_rejects_file_path(
    tmp_path: Path,
) -> None:
    embed_model = MagicMock(spec=BaseEmbedding)
    persist_path = tmp_path / "vector_index"
    persist_path.write_text(
        "Это файл.",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Путь к индексу не является директорией!",
    ):
        load_vector_index(
            persist_dir=persist_path,
            embed_model=embed_model,
        )


def test_load_vector_index_rejects_invalid_embed_model(
    tmp_path: Path,
) -> None:
    persist_dir = tmp_path / "vector_index"
    persist_dir.mkdir()

    with pytest.raises(
        TypeError,
        match="embed_model должен быть объектом BaseEmbedding!",
    ):
        load_vector_index(
            persist_dir=persist_dir,
            embed_model="не модель",  # type: ignore[arg-type]
        )


@patch("rag_labor_code.indexing.vector_index.load_index_from_storage")
@patch("rag_labor_code.indexing.vector_index.StorageContext")
def test_load_vector_index_rejects_non_vector_index(
    mock_storage_context_class: MagicMock,
    mock_load_index_from_storage: MagicMock,
    tmp_path: Path,
) -> None:
    persist_dir = tmp_path / "vector_index"
    persist_dir.mkdir()

    embed_model = MagicMock(spec=BaseEmbedding)
    storage_context = MagicMock(spec=StorageContext)

    mock_storage_context_class.from_defaults.return_value = storage_context
    mock_load_index_from_storage.return_value = object()

    with pytest.raises(
        TypeError,
        match="Загруженный объект не является VectorStoreIndex!",
    ):
        load_vector_index(
            persist_dir=persist_dir,
            embed_model=embed_model,
        )
