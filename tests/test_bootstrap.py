from pathlib import Path
from unittest.mock import MagicMock

import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import (
    BaseEmbedding,
)
from llama_index.core.schema import TextNode

from rag_labor_code import bootstrap
from rag_labor_code.bootstrap import (
    build_new_vector_index,
    build_rag_pipeline,
    extract_nodes_from_index,
    load_or_build_vector_index,
)
from rag_labor_code.config import AppConfig
from rag_labor_code.pipeline.rag_pipeline import (
    RAGPipeline,
    RAGPipelineConfig,
)


def make_config(
    tmp_path: Path,
    *,
    rebuild_index: bool = False,
) -> AppConfig:
    return AppConfig(
        saiga_model_path=tmp_path / "saiga.gguf",
        pdf_path=tmp_path / "labor_code.pdf",
        vector_index_dir=tmp_path / "vector_index",
        nemo_config_dir=tmp_path / "nemo",
        rebuild_index=rebuild_index,
    )


def test_extract_nodes_from_index_returns_stored_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_1 = TextNode(
        text="Статья 91.",
    )
    node_2 = TextNode(
        text="Статья 92.",
    )

    fake_docstore = MagicMock()

    fake_docstore.docs = {
        node_1.node_id: node_1,
        node_2.node_id: node_2,
    }

    fake_storage_context = MagicMock()
    fake_storage_context.docstore = fake_docstore

    fake_index = MagicMock(spec=VectorStoreIndex)
    fake_index.storage_context = fake_storage_context

    monkeypatch.setattr(
        bootstrap,
        "VECTOR_INDEX_TYPE",
        type(fake_index),
    )

    result = extract_nodes_from_index(
        fake_index,
    )

    assert result == [
        node_1,
        node_2,
    ]


def test_extract_nodes_from_index_rejects_empty_docstore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_docstore = MagicMock()
    fake_docstore.docs = {}

    fake_storage_context = MagicMock()
    fake_storage_context.docstore = fake_docstore

    fake_index = MagicMock(spec=VectorStoreIndex)
    fake_index.storage_context = fake_storage_context

    monkeypatch.setattr(
        bootstrap,
        "VECTOR_INDEX_TYPE",
        type(fake_index),
    )

    with pytest.raises(
        ValueError,
        match=("В сохранённом индексе " "не найдены nodes!"),
    ):
        extract_nodes_from_index(
            fake_index,
        )


def test_extract_nodes_from_index_rejects_invalid_index() -> None:
    with pytest.raises(
        TypeError,
        match=("index должен быть объектом " "VectorStoreIndex!"),
    ):
        extract_nodes_from_index(
            object(),  # type: ignore[arg-type]
        )


def test_build_new_vector_index_runs_full_ingestion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    embed_model = MagicMock(spec=BaseEmbedding)

    articles = [object()]
    documents = [object()]

    nodes = [
        TextNode(
            text="Статья 91.",
        )
    ]

    embedded_nodes = [
        TextNode(
            text="Статья 91.",
            embedding=[1.0, 0.0],
        )
    ]

    index = MagicMock(spec=VectorStoreIndex)

    extract_mock = MagicMock(return_value="raw text")
    normalize_mock = MagicMock(return_value="normalized text")
    parse_mock = MagicMock(return_value=articles)
    documents_mock = MagicMock(return_value=documents)
    nodes_mock = MagicMock(return_value=nodes)
    embed_mock = MagicMock(return_value=embedded_nodes)
    build_index_mock = MagicMock(return_value=index)
    save_index_mock = MagicMock()

    monkeypatch.setattr(
        bootstrap,
        "extract_text_from_pdf",
        extract_mock,
    )
    monkeypatch.setattr(
        bootstrap,
        "normalize_text",
        normalize_mock,
    )
    monkeypatch.setattr(
        bootstrap,
        "parse_articles",
        parse_mock,
    )
    monkeypatch.setattr(
        bootstrap,
        "articles_to_documents",
        documents_mock,
    )
    monkeypatch.setattr(
        bootstrap,
        "documents_to_nodes",
        nodes_mock,
    )
    monkeypatch.setattr(
        bootstrap,
        "embed_nodes",
        embed_mock,
    )
    monkeypatch.setattr(
        bootstrap,
        "build_vector_index",
        build_index_mock,
    )
    monkeypatch.setattr(
        bootstrap,
        "save_vector_index",
        save_index_mock,
    )

    result_index, result_nodes = build_new_vector_index(
        config=config,
        embed_model=embed_model,
    )

    assert result_index is index
    assert result_nodes is embedded_nodes

    extract_mock.assert_called_once_with(config.pdf_path)

    normalize_mock.assert_called_once_with("raw text")

    parse_mock.assert_called_once_with("normalized text")

    documents_mock.assert_called_once_with(articles)

    nodes_mock.assert_called_once_with(
        documents,
        chunk_size=512,
        chunk_overlap=100,
    )

    embed_mock.assert_called_once_with(
        nodes=nodes,
        embed_model=embed_model,
    )

    build_index_mock.assert_called_once_with(
        nodes=embedded_nodes,
        embed_model=embed_model,
    )

    save_index_mock.assert_called_once_with(
        index=index,
        persist_dir=config.vector_index_dir,
    )


def test_load_or_build_loads_existing_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    config.vector_index_dir.mkdir()

    embed_model = MagicMock(spec=BaseEmbedding)

    index = MagicMock(spec=VectorStoreIndex)

    nodes = [
        TextNode(
            text="Статья 91.",
        )
    ]

    load_mock = MagicMock(return_value=index)
    extract_mock = MagicMock(return_value=nodes)
    build_mock = MagicMock()

    monkeypatch.setattr(
        bootstrap,
        "load_vector_index",
        load_mock,
    )
    monkeypatch.setattr(
        bootstrap,
        "extract_nodes_from_index",
        extract_mock,
    )
    monkeypatch.setattr(
        bootstrap,
        "build_new_vector_index",
        build_mock,
    )

    result = load_or_build_vector_index(
        config=config,
        embed_model=embed_model,
    )

    assert result == (
        index,
        nodes,
    )

    load_mock.assert_called_once_with(
        persist_dir=config.vector_index_dir,
        embed_model=embed_model,
    )

    extract_mock.assert_called_once_with(index)

    build_mock.assert_not_called()


def test_load_or_build_builds_when_index_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    embed_model = MagicMock(spec=BaseEmbedding)

    expected = (
        MagicMock(spec=VectorStoreIndex),
        [TextNode(text="Статья 91.")],
    )

    build_mock = MagicMock(return_value=expected)
    load_mock = MagicMock()

    monkeypatch.setattr(
        bootstrap,
        "build_new_vector_index",
        build_mock,
    )
    monkeypatch.setattr(
        bootstrap,
        "load_vector_index",
        load_mock,
    )

    result = load_or_build_vector_index(
        config=config,
        embed_model=embed_model,
    )

    assert result == expected

    build_mock.assert_called_once_with(
        config=config,
        embed_model=embed_model,
    )

    load_mock.assert_not_called()


def test_load_or_build_rebuilds_when_forced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = make_config(
        tmp_path,
        rebuild_index=True,
    )

    config.vector_index_dir.mkdir()

    embed_model = MagicMock(spec=BaseEmbedding)

    expected = (
        MagicMock(spec=VectorStoreIndex),
        [TextNode(text="Статья 91.")],
    )

    build_mock = MagicMock(return_value=expected)
    load_mock = MagicMock()

    monkeypatch.setattr(
        bootstrap,
        "build_new_vector_index",
        build_mock,
    )
    monkeypatch.setattr(
        bootstrap,
        "load_vector_index",
        load_mock,
    )

    result = load_or_build_vector_index(
        config=config,
        embed_model=embed_model,
    )

    assert result == expected

    build_mock.assert_called_once_with(
        config=config,
        embed_model=embed_model,
    )

    load_mock.assert_not_called()


def test_build_rag_pipeline_assembles_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline_config = RAGPipelineConfig(
        retrieval_top_k=12,
    )

    config = AppConfig(
        saiga_model_path=tmp_path / "saiga.gguf",
        pdf_path=tmp_path / "labor_code.pdf",
        vector_index_dir=tmp_path / "index",
        nemo_config_dir=tmp_path / "nemo",
        embedding_device="cpu",
        reranker_device="cpu",
        n_ctx=4096,
        n_gpu_layers=0,
        n_threads=4,
        chat_format="chatml",
        pipeline_config=pipeline_config,
    )

    embed_model = object()
    index = object()
    nodes = [object()]
    bm25 = object()
    reranker = object()
    llm = object()
    nemo = object()
    pipeline = object()

    embed_factory = MagicMock(return_value=embed_model)
    index_factory = MagicMock(
        return_value=(
            index,
            nodes,
        )
    )
    bm25_factory = MagicMock(return_value=bm25)
    reranker_factory = MagicMock(return_value=reranker)
    llm_factory = MagicMock(return_value=llm)
    nemo_factory = MagicMock(return_value=nemo)
    pipeline_factory = MagicMock(return_value=pipeline)

    monkeypatch.setattr(
        bootstrap,
        "create_e5_embed_model",
        embed_factory,
    )
    monkeypatch.setattr(
        bootstrap,
        "load_or_build_vector_index",
        index_factory,
    )
    monkeypatch.setattr(
        bootstrap,
        "build_bm25_retriever",
        bm25_factory,
    )
    monkeypatch.setattr(
        bootstrap,
        "create_cross_encoder",
        reranker_factory,
    )
    monkeypatch.setattr(
        bootstrap,
        "create_saiga_llm",
        llm_factory,
    )
    monkeypatch.setattr(
        bootstrap,
        "create_nemo_guardrails_adapter",
        nemo_factory,
    )
    monkeypatch.setattr(
        bootstrap,
        "RAGPipeline",
        pipeline_factory,
    )

    result = build_rag_pipeline(config)

    assert result is pipeline

    embed_factory.assert_called_once_with(
        device="cpu",
    )

    index_factory.assert_called_once_with(
        config=config,
        embed_model=embed_model,
    )

    bm25_factory.assert_called_once_with(
        nodes=nodes,
        top_k=12,
    )

    reranker_factory.assert_called_once_with(
        device="cpu",
    )

    llm_factory.assert_called_once_with(
        model_path=config.saiga_model_path,
        n_ctx=4096,
        n_gpu_layers=0,
        n_threads=4,
        chat_format="chatml",
    )

    nemo_factory.assert_called_once_with(
        config_path=config.nemo_config_dir,
        llm=llm,
    )

    pipeline_factory.assert_called_once_with(
        index=index,
        bm25_retriever=bm25,
        reranker=reranker,
        llm=llm,
        config=pipeline_config,
        nemo_guardrails=nemo,
    )


def test_build_rag_pipeline_rejects_invalid_config() -> None:
    with pytest.raises(
        TypeError,
        match=("config должен быть объектом " "AppConfig!"),
    ):
        build_rag_pipeline(
            object(),  # type: ignore[arg-type]
        )
