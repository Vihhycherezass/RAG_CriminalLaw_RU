from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import (
    BaseEmbedding,
)
from llama_index.core.schema import BaseNode

from rag_labor_code.config import AppConfig
from rag_labor_code.embeddings.e5_embedder import (
    embed_nodes,
)
from rag_labor_code.embeddings.model_factory import (
    create_e5_embed_model,
)
from rag_labor_code.generation.saiga_generator import (
    create_saiga_llm,
)
from rag_labor_code.guardrails.nemo_guardrails import (
    create_nemo_guardrails_adapter,
)
from rag_labor_code.indexing.vector_index import (
    build_vector_index,
    load_vector_index,
    save_vector_index,
)
from rag_labor_code.ingestion.document_builder import (
    articles_to_documents,
)
from rag_labor_code.ingestion.node_builder import (
    documents_to_nodes,
)
from rag_labor_code.ingestion.pdf_parser import (
    extract_text_from_pdf,
    normalize_text,
    parse_articles,
)
from rag_labor_code.pipeline.rag_pipeline import (
    RAGPipeline,
)
from rag_labor_code.reranking.cross_encoder import (
    create_cross_encoder,
)
from rag_labor_code.retrieval.bm25_retriever import (
    build_bm25_retriever,
)

VECTOR_INDEX_TYPE = VectorStoreIndex
APP_CONFIG_TYPE = AppConfig


def extract_nodes_from_index(
    index: VectorStoreIndex,
) -> list[BaseNode]:

    if not isinstance(index, VECTOR_INDEX_TYPE):
        raise TypeError("index должен быть объектом VectorStoreIndex!")

    stored_docs = index.storage_context.docstore.docs

    nodes = [
        document for document in stored_docs.values() if isinstance(document, BaseNode)
    ]

    if not nodes:
        raise ValueError("В сохранённом индексе не найдены nodes!")

    return nodes


def build_new_vector_index(
    config: AppConfig,
    embed_model: BaseEmbedding,
) -> tuple[VectorStoreIndex, list[BaseNode]]:

    if not isinstance(config, APP_CONFIG_TYPE):
        raise TypeError("config должен быть объектом AppConfig!")

    if not isinstance(embed_model, BaseEmbedding):
        raise TypeError("embedded_model должен быть объектом BaseEmbedding!")

    raw_text = extract_text_from_pdf(config.pdf_path)

    normalized_text = normalize_text(raw_text)

    articles = parse_articles(normalized_text)

    documents = articles_to_documents(articles)

    nodes = documents_to_nodes(
        documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    embedded_nodes = embed_nodes(
        nodes=nodes,
        embed_model=embed_model,
    )

    index = build_vector_index(
        nodes=embedded_nodes,
        embed_model=embed_model,
    )

    save_vector_index(
        index=index,
        persist_dir=config.vector_index_dir,
    )

    return index, embedded_nodes


def load_or_build_vector_index(
    config: AppConfig,
    embed_model: BaseEmbedding,
) -> tuple[VectorStoreIndex, list[BaseNode]]:

    if not isinstance(config, APP_CONFIG_TYPE):
        raise TypeError("config должен быть объектом AppConfig!")

    if not isinstance(embed_model, BaseEmbedding):
        raise TypeError("embed_model должен быть объектом BaseEmbedding!")

    if not config.rebuild_index and config.vector_index_dir.exists():
        index = load_vector_index(
            persist_dir=config.vector_index_dir, embed_model=embed_model
        )

        nodes = extract_nodes_from_index(index)

        return index, nodes

    return build_new_vector_index(
        config=config,
        embed_model=embed_model,
    )


def build_rag_pipeline(
    config: AppConfig,
) -> RAGPipeline:

    if not isinstance(config, APP_CONFIG_TYPE):
        raise TypeError("config должен быть объектом AppConfig!")

    embed_model = create_e5_embed_model(
        device=config.embedding_device,
    )

    index, nodes = load_or_build_vector_index(
        config=config,
        embed_model=embed_model,
    )

    bm25 = build_bm25_retriever(
        nodes=nodes,
        top_k=config.pipeline_config.retrieval_top_k,
    )

    reranker = create_cross_encoder(
        device=config.reranker_device,
    )

    llm = create_saiga_llm(
        model_path=config.saiga_model_path,
        n_ctx=config.n_ctx,
        n_gpu_layers=config.n_gpu_layers,
        n_threads=config.n_threads,
    )

    nemo = create_nemo_guardrails_adapter(config_path=config.nemo_config_dir, llm=llm)

    pipeline = RAGPipeline(
        index=index,
        bm25_retriever=bm25,
        reranker=reranker,
        llm=llm,
        config=config.pipeline_config,
        nemo_guardrails=nemo,
    )

    return pipeline
