import pytest
from llama_index.core import Document
from llama_index.core.schema import BaseNode


from rag_labor_code.ingestion.node_builder import documents_to_nodes


def test_documents_to_nodes_rejects_empty_documents() -> None:
    with pytest.raises(ValueError, match="Список документов пуст!"):
        documents_to_nodes([])


def test_documents_to_nodes_creates_nodes() -> None:
    long_text = (
        "Рабочее время — время, в течение которого работник "
        "исполняет трудовые обязанности. "
    ) * 100

    documents = [
        Document(
            text=long_text,
            metadata={
                "article_num": "91",
            },
        )
    ]

    nodes = documents_to_nodes(
        documents=documents,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert nodes

    for node in nodes:
        assert isinstance(node, BaseNode)


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 150),
    ],
)
def test_documents_to_nodes_rejects_invalid_chunk_parameters(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    documents = [
        Document(
            text="Рабочее время — время исполнения трудовых обязанностей.",
        )
    ]

    with pytest.raises(ValueError):
        documents_to_nodes(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_documents_to_nodes_splits_long_document() -> None:
    long_text = (
        "Рабочее время — время, в течение которого работник "
        "исполняет трудовые обязанности. "
    ) * 50

    documents = [
        Document(
            text=long_text,
            metadata={
                "article_num": "91",
            },
        )
    ]

    nodes = documents_to_nodes(
        documents=documents,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(nodes) > 1


def test_documents_to_nodes_preserves_metadata() -> None:
    long_text = (
        "Рабочее время — время, в течение которого работник "
        "исполняет трудовые обязанности. "
    ) * 10

    documents = [
        Document(
            text=long_text,
            metadata={
                "article_num": "91",
                "title": "Понятие рабочего времени",
                "source": "ТК РФ",
                "document_type": "labor_code_article",
            },
        )
    ]

    nodes = documents_to_nodes(
        documents=documents,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert nodes

    for node in nodes:
        assert node.metadata["article_num"] == "91"
        assert node.metadata["title"] == "Понятие рабочего времени"
        assert node.metadata["source"] == "ТК РФ"
        assert node.metadata["document_type"] == "labor_code_article"


def test_documents_to_nodes_creates_non_empty_nodes() -> None:
    long_text = (
        "Рабочее время — время, в течение которого работник "
        "исполняет трудовые обязанности. "
    ) * 100

    documents = [
        Document(
            text=long_text,
            metadata={
                "article_num": "91",
                "title": "Понятие рабочего времени",
                "source": "ТК РФ",
                "document_type": "labor_code_article",
            },
        )
    ]

    nodes = documents_to_nodes(
        documents=documents,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert nodes

    for node in nodes:
        assert node.get_content().strip()


def test_documents_to_nodes_keeps_article_metadata_separate() -> None:
    article_91_text = (
        "УНИКАЛЬНЫЙ_ТЕКСТ_СТАТЬИ_91. " "Рабочее время регулируется статьёй 91. "
    ) * 50

    article_92_text = (
        "УНИКАЛЬНЫЙ_ТЕКСТ_СТАТЬИ_92. "
        "Нормальная продолжительность рабочего времени регулируется статьёй 92. "
    ) * 50

    documents = [
        Document(
            text=article_91_text,
            metadata={
                "article_num": "91",
                "title": "Понятие рабочего времени",
            },
        ),
        Document(
            text=article_92_text,
            metadata={
                "article_num": "92",
                "title": "Нормальная продолжительность рабочего времени",
            },
        ),
    ]

    nodes = documents_to_nodes(
        documents=documents,
        chunk_size=100,
        chunk_overlap=20,
    )

    article_numbers = {node.metadata["article_num"] for node in nodes}

    assert article_numbers == {"91", "92"}

    for node in nodes:
        article_num = node.metadata["article_num"]
        node_text = node.get_content()

        if article_num == "91":
            assert "УНИКАЛЬНЫЙ_ТЕКСТ_СТАТЬИ_91" in node_text
            assert "УНИКАЛЬНЫЙ_ТЕКСТ_СТАТЬИ_92" not in node_text

        elif article_num == "92":
            assert "УНИКАЛЬНЫЙ_ТЕКСТ_СТАТЬИ_92" in node_text
            assert "УНИКАЛЬНЫЙ_ТЕКСТ_СТАТЬИ_91" not in node_text

        else:
            pytest.fail(f"Получен узел с неизвестным номером статьи: {article_num}")
