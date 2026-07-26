from llama_index.core import Document

from .models import Article


def articles_to_document(articles: list[Article]) -> list[Document]:
    """Преобразует статьи в документы LlamaIndex."""
