from llama_index.core import Document

from .models import Article
from .validation import validate_articles


def articles_to_documents(articles: list[Article]) -> list[Document]:
    """Преобразует статьи в документы LlamaIndex."""
    validate_articles(articles)

    documents: list[Document] = []

    for article in articles:
        title = f"Статья {article.article_num}. {article.title}"

        full_text = f"{title}\n\n{article.content}"

        metadata = {
            "article_num": article.article_num,
            "title": article.title,
            "source": article.source,
            "document_type": "labor_code_article",
        }

        document = Document(text=full_text, metadata=metadata)

        documents.append(document)

    return documents
