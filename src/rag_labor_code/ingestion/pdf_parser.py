from pathlib import Path
from .models import Article

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Извлекает текст из всех страниц PDF."""
    pass

def normalize_text(text: str) -> str:
    """Нормализуем извлеченный из PDF текст"""
    pass

def parse_articles(text: str) -> list[Article]:
    """Разделяет нормализованный текст на статьи."""
    pass