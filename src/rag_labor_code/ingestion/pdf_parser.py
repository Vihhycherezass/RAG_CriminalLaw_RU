from pathlib import Path

import pdfplumber

from .models import Article

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Извлекает текст из всех страниц PDF."""
    if not pdf_path.exists():
        raise FileNotFoundError("Файл не найден!")
    
    if not pdf_path.is_file():
        raise ValueError("Указанный путь не является файлом!")

    if pdf_path.suffix.lower() != '.pdf':
        raise ValueError("Расширение файла не .pdf")
    
    page_texts = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text.strip():
              page_texts.append(page_text)
        if not page_texts:
            raise ValueError("Не удалось извлечь текст из файла!")  
        
    return "\n\n".join(page_texts)
    
def normalize_text(text: str) -> str:
    """Нормализуем извлеченный из PDF текст"""
    pass

def parse_articles(text: str) -> list[Article]:
    """Разделяет нормализованный текст на статьи."""
    pass