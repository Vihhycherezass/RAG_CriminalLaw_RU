import re
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
    
    page_texts: list[str] = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                page_texts.append(page_text)
        if not page_texts:
            raise ValueError("Не удалось извлечь текст из файла!")  
        
    return "\n\n".join(page_texts)
     
    
def normalize_text(text: str) -> str:
    """Нормализуем извлеченный из PDF текст"""
    if not text or not text.strip():
        raise ValueError("Текст не должен быть пустым!")
    
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    
    text = text.replace("\xa0", " ")
    
    text = text.replace("\u00ad", "")
    
    normalized_lines = []
    
    for line in text.splitlines():
        normalized_line = re.sub(r"[ \t]+", " ", line).strip()
        normalized_lines.append(normalized_line)
        
    normalized_text = '\n'.join(normalized_lines)
    normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text).strip()
    
    if not normalized_text:
        ValueError("Результат пустой!")
    
    return normalized_text


def parse_articles(text: str) -> list[Article]:
    """Разделяет нормализованный текст на статьи."""
    pass