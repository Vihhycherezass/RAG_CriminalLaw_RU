import re
from pathlib import Path

import pdfplumber

from .models import Article


ARTICLE_HEADER_PATTERN = re.compile(
    r"^Статья\s+(?P<number>\d+(?:\.\d+)*)\.\s*(?P<title>[^\n]*)$",
    flags=re.MULTILINE,
)


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
    """Нормализует текст, извлеченный из PDF"""
    if not text or not text.strip():
        raise ValueError("Текст не должен быть пустым!")
    
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    
    text = text.replace("\xa0", " ")
    
    text = text.replace("\u00ad", "")
    
    normalized_lines: list[str] = []
    
    for line in text.splitlines():
        normalized_line = re.sub(r"[ \t]+", " ", line).strip()
        normalized_lines.append(normalized_line)
        
    normalized_text = "\n".join(normalized_lines)
    normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text).strip()
    
    if not normalized_text:
        raise ValueError("После нормализации текст оказался пустым!")
    
    return normalized_text


def parse_articles(text: str, source: str = "Трудовой кодекс Российской Федерации") -> list[Article]:
    """Разделяет нормализованный текст на статьи."""
    if not text or not text.strip():
        raise ValueError("Текст для парсинга не должен быть пустым!")
    
    if not source or not source.strip():
        raise ValueError("Источник не должен быть пустым!")
    
    matches = list(ARTICLE_HEADER_PATTERN.finditer(text))
    
    if not matches:
        raise ValueError("В тексте не найдены заголовки статьи!")
    
    articles: list[Article] = []
    
    for i, match in enumerate(matches):
        article_num = match.group("number").strip() 
        title = match.group("title").strip()
        
        content_start = match.end()
        
        if i + 1 < len(matches):
            content_end = matches[i+1].start()
        else:
            content_end = len(text)
            
        content:str = text[content_start:content_end].strip()
                
        article = Article(
            article_num=article_num,
            title=title,
            content=content,
            source=source
        )
        
        articles.append(article)
        
    return articles
