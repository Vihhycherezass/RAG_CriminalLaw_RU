from dataclasses import dataclass

@dataclass(frozen=True)
class Article:
    article_num: str
    title: str
    content: str
    source: str