# model/chroma.py

from typing import Optional, List
from pydantic import BaseModel

class ChromaQuerySchema(BaseModel):
    title: Optional[str] = None
    abstract: Optional[str] = None
    author: Optional[List[str]] = None
    source: Optional[List[str]] = None
    keyword: Optional[List[str]] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    id_list: Optional[List[str]] = None
    limit: int = 20
    offset: int = 0
    min_citation_counts: Optional[int] = None
    max_citation_counts: Optional[int] = None