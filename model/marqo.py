from typing import Optional

from pydantic import BaseModel


class MarqoQuerySchema(BaseModel):
    title: Optional[str] = None
    abstract: Optional[str] = None
    author: Optional[list] = None
    source: Optional[list] = None
    keyword: Optional[list] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    id_list: Optional[list] = None
    limit: int = 20
    offset: int = 0
    min_citation_counts: Optional[int] = None
    max_citation_counts: Optional[int] = None
