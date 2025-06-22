from pydantic import BaseModel
from typing import Optional


class SearchResult(BaseModel):
    title: str
    url: str
    summary: str
    query: Optional[str] = None

    def to_dict(self):
        return {
            "title": self.title,
            "url": self.url,
            "summary": self.summary,
        }
