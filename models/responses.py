from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class SearchResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    summary: str
    statistics: Dict[str, Any]


class GraphNode(BaseModel):
    id: str
    label: str
    type: str  # summary, collection, hadith, narrator, themeгде responses.p
    collection: Optional[str] = None
    tooltip: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class GraphEdge(BaseModel):
    source: str
    target: str
    color: str
    type: Optional[str] = None  # narrator, similarity, hierarchical


class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    summary: str


class CollectionInfo(BaseModel):
    name: str
    display_name: str
    description: str
    hadith_count: int


class HadithDetail(BaseModel):
    uid_chunked: str
    content: str
    arabic_text: Optional[str] = None
    collection: str
    book: Optional[str] = None
    chapter: Optional[str] = None
    hadith_number: Optional[str] = None
    narrator: Optional[str] = None
    grade: Optional[str] = None
    themes: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    related_hadiths: Optional[List[Dict[str, Any]]] = None


class NarratorInfo(BaseModel):
    name: str
    hadith_count: int
    collections: List[str]
    biography: Optional[str] = None
    reliability: Optional[str] = None
    period: Optional[str] = None
    top_themes: Optional[List[str]] = None