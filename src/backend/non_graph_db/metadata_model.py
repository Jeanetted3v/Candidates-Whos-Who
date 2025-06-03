from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class EmbeddingMetadata(BaseModel):
    """Structured metadata extracted from election-related text."""
    
    # Primary categorization
    primary_category: str = Field(
        description="Main category of the content (e.g., 'candidate_profile', 'party_policy', 'constituency_overview')"
    )
    
    # Core entities
    entity_names: List[str] = Field(
        description="All named entities mentioned in the text (people, parties, places, organizations)",
        default_factory=list
    )
    
    # Extracted keywords by type
    keywords: Optional[List[str]] = Field(
        description="Topical keywords extracted from the text (excluding named entities)",
        default_factory=None,
    )
    
    # Related topics for broader context
    related_topics: List[str] = Field(
        description="Broader topics related to this content that may be useful for retrieval",
        default_factory=list
    )