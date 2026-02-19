from typing import List
from pydantic import BaseModel, Field


class Source(BaseModel):
    """Source of the information"""

    url: str = Field(description="URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Answer")
    sources: List[Source] = Field(
        description="List of sources used to generate the answer", default_factory=list
    )
