from pydantic import BaseModel, Field, validator
from typing import Optional, List
from pathlib import Path
import os

class DocumentChunk(BaseModel):
    text: str = Field(..., min_length=1)
    page: Optional[int] = Field(None, ge=1)
    filepath: str
    chunk_index: int = Field(..., ge=0)

    @validator('filepath')
    def validate_filepath(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"File path does not exist: {v}")
        return v

class DocumentMetadata(BaseModel):
    filepath: str
    original_text: str = Field(..., min_length=1)
    chunk_index: int = Field(..., ge=0)
    page: Optional[int] = Field(None, ge=1)

class DocumentCluster(BaseModel):
    passages: List[str] = Field(..., min_items=1)
    embeddings: List[List[float]] = Field(..., min_items=1)

class APIResponse(BaseModel):
    role: str = Field(..., regex="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)

class ConversationHistory(BaseModel):
    messages: List[APIResponse] = Field(..., min_items=1)
