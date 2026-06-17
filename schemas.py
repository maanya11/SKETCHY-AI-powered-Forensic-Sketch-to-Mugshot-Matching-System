from pydantic import BaseModel, Field
from typing import Optional, List

class SuspectCreate(BaseModel):
    suspect_id: str
    case_id: Optional[str] = None
    name: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class IndexStatus(BaseModel):
    indexed: int
    index_path: str
