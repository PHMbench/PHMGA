from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field
import uuid

class AnalysisPlan(BaseModel):
    """List of coarse-grained analysis steps."""

    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    steps: List[str] = Field(default_factory=list)
