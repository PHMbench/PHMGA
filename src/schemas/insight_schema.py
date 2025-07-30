from __future__ import annotations

from typing import Tuple
from pydantic import BaseModel, Field
import uuid

class AnalysisInsight(BaseModel):
    """Insight generated from comparing nodes."""

    insight_id: str = Field(default_factory=lambda: f"ins_{uuid.uuid4().hex[:8]}")
    content: str
    severity_score: float = Field(ge=0.0, le=1.0)
    compared_nodes: Tuple[str, str] = Field(..., description="Tuple of (reference_node_id, test_node_id) that were compared.")
