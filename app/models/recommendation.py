from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: str
    limit: int = 5
    org_id: str | None = None


class RecommendationItem(BaseModel):
    collection: str
    item_id: str
    title: str
    summary: str = ""
    score: float = 0.0
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    platform_id: str
    user_id: str
    based_on: list[str] = Field(default_factory=list)
    recommendations: list[RecommendationItem] = Field(default_factory=list)
