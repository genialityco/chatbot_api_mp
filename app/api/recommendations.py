from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.auth import get_platform_context
from app.models.recommendation import RecommendationRequest, RecommendationResponse
from app.services.recommendation_service import RecommendationService

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


@router.post("", response_model=RecommendationResponse)
async def recommend(
    request: RecommendationRequest,
    ctx: dict = Depends(get_platform_context),
):
    platform = ctx["platform"]
    org_id = request.org_id or ctx["org_id"]
    service = RecommendationService(platform=platform, org_id=org_id)
    recommendations, keywords = await service.recommend_for_user(
        user_id=request.user_id,
        limit=request.limit,
    )
    return RecommendationResponse(
        platform_id=ctx["platform_id"],
        user_id=request.user_id,
        based_on=keywords,
        recommendations=recommendations,
    )
