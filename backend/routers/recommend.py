"""Topic recommendation router."""
from fastapi import APIRouter, HTTPException
from backend.models.schemas import RecommendRequest, RecommendResponse
from backend.services.recommend_service import recommend_service

router = APIRouter(prefix="/api/recommend", tags=["Topic Recommender"])


@router.post("/topics", response_model=RecommendResponse)
async def recommend_topics(req: RecommendRequest):
    """Get personalised topic recommendations and a 7-day study plan."""
    try:
        result = recommend_service.recommend(req.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return RecommendResponse(**result)
