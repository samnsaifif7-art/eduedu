"""Performance prediction router."""
from fastapi import APIRouter, HTTPException
from backend.models.schemas import PredictRequest, PredictResponse
from backend.services.predict_service import predict_service

router = APIRouter(prefix="/api/predict", tags=["Performance Prediction"])


@router.post("/performance", response_model=PredictResponse)
async def predict_performance(req: PredictRequest):
    """Predict student grade, risk level, and get recommendations."""
    try:
        result = predict_service.predict(req.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return PredictResponse(**result)
