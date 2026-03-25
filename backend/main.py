"""
EduMind AI — FastAPI application entry point.
Startup warms up ML models; middleware handles logging + errors.
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Allow running as `uvicorn backend.main:app` from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from middleware import RequestLoggingMiddleware, add_error_handlers
from backend.routers import rag, predict, recommend
from backend.services.rag_service import rag_service
from backend.services.predict_service import predict_service
from backend.services.recommend_service import recommend_service

logger = logging.getLogger("edumind")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm all models at startup."""
    logger.info("🚀 EduMind AI starting up …")
    rag_service.warmup()
    predict_service.warmup()
    recommend_service.warmup()
    logger.info("✅ All models ready. Visit http://localhost:8000/docs")
    yield
    logger.info("👋 EduMind AI shutting down.")


app = FastAPI(
    title="EduMind AI",
    description=(
        "Intelligent School Learning Platform — "
        "RAG Q&A · Performance Prediction · Adaptive Topic Recommendations"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
add_error_handlers(app)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(rag.router)
app.include_router(predict.router)
app.include_router(recommend.router)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service": "EduMind AI"}


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Welcome to EduMind AI 🎓",
        "docs": "/docs",
        "endpoints": {
            "rag_ingest": "POST /api/rag/ingest",
            "rag_ask": "POST /api/rag/ask",
            "rag_status": "GET /api/rag/status",
            "predict": "POST /api/predict/performance",
            "recommend": "POST /api/recommend/topics",
        },
    }
