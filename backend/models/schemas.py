"""
Pydantic v2 request / response schemas for EduMind AI.
"""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


# ── RAG ─────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks_indexed: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What is photosynthesis?"])
    top_k: int = Field(default=3, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    model_used: str
    response_time_ms: float


class RAGStatusResponse(BaseModel):
    indexed: bool
    chunks_count: int
    sources: list[str]


# ── Performance predictor ────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    attendance_pct: float = Field(..., ge=0, le=100, examples=[85])
    study_hours_per_day: float = Field(..., ge=0, le=24, examples=[3.5])
    prev_exam_score: float = Field(..., ge=0, le=100, examples=[72])
    assignments_completed_pct: float = Field(..., ge=0, le=100, examples=[80])
    sleep_hours: float = Field(..., ge=0, le=24, examples=[7])
    extracurricular_activities: int = Field(default=0, ge=0, le=10, examples=[2])
    subject: str = Field(default="General", examples=["Math"])


class PredictResponse(BaseModel):
    predicted_grade: str
    predicted_score: float
    risk_level: str
    recommendations: list[str]
    feature_importance: dict[str, float]


# ── Topic recommender ────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    student_id: str = Field(..., examples=["STU001"])
    completed_topics: list[str] = Field(default_factory=list, examples=[["Integers", "Fractions"]])
    weak_subjects: list[str] = Field(default_factory=list, examples=[["Math"]])
    grade_level: int = Field(..., ge=6, le=10, examples=[9])
    learning_style: str = Field(default="mixed", examples=["visual"])


class TopicItem(BaseModel):
    topic: str
    subject: str
    grade: int
    difficulty: str
    description: str
    similarity_score: float


class RecommendResponse(BaseModel):
    recommended_topics: list[TopicItem]
    daily_study_plan: dict[str, list[str]]
    motivational_message: str
    total_topics_available: int
