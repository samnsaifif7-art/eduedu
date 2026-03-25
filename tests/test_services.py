"""
15 unit tests for EduMind AI services.
Run with: pytest tests/ -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# PredictService
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictService:
    @pytest.fixture(autouse=True)
    def setup(self):
        from backend.services.predict_service import PredictService
        self.svc = PredictService()
        self.svc.warmup()

    def _sample(self, **overrides):
        base = {
            "attendance_pct": 85,
            "study_hours_per_day": 3.5,
            "prev_exam_score": 72,
            "assignments_completed_pct": 80,
            "sleep_hours": 7,
            "extracurricular_activities": 2,
            "subject": "Math",
        }
        base.update(overrides)
        return base

    def test_predict_returns_valid_grade(self):
        result = self.svc.predict(self._sample())
        assert result["predicted_grade"] in ["F", "D", "C", "B-", "B", "A-", "A+"]

    def test_predict_score_in_range(self):
        result = self.svc.predict(self._sample())
        assert 0 <= result["predicted_score"] <= 100

    def test_predict_risk_level_valid(self):
        result = self.svc.predict(self._sample())
        assert result["risk_level"] in ["Low", "Medium", "High"]

    def test_predict_recommendations_nonempty(self):
        result = self.svc.predict(self._sample())
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) >= 1

    def test_predict_feature_importance_present(self):
        result = self.svc.predict(self._sample())
        assert isinstance(result["feature_importance"], dict)
        assert len(result["feature_importance"]) > 0

    def test_poor_student_high_risk(self):
        result = self.svc.predict(self._sample(
            attendance_pct=40, prev_exam_score=35, assignments_completed_pct=40
        ))
        assert result["risk_level"] == "High"

    def test_excellent_student_low_risk(self):
        result = self.svc.predict(self._sample(
            attendance_pct=98, prev_exam_score=95, assignments_completed_pct=99
        ))
        assert result["risk_level"] == "Low"

    def test_low_attendance_triggers_recommendation(self):
        result = self.svc.predict(self._sample(attendance_pct=50))
        combined = " ".join(result["recommendations"])
        assert "ttendance" in combined or "75" in combined


# ─────────────────────────────────────────────────────────────────────────────
# RecommendService
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommendService:
    @pytest.fixture(autouse=True)
    def setup(self):
        from backend.services.recommend_service import RecommendService
        self.svc = RecommendService()
        self.svc.warmup()

    def _sample(self, **overrides):
        base = {
            "student_id": "STU001",
            "completed_topics": ["Integers", "Fractions"],
            "weak_subjects": ["Math"],
            "grade_level": 9,
            "learning_style": "visual",
        }
        base.update(overrides)
        return base

    def test_returns_recommendations(self):
        result = self.svc.recommend(self._sample())
        assert len(result["recommended_topics"]) > 0

    def test_completed_topics_excluded(self):
        completed = ["Integers", "Fractions and Decimals"]
        result = self.svc.recommend(self._sample(completed_topics=completed))
        topics = [t["topic"] for t in result["recommended_topics"]]
        for done in completed:
            assert done not in topics

    def test_seven_day_plan_keys(self):
        result = self.svc.recommend(self._sample())
        days = result["daily_study_plan"]
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
            assert day in days

    def test_motivational_message_nonempty(self):
        result = self.svc.recommend(self._sample())
        assert isinstance(result["motivational_message"], str)
        assert len(result["motivational_message"]) > 5

    def test_weak_subject_topics_boosted(self):
        result = self.svc.recommend(self._sample(weak_subjects=["Math"]))
        subjects = [t["subject"] for t in result["recommended_topics"][:5]]
        assert "Math" in subjects

    def test_total_topics_available_positive(self):
        result = self.svc.recommend(self._sample())
        assert result["total_topics_available"] > 0

    def test_similarity_scores_in_range(self):
        result = self.svc.recommend(self._sample())
        for t in result["recommended_topics"]:
            assert 0.0 <= t["similarity_score"] <= 2.0  # can exceed 1 with boost


# ─────────────────────────────────────────────────────────────────────────────
# RAGService (unit tests — no PDF required)
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGService:
    @pytest.fixture(autouse=True)
    def setup(self):
        from backend.services.rag_service import RAGService
        self.svc = RAGService()

    def test_status_initially_not_indexed(self):
        status = self.svc.status()
        assert status["indexed"] is False
        assert status["chunks_count"] == 0

    def test_ask_before_ingest_raises(self):
        with pytest.raises(RuntimeError):
            self.svc.ask("What is photosynthesis?")

    def test_chunk_creation(self):
        text = " ".join([f"word{i}" for i in range(600)])
        chunks = self.svc._make_chunks(text)
        assert len(chunks) > 1
        assert all(len(c.split()) <= 250 for c in chunks)
