"""
Student Performance Prediction Service.
GradientBoosting + RandomForest soft-voting ensemble.
Trained on 8 000 synthetic student records.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("edumind.predict")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

GRADE_BINS   = [0,  40,  50,  60,  70,  80,  90, 101]
GRADE_LABELS = ["F", "D", "C", "B-", "B", "A-", "A+"]


class PredictService:
    def __init__(self):
        self._model = None
        self._label_enc = None

    # ── Training ──────────────────────────────────────────────────────────────

    def _train(self):
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split

        logger.info("Training improved model on %d samples …", config.N_TRAIN_SAMPLES)
        rng = np.random.default_rng(config.RANDOM_STATE)
        n = config.N_TRAIN_SAMPLES

        attendance       = rng.uniform(40, 100, n)
        study_hours      = rng.uniform(0.5, 8, n)
        prev_score       = rng.uniform(30, 100, n)
        assignments      = rng.uniform(30, 100, n)
        sleep_hours      = rng.uniform(4, 10, n)
        extracurricular  = rng.integers(0, 6, n).astype(float)

        # Base score with realistic correlations
        base = (
            0.28 * prev_score
            + 0.22 * attendance
            + 0.18 * study_hours * 10
            + 0.15 * assignments
            + 0.07 * sleep_hours * 5
            - 0.03 * extracurricular * 2
            + rng.normal(0, 6, n)
        )
        score = np.clip(base, 0, 100)

        grade_idx = np.digitize(score, GRADE_BINS[1:-1])
        grade_labels = [GRADE_LABELS[i] for i in grade_idx]

        X = self._engineer(
            attendance, study_hours, prev_score, assignments, sleep_hours, extracurricular
        )

        le = LabelEncoder()
        y = le.fit_transform(grade_labels)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_STATE
        )

        gb = GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05,
            max_depth=4, random_state=config.RANDOM_STATE
        )
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=6,
            random_state=config.RANDOM_STATE, n_jobs=-1
        )
        gb.fit(X_tr, y_tr)
        rf.fit(X_tr, y_tr)

        train_acc = np.mean(gb.predict(X_tr) == y_tr)
        test_acc  = np.mean(gb.predict(X_te) == y_te)
        logger.info("Train: %.0f%% | Test: %.0f%%", train_acc * 100, test_acc * 100)

        self._gb = gb
        self._rf = rf
        self._label_enc = le
        self._model = gb  # primary

    @staticmethod
    def _engineer(attendance, study_hours, prev_score, assignments, sleep_hours, extracurricular):
        study_x_attendance = study_hours * attendance / 100
        prev_x_assignments = prev_score * assignments / 100
        sleep_study_ratio  = np.where(study_hours > 0, sleep_hours / study_hours, 0)
        consistency_score  = (attendance + assignments) / 2

        return np.column_stack([
            attendance, study_hours, prev_score, assignments,
            sleep_hours, extracurricular,
            study_x_attendance, prev_x_assignments,
            sleep_study_ratio, consistency_score,
        ])

    def warmup(self):
        if self._model is None:
            self._train()

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, data: dict) -> dict:
        self.warmup()

        att  = data["attendance_pct"]
        sh   = data["study_hours_per_day"]
        prev = data["prev_exam_score"]
        asgn = data["assignments_completed_pct"]
        slp  = data["sleep_hours"]
        ec   = float(data.get("extracurricular_activities", 0))

        X = self._engineer(
            np.array([att]), np.array([sh]), np.array([prev]),
            np.array([asgn]), np.array([slp]), np.array([ec])
        )

        # Soft voting
        gb_proba = self._gb.predict_proba(X)
        rf_proba = self._rf.predict_proba(X)
        avg_proba = (gb_proba + rf_proba) / 2
        pred_idx = int(np.argmax(avg_proba[0]))
        grade = self._label_enc.inverse_transform([pred_idx])[0]

        # Estimated score
        score_map = {"F": 35, "D": 45, "C": 55, "B-": 65, "B": 75, "A-": 85, "A+": 94}
        predicted_score = score_map.get(grade, 60) + np.random.uniform(-2, 2)

        risk = self._risk_level(att, prev, asgn)
        recs = self._recommendations(grade, att, sh, prev, asgn, slp)

        feature_names = [
            "attendance_pct", "study_hours", "prev_exam_score",
            "assignments_pct", "sleep_hours", "extracurricular",
            "study_x_attendance", "prev_x_assignments",
            "sleep_study_ratio", "consistency_score",
        ]
        importance = dict(zip(feature_names, self._gb.feature_importances_.tolist()))
        importance = dict(sorted(importance.items(), key=lambda x: -x[1])[:6])

        return {
            "predicted_grade": grade,
            "predicted_score": round(float(predicted_score), 1),
            "risk_level": risk,
            "recommendations": recs,
            "feature_importance": {k: round(v, 3) for k, v in importance.items()},
        }

    @staticmethod
    def _risk_level(attendance, prev_score, assignments) -> str:
        score = (attendance + prev_score + assignments) / 3
        if score >= 75:
            return "Low"
        if score >= 55:
            return "Medium"
        return "High"

    @staticmethod
    def _recommendations(grade, attendance, study_hours, prev_score, assignments, sleep) -> list[str]:
        recs = []
        if attendance < 75:
            recs.append("⚠️ Attendance below 75% — aim for 85%+ to stay on track.")
        if study_hours < 3:
            recs.append("📚 Increase daily study time to at least 3 hours for consistent improvement.")
        if prev_score < 60:
            recs.append("📝 Focus on revising fundamentals — use NCERT examples and practice problems.")
        if assignments < 70:
            recs.append("✅ Complete 90%+ assignments; they directly reinforce classroom learning.")
        if sleep < 6:
            recs.append("😴 Sleep 7–8 hours nightly — it significantly improves memory retention.")
        if grade in ("A-", "A+"):
            recs.append("🌟 Excellent performance! Try advanced problems or Olympiad preparation.")
        if not recs:
            recs.append("👍 Keep up the good work! Maintain your current study habits.")
        return recs


predict_service = PredictService()
