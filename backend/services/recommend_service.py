"""
Adaptive Topic Recommender — TF-IDF cosine similarity on NCERT curriculum.
"""
from __future__ import annotations

import random
import logging
from typing import Any

logger = logging.getLogger("edumind.recommend")

# ── NCERT Curriculum Data (Grades 6-10) ──────────────────────────────────────

CURRICULUM: list[dict] = [
    # ── Math ────────────────────────────────────────────────────────────────
    {"topic": "Integers", "subject": "Math", "grade": 6, "difficulty": "Easy",
     "description": "Negative numbers, number line, addition and subtraction of integers, properties"},
    {"topic": "Fractions and Decimals", "subject": "Math", "grade": 6, "difficulty": "Easy",
     "description": "Operations on fractions, multiplication division decimals word problems"},
    {"topic": "Algebra Basics", "subject": "Math", "grade": 7, "difficulty": "Medium",
     "description": "Variables expressions equations simple linear equations algebraic identities"},
    {"topic": "Geometry Triangles", "subject": "Math", "grade": 7, "difficulty": "Medium",
     "description": "Properties triangles congruence similarity Pythagoras theorem angle sum"},
    {"topic": "Rational Numbers", "subject": "Math", "grade": 8, "difficulty": "Medium",
     "description": "Properties of rational numbers representation number line operations"},
    {"topic": "Linear Equations in Two Variables", "subject": "Math", "grade": 8, "difficulty": "Medium",
     "description": "Graphical method substitution elimination cross multiplication applications"},
    {"topic": "Polynomials", "subject": "Math", "grade": 9, "difficulty": "Hard",
     "description": "Degree zeros remainder theorem factor theorem algebraic identities"},
    {"topic": "Coordinate Geometry", "subject": "Math", "grade": 9, "difficulty": "Medium",
     "description": "Cartesian plane distance formula section formula area of triangle"},
    {"topic": "Quadratic Equations", "subject": "Math", "grade": 10, "difficulty": "Hard",
     "description": "Standard form factorization completing the square quadratic formula discriminant"},
    {"topic": "Arithmetic Progressions", "subject": "Math", "grade": 10, "difficulty": "Medium",
     "description": "Nth term sum of n terms applications real world problems"},
    {"topic": "Trigonometry", "subject": "Math", "grade": 10, "difficulty": "Hard",
     "description": "Trigonometric ratios identities heights distances applications"},
    {"topic": "Statistics and Probability", "subject": "Math", "grade": 10, "difficulty": "Medium",
     "description": "Mean median mode grouped data probability events sample space"},

    # ── Science ─────────────────────────────────────────────────────────────
    {"topic": "Cell Biology", "subject": "Science", "grade": 8, "difficulty": "Medium",
     "description": "Cell structure organelles plant animal cells cell division mitosis meiosis"},
    {"topic": "Photosynthesis", "subject": "Science", "grade": 7, "difficulty": "Easy",
     "description": "Chlorophyll light dark reactions glucose oxygen carbon dioxide process"},
    {"topic": "Force and Motion", "subject": "Science", "grade": 8, "difficulty": "Medium",
     "description": "Newton laws friction pressure speed velocity acceleration graphs"},
    {"topic": "Chemical Reactions", "subject": "Science", "grade": 10, "difficulty": "Hard",
     "description": "Types reactions balancing equations reactants products acid base salts"},
    {"topic": "Electricity", "subject": "Science", "grade": 10, "difficulty": "Hard",
     "description": "Ohm law resistance series parallel circuits power Joule heating"},
    {"topic": "Light Optics", "subject": "Science", "grade": 10, "difficulty": "Hard",
     "description": "Reflection refraction lenses mirrors image formation human eye defects"},
    {"topic": "Heredity Evolution", "subject": "Science", "grade": 10, "difficulty": "Hard",
     "description": "Mendel laws chromosomes DNA variation natural selection speciation"},
    {"topic": "Nutrition in Plants", "subject": "Science", "grade": 7, "difficulty": "Easy",
     "description": "Autotrophs heterotrophs parasitic saprophytic symbiotic nutrition modes"},
    {"topic": "Sound", "subject": "Science", "grade": 8, "difficulty": "Medium",
     "description": "Vibration wave propagation frequency amplitude loudness pitch echo"},
    {"topic": "Atoms and Molecules", "subject": "Science", "grade": 9, "difficulty": "Medium",
     "description": "Laws of chemical combination atomic molecular mass mole concept formulae"},

    # ── English ─────────────────────────────────────────────────────────────
    {"topic": "Grammar - Tenses", "subject": "English", "grade": 6, "difficulty": "Easy",
     "description": "Present past future perfect continuous tense forms verb conjugation"},
    {"topic": "Reading Comprehension", "subject": "English", "grade": 8, "difficulty": "Medium",
     "description": "Inference main idea vocabulary context clues passage analysis questions"},
    {"topic": "Essay Writing", "subject": "English", "grade": 9, "difficulty": "Medium",
     "description": "Introduction body conclusion paragraph structure argumentative descriptive"},
    {"topic": "Literature Analysis", "subject": "English", "grade": 10, "difficulty": "Hard",
     "description": "Theme character plot setting symbolism metaphor poetic devices"},

    # ── Social Science ───────────────────────────────────────────────────────
    {"topic": "French Revolution", "subject": "Social Science", "grade": 9, "difficulty": "Medium",
     "description": "Causes Estates liberty equality fraternity Robespierre Napoleon aftermath"},
    {"topic": "Indian Independence Movement", "subject": "Social Science", "grade": 10, "difficulty": "Medium",
     "description": "Non-cooperation Civil Disobedience Quit India Gandhi Nehru Partition"},
    {"topic": "Democracy and Constitution", "subject": "Social Science", "grade": 9, "difficulty": "Medium",
     "description": "Fundamental rights DPSP federal structure parliament elections judiciary"},
    {"topic": "Climate and Agriculture", "subject": "Social Science", "grade": 8, "difficulty": "Easy",
     "description": "Monsoon crop patterns Kharif Rabi food security land use irrigation"},
    {"topic": "Globalisation and Economy", "subject": "Social Science", "grade": 10, "difficulty": "Hard",
     "description": "MNCs trade liberalisation WTO impact India consumers workers"},

    # ── Hindi ────────────────────────────────────────────────────────────────
    {"topic": "व्याकरण - संज्ञा और सर्वनाम", "subject": "Hindi", "grade": 6, "difficulty": "Easy",
     "description": "संज्ञा के भेद सर्वनाम वचन लिंग कारक विभक्ति"},
    {"topic": "पद्य - कविता विश्लेषण", "subject": "Hindi", "grade": 9, "difficulty": "Medium",
     "description": "कविता की भाषा भाव अलंकार छंद रस काव्य सौंदर्य"},
    {"topic": "गद्य - निबंध लेखन", "subject": "Hindi", "grade": 10, "difficulty": "Medium",
     "description": "रूपरेखा प्रस्तावना विस्तार उपसंहार विचारात्मक वर्णनात्मक निबंध"},
]

MOTIVATIONAL_MESSAGES = [
    "Consistency beats talent. Keep showing up! 🎯",
    "Every expert was once a beginner. You're on the right path! 🌟",
    "Small daily improvements lead to stunning results. Keep going! 💪",
    "Your future self will thank you for studying today! 📚",
    "Progress, not perfection. One topic at a time! ✨",
    "The secret of getting ahead is getting started. You've got this! 🚀",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class RecommendService:
    def __init__(self):
        self._vectorizer = None
        self._tfidf_matrix = None

    def _build_tfidf(self):
        if self._vectorizer is not None:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        descriptions = [t["description"] + " " + t["topic"] for t in CURRICULUM]
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        self._tfidf_matrix = self._vectorizer.fit_transform(descriptions)
        logger.info("TF-IDF matrix built for %d topics.", len(CURRICULUM))

    def warmup(self):
        self._build_tfidf()

    def recommend(self, data: dict) -> dict:
        self.warmup()
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        grade      = data["grade_level"]
        weak_subjs = [s.strip() for s in data.get("weak_subjects", [])]
        completed  = [t.strip().lower() for t in data.get("completed_topics", [])]
        style      = data.get("learning_style", "mixed")

        # Filter by grade (±1)
        candidates = [
            t for t in CURRICULUM
            if abs(t["grade"] - grade) <= 1
            and t["topic"].lower() not in completed
        ]

        if not candidates:
            candidates = [t for t in CURRICULUM if t["topic"].lower() not in completed]

        # Build query from weak subjects + style
        query_parts = weak_subjs + [style, "learning study practice"]
        query = " ".join(query_parts)

        query_vec = self._vectorizer.transform([query])
        cand_idx  = [CURRICULUM.index(c) for c in candidates]
        cand_matrix = self._tfidf_matrix[cand_idx]
        scores = cosine_similarity(query_vec, cand_matrix).flatten()

        # Boost weak subjects
        boosted = []
        for i, (topic, score) in enumerate(zip(candidates, scores)):
            boost = 0.2 if topic["subject"] in weak_subjs else 0.0
            boosted.append((topic, float(score) + boost))

        boosted.sort(key=lambda x: -x[1])
        top_topics = boosted[:15]

        topic_items = [
            {
                "topic": t["topic"],
                "subject": t["subject"],
                "grade": t["grade"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "similarity_score": round(s, 3),
            }
            for t, s in top_topics
        ]

        # 7-day plan
        plan_topics = [t["topic"] for t, _ in top_topics[:14]]
        daily_plan: dict[str, list[str]] = {}
        idx = 0
        for day in DAYS:
            count = 2
            daily_plan[day] = plan_topics[idx : idx + count]
            idx += count
            if idx >= len(plan_topics):
                idx = 0

        return {
            "recommended_topics": topic_items,
            "daily_study_plan": daily_plan,
            "motivational_message": random.choice(MOTIVATIONAL_MESSAGES),
            "total_topics_available": len(candidates),
        }


recommend_service = RecommendService()
