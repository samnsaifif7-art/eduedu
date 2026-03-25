"""
Generate synthetic student performance dataset.
Produces data/sample_students.csv with 500 realistic records.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

SUBJECTS = ["Math", "Science", "English", "Social Science", "Hindi"]
RANDOM_STATE = 42
N = 500


def main():
    rng = np.random.default_rng(RANDOM_STATE)

    attendance       = rng.uniform(40, 100, N).round(1)
    study_hours      = rng.uniform(0.5, 8,  N).round(1)
    prev_score       = rng.uniform(30, 100, N).round(1)
    assignments      = rng.uniform(30, 100, N).round(1)
    sleep_hours      = rng.uniform(4,  10,  N).round(1)
    extracurricular  = rng.integers(0, 6,   N)
    subjects         = rng.choice(SUBJECTS, N)

    base_score = (
        0.28 * prev_score
        + 0.22 * attendance
        + 0.18 * study_hours * 10
        + 0.15 * assignments
        + 0.07 * sleep_hours * 5
        - 0.03 * extracurricular * 2
        + rng.normal(0, 6, N)
    )
    final_score = np.clip(base_score, 0, 100).round(1)

    bins   = [0, 40, 50, 60, 70, 80, 90, 101]
    labels = ["F", "D", "C", "B-", "B", "A-", "A+"]
    grades = pd.cut(final_score, bins=bins, labels=labels, right=False)

    student_ids = [f"STU{str(i+1).zfill(4)}" for i in range(N)]

    df = pd.DataFrame({
        "student_id":                student_ids,
        "subject":                   subjects,
        "attendance_pct":            attendance,
        "study_hours_per_day":       study_hours,
        "prev_exam_score":           prev_score,
        "assignments_completed_pct": assignments,
        "sleep_hours":               sleep_hours,
        "extracurricular_activities":extracurricular,
        "final_score":               final_score,
        "grade":                     grades,
    })

    out = Path(__file__).parent / "sample_students.csv"
    df.to_csv(out, index=False)
    print(f"✅ Generated {N} student records → {out}")
    print(df["grade"].value_counts().sort_index())


if __name__ == "__main__":
    main()
