# 🎬 EduMind AI — Demo Guide

This guide walks you through a complete demo of all three AI modules.

---

## Setup (2 minutes)

```bash
git clone https://github.com/YOUR_USERNAME/edumind-ai.git
cd edumind-ai
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > backend/.env
python data/generate_dataset.py
cd backend && uvicorn main:app --reload --port 8000
```

Open `frontend/index.html` in your browser.

---

## Module 1 — EduBot RAG Q&A

1. Download any NCERT PDF (e.g. Class 9 Science from ncert.nic.in)
2. In the **EduBot Q&A** tab, drag-and-drop the PDF
3. Wait for "X chunks indexed" confirmation
4. Ask questions:
   - *"What is photosynthesis?"*
   - *"Explain Newton's laws of motion"*
   - *"What are the products of cellular respiration?"*

**What to highlight:** confidence score, source citation, response time, Llama-3 model label.

---

## Module 2 — Performance Predictor

Switch to the **Performance Predictor** tab.

**Demo scenario — At-risk student:**
- Attendance: 55%
- Study Hours: 1.5
- Previous Score: 42
- Assignments: 45%
- Sleep: 5 hrs
- Extracurricular: 4

Expected output: Grade D/F, High risk, multiple recommendations.

**Demo scenario — Strong student:**
- Attendance: 95%
- Study Hours: 5
- Previous Score: 88
- Assignments: 95%
- Sleep: 8 hrs
- Extracurricular: 1

Expected output: Grade A/A+, Low risk.

---

## Module 3 — Topic Recommender

Switch to the **Topic Recommender** tab.

**Demo config:**
- Student ID: STU001
- Grade: 9
- Weak Subjects: Math, Science
- Completed Topics: Integers, Fractions
- Learning Style: Visual

Expected output: 12 topic cards prioritising Math & Science, complete 7-day study plan.

---

## API Exploration

Visit **http://localhost:8000/docs** for interactive Swagger UI.

Try the endpoints directly:
```bash
# Health check
curl http://localhost:8000/health

# Performance prediction
curl -X POST http://localhost:8000/api/predict/performance \
  -H "Content-Type: application/json" \
  -d '{"attendance_pct":85,"study_hours_per_day":3.5,"prev_exam_score":72,"assignments_completed_pct":80,"sleep_hours":7,"extracurricular_activities":2,"subject":"Math"}'
```
