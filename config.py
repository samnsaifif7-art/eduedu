"""
Centralized configuration for EduMind AI.
Windows-safe path handling using pathlib.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from backend directory if running from there, else root
env_path = Path(__file__).parent / "backend" / ".env"
if not env_path.exists():
    env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ── Base paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BASE_DIR / "data"

# ── Temp / model cache (OS-agnostic) ────────────────────────────────────────
import tempfile
TEMP_DIR = Path(tempfile.gettempdir())
MODEL_CACHE = TEMP_DIR / "edumind_model_cache"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

# ── Groq / LLM ──────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama3-8b-8192"

# ── RAG ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CHUNK_SIZE: int = 250          # words per chunk
CHUNK_OVERLAP: int = 50        # words overlap
TOP_K_DEFAULT: int = 3

# ── ML predictor ────────────────────────────────────────────────────────────
N_TRAIN_SAMPLES: int = 8_000
RANDOM_STATE: int = 42

# ── Recommender ─────────────────────────────────────────────────────────────
NCERT_SUBJECTS = ["Math", "Science", "English", "Social Science", "Hindi"]
GRADES = list(range(6, 11))    # 6 → 10
