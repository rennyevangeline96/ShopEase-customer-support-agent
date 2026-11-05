"""Central settings: paths, model names and thresholds. Secrets come from .env only."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Paths
BACKEND_DIR = BASE_DIR / "backend"
CATALOG_DIR = BACKEND_DIR / "catalog"            # RAG 1 source (Query Agent)
KB_DIR = BACKEND_DIR / "knowledge_base"          # RAG 2 source (Knowledge Agent)
INDEX_DIR = BACKEND_DIR / "faiss_index"          # generated, git-ignored
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "tickets.csv"

# Models (key is read from .env, never hard-coded)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Retrieval: best match below this cosine score counts as "no matching document".
# 0.30 is a starting value; we calibrate it with the 6 test complaints.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.30"))
KEYWORD_THRESHOLD = 0.40   # only used if the embedding API is unavailable

# Backend API. Leave BACKEND_URL empty and `streamlit run app.py` starts the FastAPI backend
# itself on BACKEND_PORT. Set BACKEND_URL to use a backend that is already running elsewhere.
BACKEND_URL = os.getenv("BACKEND_URL", "").strip().rstrip("/")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

# Workflow
CONFIDENCE_THRESHOLD = 70  # Resolution confidence below this -> escalate
MAX_MESSAGES = 6           # safety net for the team's termination condition