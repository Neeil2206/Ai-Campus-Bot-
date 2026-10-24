import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory of CampusBot project
BASE_DIR = Path(__file__).resolve().parent

# Load .env file
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE, override=True)

class Config:
    """Central configuration for CampusBot application."""
    
    # Project paths
    BASE_DIR = BASE_DIR
    DATA_DIR = BASE_DIR / "data" / "documents"
    DEMO_DATA_DIR = BASE_DIR / "data" / "demo"
    VECTOR_STORE_DIR = BASE_DIR / "vector_store"
    DATABASE_PATH = BASE_DIR / "database" / "campusbot.db"
    
    # Ensure critical directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Ollama Local AI Settings (100% Free & Local)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b").strip()
    OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text").strip()
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
    
    # RAG & Chunking Parameters
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
    TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "4"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.20"))
    
    # Flask Server Settings
    FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "campusbot_secure_secret_key_2026")
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "5000"))
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    # Allowed Document Extensions
    ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}

    # Demo-mode settings for the local CampusBot demo knowledge base.
    DEMO_MODE = os.getenv("DEMO_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}

    # The exact six clickable starter questions shown on the welcome screen.
    # These are sent to the real RAG + LLM pipeline (no hardcoded answers).
    SUGGESTED_QUESTIONS = [
        "What engineering branches are available?",
        "What is the admission process?",
        "What are the eligibility criteria?",
        "What is the fee structure?",
        "What documents are required for admission?",
        "What scholarships and financial aid are available?",
    ]

    # Canonical fallback returned when the knowledge base does not contain
    # information relevant to the student's question.
    FALLBACK_MESSAGE = (
        "I couldn't find this information in the available college documents. "
        "Please contact the college admission office to confirm the details."
    )

    # Brief label appended to grounded answers when the supporting documents are
    # illustrative/fictional demonstration data (never presented as official info).
    DEMO_SOURCE_NOTE = (
        "_Note: This answer is based on CampusBot's illustrative demonstration "
        "documents. Please confirm official details with the college admission office._"
    )

    # Legacy hardcoded demo answers are intentionally REMOVED. All six questions
    # are answered through the real retrieval + local LLM pipeline so the chatbot
    # stays document-grounded and is never a static mockup.
    DEMO_QUESTION_ANSWERS = {}

