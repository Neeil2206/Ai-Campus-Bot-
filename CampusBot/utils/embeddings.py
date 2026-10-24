import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment configuration
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE, override=True)

class Config:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()

# Lazy-loaded sentence transformer model instance
_embedding_model = None

def get_transformer_model():
    """
    Returns the pre-loaded SentenceTransformer model instance (all-MiniLM-L6-v2).
    """
    global _embedding_model
    if _embedding_model is None:
        print("Loading local SentenceTransformer model 'all-MiniLM-L6-v2'...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Local embedding model loaded successfully.")
    return _embedding_model

def get_embedding(text, model=None):
    """
    Generates a single embedding for the text using sentence-transformers.
    """
    if not text:
        return []
    
    transformer = get_transformer_model()
    # Normalize text newlines to align with standard ingestions
    cleaned_text = text.replace("\n", " ")
    vector = transformer.encode(cleaned_text)
    return vector.tolist()

def get_embeddings_batch(texts, model=None):
    """
    Generates embeddings for a list of texts using sentence-transformers.
    """
    if not texts:
        return []
    
    transformer = get_transformer_model()
    cleaned_texts = [text.replace("\n", " ") for text in texts]
    vectors = transformer.encode(cleaned_texts)
    return [v.tolist() for v in vectors]
