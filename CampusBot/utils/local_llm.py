import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load configuration relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE, override=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
if OLLAMA_BASE_URL.endswith("/"):
    OLLAMA_BASE_URL = OLLAMA_BASE_URL[:-1]

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()

def get_ollama_status():
    """
    Checks if the local Ollama service is running.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        return False

def get_ollama_models():
    """
    Retrieves the list of installed models from Ollama.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []

def select_model():
    """
    Selects the configured OLLAMA_MODEL if it exists in tags, otherwise falls back
    to the first available model on the user's system.
    """
    models = get_ollama_models()
    if not models:
        return OLLAMA_MODEL
    
    # Exact or partial match
    for m in models:
        if OLLAMA_MODEL in m or m in OLLAMA_MODEL:
            return m
            
    # Default fallback to first available local model
    return models[0]

def generate_response(messages, model=None, temperature=0.2):
    """
    Sends a chat request to Ollama's api/chat endpoint.
    """
    if model is None:
        model = select_model()
        
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": temperature
        },
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
        else:
            raise RuntimeError(f"Ollama chat endpoint returned HTTP {response.status_code}: {response.text}")
    except Exception as e:
        raise RuntimeError(f"Failed to communicate with local Ollama service: {e}")
