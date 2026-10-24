import socket
import requests
from urllib.parse import urlparse
from config import Config

class OllamaOfflineError(Exception):
    """Raised when Ollama local service is not reachable."""
    pass

class OllamaModelNotFoundError(Exception):
    """Raised when the specified model has not been pulled in Ollama."""
    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__(
            f"The local AI model '{model_name}' is not installed in Ollama. "
            f"Please download it by opening Command Prompt or PowerShell and running: ollama pull {model_name}"
        )

def check_ollama_reachable(timeout=0.8):
    """
    Checks if Ollama service is running locally on Config.OLLAMA_BASE_URL.
    Uses an ultra-fast socket probe so calls never block when Ollama is offline.
    """
    try:
        parsed = urlparse(Config.OLLAMA_BASE_URL)
        host = parsed.hostname or "127.0.0.1"
        if host == "localhost":
            host = "127.0.0.1"
        port = parsed.port or 11434
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        res = sock.connect_ex((host, port))
        sock.close()
        if res != 0:
            return False
            
        response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False

def get_available_models(timeout=2.0):
    """
    Fetches the list of locally pulled models from Ollama /api/tags.
    Returns list of model name strings (e.g. ['llama3.2:3b', 'nomic-embed-text:latest']).
    """
    if not check_ollama_reachable():
        return []
    try:
        url = f"{Config.OLLAMA_BASE_URL}/api/tags"
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            return [m for m in models if m]
    except Exception:
        pass
    return []

def is_model_available(model_name, available_models=None):
    """
    Checks if a particular model (or tag variation) is in the available models list.
    Handles matching both 'llama3.2:3b' and 'llama3.2:3b:latest' or 'llama3.2'.
    """
    if available_models is None:
        available_models = get_available_models()
        
    model_name_clean = model_name.split(":")[0].lower()
    for m in available_models:
        m_lower = m.lower()
        if model_name.lower() in m_lower or m_lower.startswith(model_name_clean):
            return True
    return False

def get_ai_status():
    """
    Gathers detailed status of local Ollama AI engine.
    Returns a dictionary with connectivity, models, and exact recommendations.
    """
    reachable = check_ollama_reachable()
    if not reachable:
        return {
            "ollama_online": False,
            "chat_model": Config.OLLAMA_CHAT_MODEL,
            "chat_model_available": False,
            "embed_model": Config.OLLAMA_EMBED_MODEL,
            "embed_model_available": False,
            "available_models": [],
            "status_message": "CampusBot's local AI service is unavailable. Please start Ollama and try again.",
            "setup_command": f"ollama pull {Config.OLLAMA_CHAT_MODEL}"
        }
        
    models = get_available_models()
    chat_ok = is_model_available(Config.OLLAMA_CHAT_MODEL, models)
    embed_ok = is_model_available(Config.OLLAMA_EMBED_MODEL, models)
    
    if not chat_ok:
        msg = f"Chat model '{Config.OLLAMA_CHAT_MODEL}' is missing. Run: ollama pull {Config.OLLAMA_CHAT_MODEL}"
    elif not embed_ok:
        msg = f"Embedding model '{Config.OLLAMA_EMBED_MODEL}' is missing. Run: ollama pull {Config.OLLAMA_EMBED_MODEL}"
    else:
        msg = "Local Ollama AI is connected and ready."
        
    return {
        "ollama_online": True,
        "chat_model": Config.OLLAMA_CHAT_MODEL,
        "chat_model_available": chat_ok,
        "embed_model": Config.OLLAMA_EMBED_MODEL,
        "embed_model_available": embed_ok,
        "available_models": models,
        "status_message": msg,
        "setup_command": f"ollama pull {Config.OLLAMA_CHAT_MODEL}" if not chat_ok else (f"ollama pull {Config.OLLAMA_EMBED_MODEL}" if not embed_ok else "")
    }

def call_ollama_chat(messages, model=None, temperature=0.2, timeout=None):
    """
    Calls Ollama /api/chat with the supplied messages list.
    Validates reachability and model availability before making request.
    Raises OllamaOfflineError or OllamaModelNotFoundError with clear actionable instructions.
    """
    if model is None:
        model = Config.OLLAMA_CHAT_MODEL
    if timeout is None:
        timeout = Config.REQUEST_TIMEOUT
        
    # 1. Connectivity Check
    if not check_ollama_reachable():
        raise OllamaOfflineError(
            "CampusBot's local AI service is unavailable. Please start Ollama and try again."
        )
        
    # 2. Model Check
    available = get_available_models()
    if not is_model_available(model, available):
        raise OllamaModelNotFoundError(model)
        
    matched_model = model
    model_prefix = model.split(":")[0].lower()
    for m in available:
        if model.lower() in m.lower() or m.lower().startswith(model_prefix):
            matched_model = m
            break

    # 3. Request Execution
    url = f"{Config.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": matched_model,
        "messages": messages,
        "options": {
            "temperature": temperature
        },
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            return data.get("message", {}).get("content", "").strip()
        else:
            raise RuntimeError(f"Ollama returned HTTP error {response.status_code}: {response.text}")
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Local AI inference timed out after {timeout} seconds. Ollama model may still be loading into RAM.")
    except Exception as e:
        if isinstance(e, (OllamaOfflineError, OllamaModelNotFoundError)):
            raise e
        raise RuntimeError(f"Failed to communicate with local Ollama service: {str(e)}")
