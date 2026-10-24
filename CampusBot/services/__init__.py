# Services package
from .ollama_service import (
    check_ollama_reachable,
    get_available_models,
    is_model_available,
    get_ai_status,
    call_ollama_chat,
    OllamaOfflineError,
    OllamaModelNotFoundError
)
