from flask import Blueprint, jsonify
from config import Config
from services.ollama_service import get_ai_status
from services.rag_service import vector_store

status_bp = Blueprint("status", __name__)

@status_bp.route("/api/status", methods=["GET"])
def get_system_status():
    """
    Returns verified, real-time status of the local AI pipeline:
    - Flask backend connectivity
    - Ollama server reachability (http://localhost:11434)
    - Chat model availability (llama3.2:3b)
    - Embedding model availability (nomic-embed-text)
    - Vector database index status and document/chunk counts
    """
    try:
        ai_status = get_ai_status()
        vector_stats = vector_store.get_stats()
        
        has_index = vector_stats.get("total_chunks", 0) > 0
        
        return jsonify({
            "status": "online",
            "backend": "Flask",
            "ollama_online": ai_status.get("ollama_online", False),
            "chat_model": ai_status.get("chat_model", Config.OLLAMA_CHAT_MODEL),
            "chat_model_available": ai_status.get("chat_model_available", False),
            "embed_model": ai_status.get("embed_model", Config.OLLAMA_EMBED_MODEL),
            "embed_model_available": ai_status.get("embed_model_available", False),
            "ai_available": ai_status.get("ollama_online", False) and ai_status.get("chat_model_available", False),
            "vector_store": {
                "status": "connected" if has_index else "empty",
                "documents_count": vector_stats.get("total_documents", 0),
                "chunks_count": vector_stats.get("total_chunks", 0),
                "documents": vector_stats.get("documents", [])
            },
            "available_models": ai_status.get("available_models", []),
            "message": ai_status.get("status_message", ""),
            "setup_command": ai_status.get("setup_command", "")
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "backend": "Flask",
            "ollama_online": False,
            "ai_available": False,
            "error": "Failed to query system status.",
            "message": "CampusBot's local AI service is unavailable. Please start Ollama and try again."
        }), 500
