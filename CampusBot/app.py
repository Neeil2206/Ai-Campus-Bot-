import os
import sys
from pathlib import Path
from flask import Flask, render_template, jsonify, request

from config import Config
from database.db import init_db
from services.ollama_service import check_ollama_reachable, is_model_available, get_available_models
from services.rag_service import vector_store, ensure_index_current

# Import modular blueprints
from routes.chat import chat_bp
from routes.status import status_bp
from routes.conversations import conversations_bp
from routes.documents import documents_bp

# Initialize Flask application
app = Flask(__name__)
app.secret_key = Config.FLASK_SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload limit

# Register Blueprints
app.register_blueprint(status_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(conversations_bp)
app.register_blueprint(documents_bp)

# Initialize local SQLite database
init_db()

# Ensure the vector index matches the active embedding strategy and on-disk
# documents. This is a one-time, guarded rebuild (only when the stored index is
# stale or empty); it does NOT rebuild the knowledge base on every startup.
ensure_index_current()

@app.route("/")
def index():
    """Renders the comprehensive CampusBot Landing Page."""
    return render_template("index.html")

@app.route("/chat")
def chat_page():
    """Renders the dedicated, full-featured CampusBot Chatbot Interface."""
    return render_template("chat.html", suggested_questions=Config.SUGGESTED_QUESTIONS)

@app.route("/admin")
def admin_page():
    """Renders the document ingestion and administrative dashboard."""
    return render_template("admin.html")

# Global Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Resource not found", "status": 404}), 404
    return render_template("index.html"), 404

@app.errorhandler(500)
def internal_error(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error", "status": 500}), 500
    return render_template("index.html"), 500

def print_startup_banner():
    """Prints diagnostic information to console when server starts."""
    ollama_online = check_ollama_reachable()
    models = get_available_models() if ollama_online else []
    chat_model_ready = is_model_available(Config.OLLAMA_CHAT_MODEL, models) if ollama_online else False
    stats = vector_store.get_stats()

    print("=" * 65, file=sys.stderr, flush=True)
    print("🎓 CampusBot: Intelligent College Admission Assistant", file=sys.stderr, flush=True)
    print("=" * 65, file=sys.stderr, flush=True)
    print(f"Project Directory:    {Config.BASE_DIR}", file=sys.stderr, flush=True)
    print(f"Database Path:        {Config.DATABASE_PATH}", file=sys.stderr, flush=True)
    print(f"Document Storage:     {Config.DATA_DIR}", file=sys.stderr, flush=True)
    print(f"Vector Store:         {stats['total_documents']} documents, {stats['total_chunks']} chunks loaded", file=sys.stderr, flush=True)
    print(f"Retrieval:            Local TF-IDF embeddings (offline, no external embedding model required)", file=sys.stderr, flush=True)
    print(f"AI Provider:          Local Ollama (100% Free & Private)", file=sys.stderr, flush=True)
    print(f"Ollama Endpoint:      {Config.OLLAMA_BASE_URL}", file=sys.stderr, flush=True)
    print(f"Ollama Service:       {'CONNECTED [Online]' if ollama_online else 'DISCONNECTED [Offline] (Start Ollama)'}", file=sys.stderr, flush=True)
    print(f"Chat Model:           {Config.OLLAMA_CHAT_MODEL} ({'READY' if chat_model_ready else 'NOT FOUND - run `ollama pull ' + Config.OLLAMA_CHAT_MODEL + '`'})", file=sys.stderr, flush=True)
    print("=" * 65, file=sys.stderr, flush=True)
    print(f"Serving at:           http://{Config.HOST}:{Config.PORT}", file=sys.stderr, flush=True)
    print("=" * 65, file=sys.stderr, flush=True)

if __name__ == "__main__":
    print_startup_banner()
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)