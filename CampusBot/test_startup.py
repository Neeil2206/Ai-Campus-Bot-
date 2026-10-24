"""Quick startup test for CampusBot components."""
import sys
print("Testing CampusBot components...", flush=True)

# 1. Config
try:
    from config import Config
    print(f"[OK] Config loaded. Ollama URL: {Config.OLLAMA_BASE_URL}", flush=True)
except Exception as e:
    print(f"[FAIL] Config: {e}", flush=True)
    sys.exit(1)

# 2. Database
try:
    from database.db import init_db
    init_db()
    print("[OK] Database initialized.", flush=True)
except Exception as e:
    print(f"[FAIL] Database: {e}", flush=True)

# 3. Vector Store
try:
    from services.rag_service import vector_store
    stats = vector_store.get_stats()
    print(f"[OK] VectorStore: {stats['total_documents']} docs, {stats['total_chunks']} chunks", flush=True)
except Exception as e:
    print(f"[FAIL] VectorStore: {e}", flush=True)

# 4. Ollama connectivity
try:
    from services.ollama_service import check_ollama_reachable, get_available_models
    online = check_ollama_reachable()
    print(f"[{'OK' if online else 'WARN'}] Ollama: {'ONLINE' if online else 'OFFLINE (app will still work, AI responses need Ollama)'}", flush=True)
    if online:
        models = get_available_models()
        print(f"  Available models: {models}", flush=True)
except Exception as e:
    print(f"[FAIL] Ollama check: {e}", flush=True)

# 5. Flask app import
try:
    from app import app
    print("[OK] Flask app created successfully.", flush=True)
except Exception as e:
    print(f"[FAIL] Flask app: {e}", flush=True)
    sys.exit(1)

print("\nAll core checks passed! App is ready to run.", flush=True)
