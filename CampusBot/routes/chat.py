import sys
import traceback
from flask import Blueprint, request, jsonify
from database.db import create_conversation
from services.rag_service import generate_chat_response

chat_bp = Blueprint("chat", __name__)

MAX_MESSAGE_LENGTH = 3000

@chat_bp.route("/api/chat", methods=["POST"])
def chat_endpoint():
    """
    Main conversational endpoint:
    - Accepts student message and optional conversation_id
    - Validates inputs
    - Retrieves relevant document passages if knowledge base is populated
    - Performs local LLM generation via Ollama (llama3.2:3b)
    - Returns structured JSON response with citations and execution mode
    """
    # 1. Validate JSON payload
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request must be JSON with 'message' field.",
            "answer": "Invalid request format.",
            "sources": [],
            "mode": "error"
        }), 400

    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Invalid JSON body provided.",
            "answer": "Invalid request body.",
            "sources": [],
            "mode": "error"
        }), 400

    user_message = data.get("message")
    if user_message is None or not isinstance(user_message, str):
        return jsonify({
            "success": False,
            "error": "Field 'message' is required and must be a string.",
            "answer": "Please provide a valid question.",
            "sources": [],
            "mode": "error"
        }), 400

    clean_message = user_message.strip()
    if not clean_message:
        return jsonify({
            "success": False,
            "error": "Question cannot be blank.",
            "answer": "Please enter a question to get started.",
            "sources": [],
            "mode": "error"
        }), 400

    if len(clean_message) > MAX_MESSAGE_LENGTH:
        return jsonify({
            "success": False,
            "error": f"Message exceeds maximum character length of {MAX_MESSAGE_LENGTH}.",
            "answer": f"Your question is too long. Please shorten it to under {MAX_MESSAGE_LENGTH} characters.",
            "sources": [],
            "mode": "error"
        }), 400

    conversation_id = data.get("conversation_id")
    if conversation_id is not None and not isinstance(conversation_id, str):
        conversation_id = None

    if not conversation_id:
        conversation_id = create_conversation(title=clean_message[:60])['id']

    try:
        # 2. Invoke RAG Generation Pipeline
        result = generate_chat_response(clean_message, conversation_id=conversation_id)
        
        # Ensure consistent schema
        status_code = 200
        return jsonify({
            "success": result.get("success", False),
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "mode": result.get("mode", "general_ai"),
            "conversation_id": result.get("conversation_id", conversation_id),
            "error": result.get("error", None)
        }), status_code

    except Exception as e:
        print(f"[CampusBot Chat Error] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return jsonify({
            "success": False,
            "answer": "CampusBot encountered an unexpected error while processing your request. Please try again.",
            "sources": [],
            "mode": "error",
            "conversation_id": conversation_id,
            "error": f"{type(e).__name__}: {str(e)}"
        }), 500
