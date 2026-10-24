from flask import Blueprint, request, jsonify
from database.db import (
    create_conversation,
    get_conversations,
    get_conversation,
    delete_conversation,
    clear_conversation,
    update_conversation_title
)

conversations_bp = Blueprint("conversations", __name__)

@conversations_bp.route("/api/conversations", methods=["GET"])
def list_conversations_endpoint():
    """Returns all student conversations stored in local SQLite."""
    try:
        convs = get_conversations()
        return jsonify({"conversations": convs}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve conversations: {str(e)}"}), 500

@conversations_bp.route("/api/conversations", methods=["POST"])
def create_conversation_endpoint():
    """Creates a new conversation session."""
    try:
        data = request.get_json(silent=True) or {}
        title = data.get("title", "New Conversation")
        if not isinstance(title, str) or not title.strip():
            title = "New Conversation"
        new_conv = create_conversation(title=title.strip()[:60])
        return jsonify(new_conv), 201
    except Exception as e:
        return jsonify({"error": f"Failed to create conversation: {str(e)}"}), 500

@conversations_bp.route("/api/conversations/<conv_id>", methods=["GET"])
def get_conversation_endpoint(conv_id):
    """Retrieves conversation details along with all messages."""
    try:
        conv = get_conversation(conv_id)
        if not conv:
            return jsonify({"error": "Conversation not found."}), 404
        return jsonify(conv), 200
    except Exception as e:
        return jsonify({"error": f"Failed to load conversation: {str(e)}"}), 500

@conversations_bp.route("/api/conversations/<conv_id>", methods=["DELETE"])
def delete_conversation_endpoint(conv_id):
    """Deletes the specified conversation from SQLite."""
    try:
        success = delete_conversation(conv_id)
        if success:
            return jsonify({"status": "success", "message": "Conversation deleted."}), 200
        return jsonify({"error": "Conversation not found."}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to delete conversation: {str(e)}"}), 500

@conversations_bp.route("/api/conversations/<conv_id>/clear", methods=["POST"])
def clear_conversation_endpoint(conv_id):
    """Clears all messages within a conversation while retaining the conversation thread."""
    try:
        conv = get_conversation(conv_id)
        if not conv:
            return jsonify({"error": "Conversation not found."}), 404
        clear_conversation(conv_id)
        return jsonify({"status": "success", "message": "Conversation messages cleared."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to clear conversation: {str(e)}"}), 500

@conversations_bp.route("/api/conversations/<conv_id>/title", methods=["PUT", "PATCH"])
def update_title_endpoint(conv_id):
    """Updates conversation title."""
    try:
        data = request.get_json(silent=True) or {}
        new_title = data.get("title", "").strip()
        if not new_title:
            return jsonify({"error": "Title cannot be empty."}), 400
        update_conversation_title(conv_id, new_title[:60])
        return jsonify({"status": "success", "title": new_title}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to update title: {str(e)}"}), 500
