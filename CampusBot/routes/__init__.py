"""
CampusBot Routes Package
Modular Flask Blueprints for chat, status, conversations, and document management.
"""

from .chat import chat_bp
from .status import status_bp
from .conversations import conversations_bp
from .documents import documents_bp

__all__ = ["chat_bp", "status_bp", "conversations_bp", "documents_bp"]
