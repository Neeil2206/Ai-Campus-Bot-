import sqlite3
import json
import uuid
import datetime
from pathlib import Path
from config import Config

def get_connection():
    """Returns a SQLite connection configured for dictionary-like row access."""
    conn = sqlite3.connect(Config.DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    """Initializes the SQLite tables for conversations and messages."""
    Config.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Conversations Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Messages Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                mode TEXT DEFAULT 'general_ai',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        
        # Indexes for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC)")
        conn.commit()

def create_conversation(title="New Conversation"):
    """Creates a new conversation record and returns it."""
    init_db()
    conv_id = str(uuid.uuid4())
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, now, now)
        )
        conn.commit()
        
    return {
        "id": conv_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": []
    }

def get_conversations():
    """Retrieves all conversations sorted by latest update first."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        
        conversations = []
        for r in rows:
            # Count messages
            cursor.execute("SELECT COUNT(*) as count FROM messages WHERE conversation_id = ?", (r["id"],))
            msg_count = cursor.fetchone()["count"]
            conversations.append({
                "id": r["id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "message_count": msg_count
            })
        return conversations

def get_conversation(conv_id):
    """Retrieves a single conversation and all its messages."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?", (conv_id,))
        conv = cursor.fetchone()
        if not conv:
            return None
        
        cursor.execute(
            "SELECT id, role, content, sources, mode, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conv_id,)
        )
        msg_rows = cursor.fetchall()
        
        messages = []
        for m in msg_rows:
            sources_val = []
            if m["sources"]:
                try:
                    sources_val = json.loads(m["sources"])
                except Exception:
                    sources_val = []
            messages.append({
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "sources": sources_val,
                "mode": m["mode"] or "general_ai",
                "created_at": m["created_at"]
            })
            
        return {
            "id": conv["id"],
            "title": conv["title"],
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "messages": messages
        }

def update_conversation_title(conv_id, title):
    """Updates the title of a conversation (e.g. from the first question)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cursor.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, conv_id)
        )
        conn.commit()

def delete_conversation(conv_id):
    """Deletes a conversation and its messages cascadingly."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()
        return cursor.rowcount > 0

def clear_conversation(conv_id):
    """Deletes all messages in a conversation without removing the conversation record."""
    with get_connection() as conn:
        cursor = conn.cursor()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
        conn.commit()

def add_message(conv_id, role, content, sources=None, mode="general_ai"):
    """Adds a message to a conversation and updates the conversation timestamp."""
    init_db()
    sources_json = json.dumps(sources or [])
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Verify conversation exists; create if not
        cursor.execute("SELECT id, title FROM conversations WHERE id = ?", (conv_id,))
        conv = cursor.fetchone()
        if not conv:
            # Auto-create conversation with snippet from user message
            title = content[:35] + ("..." if len(content) > 35 else "") if role == "user" else "New Conversation"
            cursor.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, title, now, now)
            )
        else:
            # If default title and this is the first user message, update title
            if conv["title"] in ("New Conversation", "Untitled") and role == "user":
                title = content[:35] + ("..." if len(content) > 35 else "")
                cursor.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))
        
        cursor.execute(
            """INSERT INTO messages (conversation_id, role, content, sources, mode, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (conv_id, role, content, sources_json, mode, now)
        )
        msg_id = cursor.lastrowid
        cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
        conn.commit()
        
        return {
            "id": msg_id,
            "conversation_id": conv_id,
            "role": role,
            "content": content,
            "sources": sources or [],
            "mode": mode,
            "created_at": now
        }

def get_messages(conv_id):
    """Retrieves all messages for a specific conversation."""
    conv = get_conversation(conv_id)
    return conv["messages"] if conv else []
