import unittest
import json
import sys
from pathlib import Path
from unittest import mock

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import app
from database.db import create_conversation, get_conversation

class CampusBotRoutesTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_landing_page_loads(self):
        """Verify homepage loads successfully with HTTP 200."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Your College Journey", response.data)
        self.assertIn(b"Starts Here", response.data)

    def test_chat_page_loads(self):
        """Verify dedicated chat interface loads successfully with HTTP 200."""
        response = self.client.get('/chat')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"CampusBot", response.data)
        self.assertIn(b"New Conversation", response.data)
        self.assertIn(b"What documents are required for admission?", response.data)

    def test_chat_page_shows_exactly_six_suggested_questions(self):
        """The six required starter questions are all rendered on the chat page."""
        response = self.client.get('/chat')
        self.assertEqual(response.status_code, 200)
        required = [
            b"What engineering branches are available?",
            b"What is the admission process?",
            b"What are the eligibility criteria?",
            b"What is the fee structure?",
            b"What documents are required for admission?",
            b"What scholarships and financial aid are available?",
        ]
        for question in required:
            self.assertIn(question, response.data)

    def test_admin_page_loads(self):
        """Verify knowledge base admin page loads with HTTP 200."""
        response = self.client.get('/admin')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Knowledge Base", response.data)

    def test_status_endpoint(self):
        """Verify GET /api/status returns valid system status schema."""
        response = self.client.get('/api/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("status", data)
        self.assertIn("ollama_online", data)
        self.assertIn("chat_model", data)
        self.assertIn("vector_store", data)
        self.assertIsInstance(data["vector_store"], dict)

    def test_chat_validation_empty_body(self):
        """Verify POST /api/chat rejects empty body with 400."""
        response = self.client.post('/api/chat', data=json.dumps({}), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data.get("success", True))

    def test_chat_validation_blank_message(self):
        """Verify POST /api/chat rejects whitespace message with 400."""
        response = self.client.post(
            '/api/chat',
            data=json.dumps({"message": "   "}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_conversation_lifecycle(self):
        """Verify SQLite conversation creation, listing, retrieval, clearing, and deletion."""
        # 1. Create
        create_res = self.client.post(
            '/api/conversations',
            data=json.dumps({"title": "Test Admission Inquiry"}),
            content_type='application/json'
        )
        self.assertEqual(create_res.status_code, 201)
        conv = json.loads(create_res.data)
        conv_id = conv["id"]
        self.assertEqual(conv["title"], "Test Admission Inquiry")

        # 2. List
        list_res = self.client.get('/api/conversations')
        self.assertEqual(list_res.status_code, 200)
        convs = json.loads(list_res.data).get("conversations", [])
        self.assertTrue(any(c["id"] == conv_id for c in convs))

        # 3. Retrieve
        get_res = self.client.get(f'/api/conversations/{conv_id}')
        self.assertEqual(get_res.status_code, 200)
        conv_data = json.loads(get_res.data)
        self.assertEqual(conv_data["id"], conv_id)

        # 4. Clear
        clear_res = self.client.post(f'/api/conversations/{conv_id}/clear')
        self.assertEqual(clear_res.status_code, 200)

        # 5. Delete
        del_res = self.client.delete(f'/api/conversations/{conv_id}')
        self.assertEqual(del_res.status_code, 200)

    def test_conversation_messages_do_not_leak_between_conversations(self):
        """Messages from one conversation must never appear in another."""
        from database.db import add_message
        conv_a = create_conversation(title="Conversation A")
        conv_b = create_conversation(title="Conversation B")
        add_message(conv_a["id"], "user", "Question only in A", mode="rag")

        get_a = self.client.get(f'/api/conversations/{conv_a["id"]}')
        get_b = self.client.get(f'/api/conversations/{conv_b["id"]}')
        self.assertEqual(get_a.status_code, 200)
        self.assertEqual(get_b.status_code, 200)

        msgs_a = json.loads(get_a.data).get("messages", [])
        msgs_b = json.loads(get_b.data).get("messages", [])
        self.assertTrue(any("only in A" in m["content"] for m in msgs_a))
        self.assertFalse(any("only in A" in m["content"] for m in msgs_b))

        # Cleanup
        self.client.delete(f'/api/conversations/{conv_a["id"]}')
        self.client.delete(f'/api/conversations/{conv_b["id"]}')

    @mock.patch("routes.chat.generate_chat_response")
    def test_chat_endpoint_returns_mocked_grounded_answer(self, mock_gen):
        """POST /api/chat returns the RAG pipeline result (model mocked for tests)."""
        mock_gen.return_value = {
            "success": True,
            "answer": "**Fee Structure**\n\n| Component | Amount |\n| --- | --- |\n| Tuition | 90,000 |",
            "sources": [{"document_name": "demo_fee_structure.txt", "page": 1, "similarity": 0.9, "snippet": "..."}],
            "mode": "rag",
            "conversation_id": None,
            "error": None,
        }
        response = self.client.post(
            '/api/chat',
            data=json.dumps({"message": "What is the fee structure?"}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("mode"), "rag")
        self.assertIn("Fee Structure", data.get("answer", ""))
        self.assertGreater(len(data.get("sources", [])), 0)
        self.assertTrue(mock_gen.called)

    def test_chat_rejects_empty_message(self):
        """POST /api/chat rejects a whitespace-only message with 400."""
        response = self.client.post(
            '/api/chat',
            data=json.dumps({"message": "   "}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data.get("success", True))


if __name__ == '__main__':
    unittest.main()
