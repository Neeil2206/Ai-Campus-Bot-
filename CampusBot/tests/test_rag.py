import shutil
import unittest
from unittest import mock
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import Config
from services.document_processor import clean_text, split_text_into_chunks, process_file_to_chunks
from services.embedding_service import (
    get_embedding,
    get_embeddings_batch,
    TfidfEmbedder,
    EMBEDDING_SIGNATURE,
)
from services import rag_service
from services.rag_service import (
    PersistentVectorStore,
    retrieve_relevant_passages,
    compress_grounded_answer,
    deduplicate_sources,
    build_strict_grounded_answer,
    add_demo_label,
    is_demo_document,
    generate_chat_response,
    ensure_index_current,
    rebuild_index_from_documents,
    vector_store,
)

# The exact six clickable starter questions required on the welcome screen.
EXPECTED_DEMO_QUESTIONS = [
    "What engineering branches are available?",
    "What is the admission process?",
    "What are the eligibility criteria?",
    "What is the fee structure?",
    "What documents are required for admission?",
    "What scholarships and financial aid are available?",
]


class CampusBotRAGTestCase(unittest.TestCase):
    def test_clean_text(self):
        """Verify text normalization removes extra spaces and carriage returns."""
        dirty = "CampusBot\r\n\r\nAdmission   Assistant\t\tGuidance\n\n\n\nDetails"
        cleaned = clean_text(dirty)
        self.assertNotIn("\r", cleaned)
        self.assertNotIn("   ", cleaned)
        self.assertIn("CampusBot\n\nAdmission Assistant Guidance\n\nDetails", cleaned)

    def test_split_text_into_chunks(self):
        """Verify semantic text chunking with overlap."""
        sample_text = (
            "Paragraph One: CampusBot helps students understand college admission procedures.\n\n"
            "Paragraph Two: Official document verification requires 10th and 12th marksheets.\n\n"
            "Paragraph Three: Engineering programs include Computer Science, Electronics, and Mechanical."
        )
        chunks = split_text_into_chunks(sample_text, chunk_size=120, overlap=30)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIsInstance(chunk, str)
            self.assertGreater(len(chunk), 0)

    def test_demo_document_processing(self):
        """Verify the demo knowledge document produces structured chunks with metadata."""
        demo_path = Config.DEMO_DATA_DIR / "demo_admission_knowledge.txt"
        self.assertTrue(demo_path.exists(), "demo_admission_knowledge.txt must exist")

        chunks = process_file_to_chunks(demo_path, chunk_size=500, overlap=100)
        self.assertGreater(len(chunks), 0)

        first_chunk = chunks[0]
        self.assertIn("id", first_chunk)
        self.assertIn("text", first_chunk)
        self.assertIn("metadata", first_chunk)
        self.assertEqual(first_chunk["metadata"]["document_name"], "demo_admission_knowledge.txt")

    def test_required_demo_questions_are_configured(self):
        """Verify the six starter questions match the required prompts exactly."""
        self.assertEqual(Config.SUGGESTED_QUESTIONS, EXPECTED_DEMO_QUESTIONS)
        self.assertEqual(len(Config.SUGGESTED_QUESTIONS), 6)
        self.assertTrue(Config.DEMO_MODE)

    def test_no_hardcoded_demo_answers(self):
        """The chatbot must NOT serve static hardcoded topic answers for the six questions."""
        self.assertEqual(getattr(Config, "DEMO_QUESTION_ANSWERS", {}), {})
        for question in EXPECTED_DEMO_QUESTIONS:
            answer = build_strict_grounded_answer(question, [])
            # With no supporting evidence, the strict builder must either defer to
            # the real LLM pipeline (None) or return the canonical fallback — it
            # must NEVER return a canned, pre-written topic answer.
            self.assertTrue(answer is None or answer == Config.FALLBACK_MESSAGE)

    def test_demo_knowledge_base_files_and_labels(self):
        """Verify all required demo files exist and clearly state they are fictional examples."""
        required_files = [
            "demo_admission_faq.txt",
            "demo_courses_and_eligibility.txt",
            "demo_fee_structure.txt",
            "demo_scholarships.txt",
            "demo_admission_process.txt",
            "demo_campus_facilities.txt",
        ]
        for file_name in required_files:
            path = Config.DATA_DIR / file_name
            self.assertTrue(path.exists(), f"{file_name} must exist")
            text = path.read_text(encoding="utf-8")
            self.assertIn("DEMO DATA", text.upper())
            self.assertIn("FICTIONAL", text.upper())

    def test_tfidf_embedder_is_deterministic_and_shared(self):
        """The TF-IDF embedder produces identical vectors for identical input."""
        emb = TfidfEmbedder()
        emb.fit(["admission eligibility criteria", "fee structure scholarship"])
        v1 = emb.embed("What is the fee structure?")
        v2 = emb.embed("What is the fee structure?")
        self.assertEqual(v1, v2)
        self.assertAlmostEqual(sum(x * x for x in v1) ** 0.5, 1.0, places=4)

    def test_tfidf_rejects_unrelated_queries(self):
        """Unrelated queries share (almost) no vocabulary and score near zero."""
        emb = TfidfEmbedder()
        docs = [
            "The fee structure includes tuition fee and examination fee.",
            "Eligibility criteria require Physics and Mathematics in class 12.",
            "Scholarships include merit based and need based financial aid.",
        ]
        emb.fit(docs)
        fee_vec = emb.embed("What is the fee structure?")
        mars_vec = emb.embed("What is the weather on Mars?")
        fee_doc = emb.embed(docs[0])

        def cosine(a, b):
            return sum(x * y for x, y in zip(a, b))

        self.assertGreater(cosine(fee_vec, fee_doc), cosine(mars_vec, fee_doc))
        self.assertEqual(sum(abs(x) for x in mars_vec), 0.0)

    def test_local_embeddings_generation(self):
        """Verify embedding service outputs a valid vector."""
        vec = get_embedding("What scholarships are available?")
        self.assertIsInstance(vec, list)
        self.assertGreater(len(vec), 0)


    def test_application_fee_is_not_invented(self):
        """Application-fee questions with no amount in evidence use the canonical fallback."""
        passages = [
            {"document_name": "demo_admission_faq.txt", "page": 1,
             "text": "Fees are accepted through the official online ERP portal via Net Banking."},
        ]
        answer = build_strict_grounded_answer("What is the application fee?", passages)
        self.assertIsNotNone(answer)
        self.assertEqual(answer, Config.FALLBACK_MESSAGE)

    def test_build_strict_answer_returns_none_for_grounded_topics(self):
        """Topic questions WITH supporting evidence fall through to the LLM path (return None)."""
        passages = [{"document_name": "demo_fee_structure.txt", "page": 1,
                     "text": "Tuition fee is 90,000 rupees per year."}]
        self.assertIsNone(build_strict_grounded_answer("What is the fee structure?", passages))

    def test_vector_search_and_retrieval(self):
        """Verify TF-IDF indexing and query retrieval on an admission checklist."""
        store = PersistentVectorStore(store_dir=Config.BASE_DIR / "tests" / "temp_store")
        store.clear()

        sample_chunks = [
            {
                "id": "c1",
                "text": "The admission documents checklist requires 10th marksheet, 12th certificate, and Transfer Certificate.",
                "metadata": {"document_name": "test_docs.txt", "page": 1}
            },
            {
                "id": "c2",
                "text": "Campus sports facilities include a gymnasium, football field, and basketball courts.",
                "metadata": {"document_name": "test_sports.txt", "page": 1}
            }
        ]
        store.embedder.fit([c["text"] for c in sample_chunks])
        embeddings = [store.embedder.embed(c["text"]) for c in sample_chunks]
        store.add_chunks(sample_chunks, embeddings)

        query = "What documents do I need to submit for admission?"
        query_vec = store.encode_query(query)
        hits = store.search(query_vec, top_k=1, min_similarity=0.01)

        self.assertGreater(len(hits), 0)
        top_chunk, score = hits[0]
        self.assertEqual(top_chunk["id"], "c1")
        self.assertIn("marksheet", top_chunk["text"])

        store.clear()
        temp_dir = Config.BASE_DIR / "tests" / "temp_store"
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_grounded_answer_preserves_structure_and_strips_citations(self):
        """compress_grounded_answer removes inline citations but keeps tables/lists intact."""
        raw_answer = (
            "**Required Documents**\n\n"
            "| Document | Required |\n| --- | --- |\n| 10th marksheet | Yes |\n| 12th certificate | Yes |\n\n"
            "Steps:\n\n"
            "1. Submit the 10th marksheet. [Source 1: demo_admission_faq.txt, Page 1]\n"
            "2. Submit the 12th certificate.\n"
            "3. Carry ID proof. (demo_admission_faq.txt, Page 1)"
        )
        passages = [
            {"document_name": "demo_admission_faq.txt", "page": 1, "similarity": 0.85, "snippet": "10th marksheet"},
            {"document_name": "demo_admission_faq.txt", "page": 1, "similarity": 0.81, "snippet": "Transfer certificate"},
        ]

        cleaned = compress_grounded_answer(raw_answer, passages)
        self.assertNotIn("[Source 1", cleaned)
        self.assertNotIn("(demo_admission_faq.txt", cleaned)
        self.assertIn("| Document | Required |", cleaned)
        self.assertIn("1. Submit the 10th marksheet.", cleaned)
        self.assertIn("3. Carry ID proof.", cleaned)

        deduped = deduplicate_sources(passages)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["document_name"], "demo_admission_faq.txt")

    def test_demo_label_applied_only_for_demo_sources(self):
        """Demo-grounded answers get an illustrative note; official sources do not."""
        demo_passages = [{"document_name": "demo_fee_structure.txt", "page": 1}]
        official_passages = [{"document_name": "official_policy_2026.pdf", "page": 1}]

        demo_answer = add_demo_label("The tuition fee is 90,000.", demo_passages)
        official_answer = add_demo_label("The tuition fee is 90,000.", official_passages)

        self.assertIn(Config.DEMO_SOURCE_NOTE, demo_answer)
        self.assertNotIn(Config.DEMO_SOURCE_NOTE, official_answer)

    def test_retrieval_returns_relevant_passages_for_all_six_questions(self):
        """All six starter questions must retrieve relevant chunks from the real index."""
        ensure_index_current()
        self.assertTrue(vector_store.is_signature_current())
        for question in EXPECTED_DEMO_QUESTIONS:
            passages = retrieve_relevant_passages(question)
            self.assertGreater(len(passages), 0, f"Expected relevant passages for: {question}")

    def test_retrieval_rejects_unrelated_question(self):
        """A completely unrelated question must retrieve no passages (no hallucination source)."""
        ensure_index_current()
        passages = retrieve_relevant_passages("What is the weather on Mars?")
        self.assertEqual(passages, [])


class CampusBotGenerationTestCase(unittest.TestCase):
    """Generation-path tests using a mocked Ollama call (no real model needed)."""

    def setUp(self):
        ensure_index_current()

    @mock.patch("services.rag_service.call_ollama_chat")
    def test_grounded_answer_uses_retrieved_context(self, mock_chat):
        """A documented question returns a grounded answer and real sources."""
        mock_chat.return_value = (
            "**Engineering Branches**\n\n"
            "1. **Computer Science and Engineering (CSE)**\n"
            "2. **Information Technology (IT)**\n"
            "3. **Mechanical Engineering (ME)**"
        )
        result = generate_chat_response("What engineering branches are available?", conversation_id=None)
        self.assertTrue(result["success"])
        self.assertEqual(result["mode"], "rag")
        self.assertGreater(len(result["sources"]), 0)
        self.assertIn("Computer Science", result["answer"])
        self.assertTrue(mock_chat.called)

    @mock.patch("services.rag_service.call_ollama_chat")
    def test_unrelated_question_does_not_invent_answer(self, mock_chat):
        """An unrelated question yields the canonical fallback, not fabricated facts."""
        mock_chat.return_value = Config.FALLBACK_MESSAGE
        result = generate_chat_response("What is the weather on Mars?", conversation_id=None)
        self.assertTrue(result["success"])
        self.assertEqual(result["mode"], "general_ai")
        self.assertIn("couldn't find this information", result["answer"].lower())

    @mock.patch("services.rag_service.call_ollama_chat")
    def test_empty_answer_falls_back(self, mock_chat):
        """An empty model response is replaced with the canonical fallback."""
        mock_chat.return_value = "   "
        result = generate_chat_response("What documents are required for admission?", conversation_id=None)
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], Config.FALLBACK_MESSAGE)

    @mock.patch("services.rag_service.check_ollama_reachable", return_value=False)
    def test_ollama_offline_returns_friendly_message(self, _mock_reachable):
        """When Ollama is offline, a friendly message is returned without crashing."""
        result = generate_chat_response("What is the admission process?", conversation_id=None)
        self.assertFalse(result["success"])
        self.assertEqual(result["mode"], "offline")
        self.assertIn("Ollama", result["answer"])


class CampusBotIndexRebuildTestCase(unittest.TestCase):
    """Ingestion/reindex behavior: repeatable, idempotent, no duplicate records."""

    def test_rebuild_is_idempotent(self):
        """Rebuilding twice yields the same chunk count (no duplicates)."""
        rebuild_index_from_documents()
        stats1 = vector_store.get_stats()
        rebuild_index_from_documents()
        stats2 = vector_store.get_stats()

        self.assertEqual(stats1["total_chunks"], stats2["total_chunks"])
        self.assertEqual(stats1["total_documents"], stats2["total_documents"])
        self.assertTrue(vector_store.is_signature_current())
        self.assertGreater(stats1["total_chunks"], 0)


if __name__ == '__main__':
    unittest.main()

